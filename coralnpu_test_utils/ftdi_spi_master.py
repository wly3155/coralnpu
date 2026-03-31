#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import time
import os
import libusb_package
import usb.core
from pyftdi.ftdi import Ftdi, FtdiFeatureError
from elftools.elf.elffile import ELFFile
from coralnpu_test_utils.spi_constants import SPI_V2_OP_READ, SPI_V2_OP_WRITE, SPI_V2_BEAT_BYTES

class FtdiSpiMaster:
    """A class to manage SPI communication using an FTDI device."""

    def __init__(self, usb_serial, ftdi_port=1, csr_base_addr=0x30000):
        """Initializes the FTDI SPI master."""
        backend = libusb_package.get_libusb1_backend()
        if backend is None:
            raise RuntimeError("Could not find a USB backend even with libusb-package.")

        # We perform a dummy find using the custom backend.
        # This often "registers" the backend for the current process.
        usb.core.find(backend=backend)

        self.csr_base_addr = csr_base_addr
        # pyftdi uses ftdi://<vendor>:<product>/<serial> or ftdi://<vendor>:<product>:<index>
        url = f'ftdi://::{usb_serial}/{ftdi_port}'
        print(f"Opening FTDI device at: {url}")
        self.ftdi = Ftdi()
        self.ftdi.open_mpsse_from_url(
            url,
            direction=0x0b, # SCK, MOSI, CS# outputs
            initial=0x08,   # CS# high
            frequency=30E6)

    def close(self):
        if self.ftdi:
            self.ftdi.close()
            self.ftdi = None

    def device_reset(self):
        """Drives ADBUS7 low to reset the device, then returns it high."""
        print("Resetting device...")
        # ADBUS7 is reset, active-low.
        # Original direction: 0x0b (SCK, MOSI, CS# outputs)
        # New direction with reset: 0x8b (ADBUS7, SCK, MOSI, CS# outputs)

        # 1. Assert reset (drive ADBUS7 low)
        #    Value: 0x08 (CS# high, SCK/MOSI low, ADBUS7 low)
        self.ftdi.write_data(bytes([Ftdi.SET_BITS_LOW, 0x08, 0x8b]))
        time.sleep(0.01) # 10ms reset pulse

        # 2. De-assert reset (drive ADBUS7 high)
        #    Value: 0x88 (CS# high, SCK/MOSI low, ADBUS7 high)
        self.ftdi.write_data(bytes([Ftdi.SET_BITS_LOW, 0x88, 0x8b]))
        time.sleep(0.01)

        # 3. Restore original direction mask, keeping pins in idle state.
        self.ftdi.write_data(bytes([Ftdi.SET_BITS_LOW, 0x08, 0x0b]))
        print("Reset complete.")

    def v2_send_header(self, op, addr, length):
        """Generates the MPSSE command to send a 7-byte V2 header."""
        header = bytearray([
            op & 0xFF,
            (addr >> 24) & 0xFF,
            (addr >> 16) & 0xFF,
            (addr >> 8) & 0xFF,
            addr & 0xFF,
            (length >> 8) & 0xFF,
            length & 0xFF
        ])
        return self._get_spi_write_bytes_cmd(header)

    def read_line(self, address):
        """Reads a single 128-bit line from memory via SPI V2 protocol."""
        # V2 frame: [CS Low] -> [Header] -> [Wait for 0xFE and read 16B] -> [CS High]
        cmd = bytearray()
        cmd.extend([Ftdi.SET_BITS_LOW, 0x00, 0x0b])  # CS Low
        cmd.extend(self.v2_send_header(SPI_V2_OP_READ, address, 0))

        # Read enough bytes to cover CDC latency + 16B data (128B total for safety)
        total_read_len = 128

        # Clock out dummy zeros and read MISO. Use 0x35 for NVE out, NVE in
        read_cmd = bytearray([0x35])
        length = total_read_len - 1
        read_cmd.extend([length & 0xFF, (length >> 8) & 0xFF])
        read_cmd.extend(b'\x00' * total_read_len)
        cmd.extend(read_cmd)

        # Extra cycles for CDC drain while CS is low (16 bytes = 128 cycles)
        cmd.extend(self._get_spi_write_bytes_cmd(b'\x00' * 16))

        cmd.extend([Ftdi.SET_BITS_LOW, 0x08, 0x0b])  # CS High
        cmd.append(Ftdi.SEND_IMMEDIATE)

        self.ftdi.write_data(cmd)
        miso_bytes = self.ftdi.read_data_bytes(total_read_len, attempt=4)

        # Fast path: Byte-level search (enabled by hardware byte-alignment).
        sync_idx = miso_bytes.find(b'\xfe')
        if sync_idx != -1 and sync_idx + 1 + 16 <= len(miso_bytes):
            data_bytes = miso_bytes[sync_idx + 1 : sync_idx + 17]
            return int.from_bytes(data_bytes, 'little')

        # Fallback: Efficient bit-level search for 0xFE token (11111110) if misalignment occurs.
        # Convert byte array to a single large integer for bitwise operations.
        miso_int = int.from_bytes(miso_bytes, 'big')
        total_bits = len(miso_bytes) * 8

        # Sliding window search: mask 8 bits and compare to 0xFE.
        # We search from MSB to LSB (SPI order).
        for bit_off in range(total_bits - 8 - 128):
            shift = total_bits - 8 - bit_off
            if ((miso_int >> shift) & 0xFF) == 0xFE:
                # Found token. Extract 128 data bits following it.
                data_bits = (miso_int >> (shift - 128)) & ((1 << 128) - 1)

                # Reverse byte order: SPI shifts out MSB-first, but TileLink is little-endian.
                # data_bits currently has the first byte in the most significant 8 bits.
                result = 0
                for i in range(16):
                    byte = (data_bits >> (i * 8)) & 0xFF
                    result |= byte << ((15 - i) * 8)
                return result

        print(f"Warning: V2 start token (0xFE) not found in read from 0x{address:x}")
        return 0

    def write_lines(self, start_addr, num_beats, data_as_bytes):
        """Writes multiple 128-bit lines using V2 frame protocol."""
        if num_beats == 0:
            return 0.0, 0.0
        if len(data_as_bytes) != num_beats * 16:
            raise ValueError("Data length must be num_beats * 16")

        write_start_time = time.time()
        cmd = bytearray()
        cmd.extend([Ftdi.SET_BITS_LOW, 0x00, 0x0b])  # CS Low
        cmd.extend(self.v2_send_header(SPI_V2_OP_WRITE, start_addr, num_beats - 1))
        cmd.extend(self._get_spi_write_bytes_cmd(data_as_bytes))

        # Extra cycles for CDC drain while CS is still low
        cmd.extend(self._get_spi_write_bytes_cmd(b'\x00' * 16))

        cmd.extend([Ftdi.SET_BITS_LOW, 0x08, 0x0b])  # CS High

        cmd.append(Ftdi.GET_BITS_LOW)
        cmd.append(Ftdi.SEND_IMMEDIATE)
        self.ftdi.write_data(cmd)
        self.ftdi.read_data_bytes(1)

        write_duration = time.time() - write_start_time
        return write_duration, 0.0

    def write_line(self, address, data):
        """Writes a single 128-bit line (given as an int) to an address."""
        data_as_bytes = data.to_bytes(16, 'little')
        return self.write_lines(address, 1, data_as_bytes)

    def write_word(self, address, data):
        line_addr = (address // 16) * 16
        offset = address % 16
        line_data = self.read_line(line_addr)
        mask = 0xFFFFFFFF << (offset * 8)
        updated_data = (line_data & ~mask) | (data << (offset * 8))
        self.write_line(line_addr, updated_data)

    def load_file(self, file_path, address):
        """
        Loads an arbitrary binary file into memory at a specific address.
        Handles unaligned start and end addresses correctly.
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            file_data = f.read()

        file_size = len(file_data)
        print(f"Loading {file_size} bytes from '{os.path.basename(file_path)}' "
              f"to 0x{address:x}...")
        self.load_data(file_data, address)

    def load_data(self, data, address):
        size = len(data)
        start_address = address
        end_address = start_address + size

        total_write_duration = 0.0
        total_ack_duration = 0.0
        total_prep_duration = 0.0

        data_ptr = 0

        # 1. Handle the first line if it's unaligned or if the transfer is smaller than a full line
        start_offset = start_address % 16
        if start_offset != 0 or size < 16:
            line_addr = start_address - start_offset
            bytes_to_write = min(16 - start_offset, size)

            prep_start_time = time.time()
            data_chunk = data[data_ptr : data_ptr + bytes_to_write]
            data_ptr += bytes_to_write

            old_line_int = self.read_line(line_addr)
            old_line_bytes = old_line_int.to_bytes(16, 'little')

            new_line_bytes = bytearray(old_line_bytes)
            new_line_bytes[start_offset : start_offset + bytes_to_write] = data_chunk
            new_line_int = int.from_bytes(new_line_bytes, 'little')
            total_prep_duration += (time.time() - prep_start_time)

            write_d, ack_d = self.write_line(line_addr, new_line_int)
            total_write_duration += write_d
            total_ack_duration += ack_d

        # 2. Handle all the full, aligned lines in the middle
        loop_start_addr = (start_address + 15) & ~0xF
        loop_end_addr = end_address & ~0xF
        if loop_end_addr > loop_start_addr:
            full_lines_data_size = loop_end_addr - loop_start_addr

            # Process in 4096-byte (128-line) chunks
            for i in range(0, full_lines_data_size, 4096):
                prep_start_time = time.time()
                chunk_start_addr = loop_start_addr + i
                chunk_size = min(4096, full_lines_data_size - i)
                num_lines_in_chunk = chunk_size // 16

                data_chunk_bytes = data[data_ptr : data_ptr + chunk_size]
                data_ptr += chunk_size
                total_prep_duration += (time.time() - prep_start_time)

                if data_chunk_bytes:
                    write_d, ack_d = self.write_lines(chunk_start_addr, num_lines_in_chunk, data_chunk_bytes)
                    total_write_duration += write_d
                    total_ack_duration += ack_d

        # 3. Handle the last line if it's unaligned
        end_offset = end_address % 16
        if end_offset != 0:
            # Ensure we don't re-process the first line if the whole file fits within one
            line_addr = end_address - end_offset
            if line_addr != (start_address - start_offset):
                prep_start_time = time.time()
                bytes_to_write = end_offset
                data_chunk = data[data_ptr : data_ptr + bytes_to_write]

                old_line_int = self.read_line(line_addr)
                old_line_bytes = old_line_int.to_bytes(16, 'little')

                new_line_bytes = bytearray(old_line_bytes)
                new_line_bytes[0:bytes_to_write] = data_chunk
                new_line_int = int.from_bytes(new_line_bytes, 'little')
                total_prep_duration += (time.time() - prep_start_time)

                write_d, ack_d = self.write_line(line_addr, new_line_int)
                total_write_duration += write_d
                total_ack_duration += ack_d

        total_duration = total_write_duration + total_ack_duration + total_prep_duration
        if total_duration > 0:
            rate_kbs = (size / 1024) / total_duration
            print(f"Load complete. Transferred {size} bytes "
                  f"in {total_duration:.2f} seconds ({rate_kbs:.2f} KB/s).")
            if total_duration > 0:
                print(f"  - Breakdown: Prep: {total_prep_duration:.2f}s, "
                      f"SPI Write: {total_write_duration:.2f}s, "
                      f"ACK: {total_ack_duration:.2f}s")

    def load_elf(self, elf_file, start_core=True):
        print(f'load_elf elf_file={elf_file}')
        total_bytes_transferred = 0
        total_write_duration = 0.0
        total_prep_duration = 0.0

        with open(elf_file, 'rb') as f:
            elf_reader = ELFFile(f)
            entry_point = elf_reader.header["e_entry"]
            for segment in elf_reader.iter_segments(type="PT_LOAD"):
                paddr = segment.header.p_vaddr
                data = segment.data()
                if not data:
                    continue
                total_bytes_transferred += len(data)
                print(f'vaddr: {paddr:x} len: {len(data)}')

                for i in range(0, len(data), 256):
                    prep_start_time = time.time()
                    chunk_start_addr = paddr + i
                    chunk_size = min(256, len(data) - i)
                    num_lines_in_chunk = (chunk_size + 15) // 16

                    # Pad chunk to be a multiple of 16
                    data_chunk_bytes = bytearray(data[i : i + chunk_size])
                    while len(data_chunk_bytes) % 16 != 0:
                        data_chunk_bytes.append(0)

                    total_prep_duration += (time.time() - prep_start_time)

                    if not data_chunk_bytes:
                        continue

                    write_d, _ = self.write_lines(chunk_start_addr, num_lines_in_chunk, data_chunk_bytes)
                    total_write_duration += write_d

        total_duration = total_write_duration + total_prep_duration
        if total_duration > 0:
            rate_kbs = (total_bytes_transferred / 1024) / total_duration
            print(f"ELF data loaded. Transferred {total_bytes_transferred} bytes "
                  f"in {total_duration:.2f} seconds ({rate_kbs:.2f} KB/s).")
            print(f"  - Breakdown: Prep: {total_prep_duration:.2f}s, SPI Write: {total_write_duration:.2f}s")

        if start_core:
            self.set_entry_point(entry_point)
            self.start_core()

    def set_entry_point(self, entry_point):
        """Sets the core's entry point address."""
        print(f"Setting entry point to 0x{entry_point:x}")
        self.write_word(self.csr_base_addr + 4, entry_point)

    def start_core(self):
        """Releases the core from reset to begin execution."""
        print("Starting core...")
        self.write_word(self.csr_base_addr, 1)
        self.write_word(self.csr_base_addr, 0)

    def read_word(self, address):
        """Reads a single 32-bit word from a given address."""
        line_addr = (address // 16) * 16
        offset = address % 16
        line_data = self.read_line(line_addr)
        word = (line_data >> (offset * 8)) & 0xFFFFFFFF
        return word

    def poll_for_halt(self, timeout=10.0):
        """Polls the halt status address until the core is halted."""
        print("Polling for halt...")
        halt_addr = self.csr_base_addr + 8
        start_time = time.time()
        while time.time() - start_time < timeout:
            value = self.read_word(halt_addr)
            if value == 1:
                print("Core halted.")
                return True
            time.sleep(0.01) # 10ms poll interval
        print("Timed out waiting for core to halt.")
        return False

    def read_data(self, address, size, verbose=True):
        """
        Reads a block of data of a given size from a memory address using
        efficient, chunked bulk TileLink transactions.
        """
        if size == 0:
            return bytearray()

        data = bytearray()
        bytes_remaining = size
        current_addr = address

        total_prep_duration = 0.0
        total_spi_read_duration = 0.0

        # 1. Handle the first line if the start address is unaligned
        start_offset = current_addr % 16
        if start_offset != 0:
            prep_start_time = time.time()
            line_addr = current_addr - start_offset
            bytes_to_read = min(16 - start_offset, bytes_remaining)
            line_data = self.read_line(line_addr)
            line_bytes = line_data.to_bytes(16, 'little')
            data.extend(line_bytes[start_offset : start_offset + bytes_to_read])
            bytes_remaining -= bytes_to_read
            current_addr += bytes_to_read
            total_prep_duration += (time.time() - prep_start_time)

        # 2. Read all aligned data in 16B chunks (V2 doesn't yet support bulk read response parsing in Python)
        while bytes_remaining > 0:
            bytes_to_read = min(16, bytes_remaining)
            spi_read_start_time = time.time()
            line_data = self.read_line(current_addr)
            line_bytes = line_data.to_bytes(16, 'little')
            data.extend(line_bytes[:bytes_to_read])
            total_spi_read_duration += (time.time() - spi_read_start_time)

            bytes_remaining -= bytes_to_read
            current_addr += bytes_to_read

        total_duration = total_prep_duration + total_spi_read_duration
        if verbose and total_duration > 0:
            rate_kbs = (size / 1024) / total_duration
            print(f"Read complete. Transferred {size} bytes "
                  f"in {total_duration:.2f} seconds ({rate_kbs:.2f} KB/s).")

        return data[:size]

    def _get_spi_rw_bytes_cmd(self, write_data):
        """
        Generates the core MPSSE command for a duplex SPI data exchange,
        without any CS# toggling. Uses SPI Mode 1 (Shift on Rise, Sample on Fall).
        """
        cmd = bytearray()
        exchange_len = len(write_data)
        if exchange_len == 0:
            return cmd

        # 0x34: RW Bytes, MSB first, +ve out (Rise), -ve in (Fall) -> Mode 1
        cmd.append(0x34)
        length = exchange_len - 1
        cmd.extend([length & 0xFF, (length >> 8) & 0xFF])
        cmd.extend(write_data)
        return cmd

    def _get_spi_write_bytes_cmd(self, write_data):
        """
        Generates the core MPSSE command for a simplex SPI data write,
        without any CS# toggling. Uses SPI Mode 1 (Shift on Rise).
        """
        cmd = bytearray()
        exchange_len = len(write_data)
        if exchange_len == 0:
            return cmd

        # 0x11: Write Bytes, MSB first, -ve out (Fall)
        cmd.append(0x11)
        length = exchange_len - 1
        cmd.extend([length & 0xFF, (length >> 8) & 0xFF])
        cmd.extend(write_data)
        return cmd

    def _get_idle_clocking_cmd(self, cycles):
        """Generates the raw MPSSE command for idle clocking with bit-level precision."""
        if cycles <= 0:
            return b''

        cmd = bytearray()
        # Ensure CS is high, direction is set for outputs
        cmd.extend([Ftdi.SET_BITS_LOW, 0x08, 0x0b])

        num_full_bytes = cycles // 8
        remaining_bits = cycles % 8

        if num_full_bytes > 0:
            # Command to write bytes
            cmd.append(Ftdi.WRITE_BYTES_PVE_MSB)
            length = num_full_bytes - 1
            cmd.extend([length & 0xFF, (length >> 8) & 0xFF])
            cmd.extend(b'\x00' * num_full_bytes)

        if remaining_bits > 0:
            # Command to write bits. The bit count is 0-7, so we subtract 1.
            # We write the most significant `remaining_bits` of a dummy 0x00 byte.
            cmd.extend([Ftdi.WRITE_BITS_PVE_MSB, remaining_bits - 1, 0x00])

        return cmd

    def idle_clocking(self, cycles):
        """Generates idle clock cycles with CS high."""
        cmd = self._get_idle_clocking_cmd(cycles)
        if cmd:
            self.ftdi.write_data(cmd)


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="FTDI SPI Master Utility (V2 Protocol)")
    parser.add_argument("--usb-serial", required=True, help="USB serial number of the FTDI device.")
    parser.add_argument("--ftdi-port", type=int, default=1, help="Port number of the FTDI device.")
    parser.add_argument("--csr-base-addr", type=lambda x: int(x, 0), default=0x30000, help="Base address for CSR registers (can be hex, default: 0x30000).")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Write command (word-based)
    write_parser = subparsers.add_parser("write-word", help="Write a 32-bit word to memory")
    write_parser.add_argument("addr", type=lambda x: int(x, 0), help="Memory address (can be hex)")
    write_parser.add_argument("data", type=lambda x: int(x, 0), help="Data to write (can be hex)")

    # Read command (word-based)
    read_parser = subparsers.add_parser("read-word", help="Read a 32-bit word from memory")
    read_parser.add_argument("addr", type=lambda x: int(x, 0), help="Memory address (can be hex)")

    load_elf_parser = subparsers.add_parser("load-elf", help="Load an ELF file")
    load_elf_parser.add_argument("elf_file", type=str)

    read_line_parser = subparsers.add_parser("read-line", help="Read a 128-bit line")
    read_line_parser.add_argument("addr", type=lambda x: int(x, 0), help="Memory address (can be hex)")

    reset_parser = subparsers.add_parser("reset", help="Reset the target device")

    load_file_parser = subparsers.add_parser("load-file", help="Load a binary file to a specific address")
    load_file_parser.add_argument("file_path", type=str, help="Path to the binary file")
    load_file_parser.add_argument("address", type=lambda x: int(x, 0), help="Memory address to load to (can be hex)")

    args = parser.parse_args()

    try:
        spi_master = FtdiSpiMaster(args.usb_serial, args.ftdi_port, args.csr_base_addr)
        spi_master.idle_clocking(20)

        if args.command == "write-word":
            spi_master.write_word(args.addr, args.data)
            print(f"Wrote 0x{args.data:08x} to 0x{args.addr:x}")
        elif args.command == "read-word":
            value = spi_master.read_word(args.addr)
            print(f"Read 0x{value:08x} from 0x{args.addr:x}")
        elif args.command == "load-elf":
            spi_master.load_elf(args.elf_file)
        elif args.command == "read-line":
            line_data = spi_master.read_line(args.addr)
            print(f"Line data: 0x{line_data:032x}")
        elif args.command == "reset":
            spi_master.device_reset()
        elif args.command == "load-file":
            spi_master.load_file(args.file_path, args.address)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
