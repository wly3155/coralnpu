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
import logging
import time

from elftools.elf.elffile import ELFFile
from coralnpu_hw.utils.coralnpu_soc_loader.spi_driver import SPIDriver

def write_line_via_spi(driver: SPIDriver, address: int, data: int):
    """Writes a 16-byte bus line to a given address via the SPI bridge."""
    data_bytes = data.to_bytes(16, byteorder='little')
    driver.v2_write(address, data_bytes)

def write_lines_via_spi(driver: SPIDriver, address: int, data_bytes: bytes):
    """Writes multiple 16-byte bus lines to a given address via the SPI bridge."""
    if len(data_bytes) % 16 != 0:
        raise ValueError("Data length must be a multiple of 16 bytes")
    if len(data_bytes) == 0:
        return
    driver.v2_write(address, data_bytes)

def read_line_via_spi(driver: SPIDriver, address: int) -> int:
    """Reads a single 128-bit line from memory via SPI."""
    read_data_bytes = driver.v2_read(address, 1)
    read_data = int.from_bytes(bytes(read_data_bytes), byteorder='little')
    return read_data

def write_word_via_spi(driver: SPIDriver, address: int, data: int):
    """Writes a 32-bit value by performing a read-modify-write on a 16-byte line."""
    line_addr = (address // 16) * 16
    offset = address % 16

    # Read the current line
    line_data = read_line_via_spi(driver, line_addr)

    # Create a 16-byte mask for the 4 bytes we want to change
    mask = 0xFFFFFFFF << (offset * 8)

    # Clear the bits we want to change, then OR in the new data
    updated_data = (line_data & ~mask) | (data << (offset * 8))

    # Write the modified line back
    write_line_via_spi(driver, line_addr, updated_data)

def main():
    parser = argparse.ArgumentParser(description="Load an ELF binary to the CoralNPU SoC.")
    parser.add_argument("binary", help="Path to the ELF binary to load.")
    args = parser.parse_args()

    driver = None
    try:
        driver = SPIDriver()

        # Send a few idle clock cycles to flush any reset synchronizers
        # in the DUT before starting the first real transaction.
        logging.warning("LOADER: Sending initial idle clocks to flush reset...")
        driver.idle_clocking(20)

        entry_point = 0
        logging.warning(f"LOADER: Opening ELF file: {args.binary}")
        with open(args.binary, 'rb') as f:
            elffile = ELFFile(f)
            entry_point = elffile.header.e_entry

            for segment in elffile.iter_segments():
                if segment['p_type'] != 'PT_LOAD':
                    continue

                paddr = segment['p_paddr']
                data = segment.data()
                logging.warning(f"LOADER: Loading segment to address 0x{paddr:08x}, size {len(data)} bytes")

                # Load data in pages (up to some reasonable size)
                original_len = len(data)
                # Pad data to be a multiple of 16 bytes (a line)
                if len(data) % 16 != 0:
                    data += b'\x00' * (16 - (len(data) % 16))

                page_size = 4096
                for i in range(0, len(data), page_size):
                    page_addr = paddr + i
                    page_data_bytes = data[i:i+page_size]

                    write_lines_via_spi(driver, page_addr, page_data_bytes)

                    bytes_written = min(i + len(page_data_bytes), original_len)
                    logging.warning(f"  ... wrote {bytes_written}/{original_len} bytes")
                logging.warning(f"  ... wrote {original_len}/{original_len} bytes")

        logging.warning("LOADER: Binary loaded successfully.")

        # --- Execute Program ---
        coralnpu_pc_csr_addr = 0x30004
        coralnpu_reset_csr_addr = 0x30000

        logging.warning(f"LOADER: Programming start PC to 0x{entry_point:08x}")
        write_word_via_spi(driver, coralnpu_pc_csr_addr, entry_point)

        logging.warning("LOADER: Releasing clock gate...")
        write_word_via_spi(driver, coralnpu_reset_csr_addr, 1)

        logging.warning("LOADER: Releasing reset...")
        write_word_via_spi(driver, coralnpu_reset_csr_addr, 0)

        logging.warning("LOADER: Execution started.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if driver:
            logging.info("LOADER: Closing connection.")
            driver.close()

if __name__ == "__main__":
    main()
