// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>

#include "fpga/sw/uart.h"

#define REG32(addr) (*(volatile uint32_t*)(addr))

// SPI Flash Master registers at 0x40070000
#define SPI_FLASH_BASE 0x40070000
#define SPI_REG_STATUS (SPI_FLASH_BASE + 0x00)
#define SPI_REG_CONTROL (SPI_FLASH_BASE + 0x04)
#define SPI_REG_TXDATA (SPI_FLASH_BASE + 0x08)
#define SPI_REG_RXDATA (SPI_FLASH_BASE + 0x0c)
#define SPI_REG_CSID (SPI_FLASH_BASE + 0x10)
#define SPI_REG_CSMODE (SPI_FLASH_BASE + 0x14)

// GPIO registers at 0x40030000
#define GPIO_BASE 0x40030000
#define GPIO_REG_INPUT_VAL (GPIO_BASE + 0x00)
#define GPIO_REG_OUTPUT_VAL (GPIO_BASE + 0x04)
#define GPIO_REG_OUTPUT_EN (GPIO_BASE + 0x08)

#define GPIO_FLASH_RST_BIT (1 << 4)

static void print_hex8(uint8_t val) {
  char buf[5];
  buf[0] = '0';
  buf[1] = 'x';
  int hi = (val >> 4) & 0xF;
  int lo = val & 0xF;
  buf[2] = hi < 10 ? '0' + hi : 'A' + hi - 10;
  buf[3] = lo < 10 ? '0' + lo : 'A' + lo - 10;
  buf[4] = 0;
  uart_puts(buf);
}

static void spi_init(void) {
  // Div=20, CPOL=0, CPHA=0, Enable=1
  REG32(SPI_REG_CONTROL) = 0x1403;
  // CSMODE=1 (manual CS control) for multi-byte transactions
  REG32(SPI_REG_CSMODE) = 1;
  // CSID[0]=0 → CS deasserted (high) in manual mode
  REG32(SPI_REG_CSID) = 0;
}

static uint8_t spi_xfer(uint8_t tx) {
  // Wait until TX FIFO is not full
  while (REG32(SPI_REG_STATUS) & 0x4);
  REG32(SPI_REG_TXDATA) = tx;
  // Wait until not busy
  while (REG32(SPI_REG_STATUS) & 0x1);
  return (uint8_t)REG32(SPI_REG_RXDATA);
}

static void spi_cs_assert(void) {
  // In manual mode, CSID[0]=1 asserts CS (drives low)
  REG32(SPI_REG_CSID) = 1;
}

static void spi_cs_deassert(void) {
  // Wait until not busy before deasserting
  while (REG32(SPI_REG_STATUS) & 0x1);
  // In manual mode, CSID[0]=0 deasserts CS (drives high)
  REG32(SPI_REG_CSID) = 0;
  // Small delay for CS deassert timing
  for (volatile int i = 0; i < 10; i++);
}

static bool poll_status(uint8_t expected, uint8_t* result) {
  int retry_count = 100;
  uint8_t status;
  spi_cs_assert();
  spi_xfer(0x05);
  do {
    status = spi_xfer(0x00);
    if (status == expected) {
      break;
    }
    --retry_count;
  } while (retry_count);
  spi_cs_deassert();
  if (result) {
    *result = status;
  }
  return (status == expected);
}

int main() {
  uart_init(CLOCK_FREQUENCY_MHZ);

  // Initialize GPIO 4 as output and deassert flash reset (active low)
  REG32(GPIO_REG_OUTPUT_EN) |= GPIO_FLASH_RST_BIT;
  REG32(GPIO_REG_OUTPUT_VAL) |= GPIO_FLASH_RST_BIT;
  for (volatile int i = 0; i < 100; i++);

  spi_init();

  {
    // ===== Test 1: Hardware Reset =====
    uart_puts("T1: HW Reset\r\n");

    // Set WEL
    spi_cs_assert();
    spi_xfer(0x06);  // WREN
    spi_cs_deassert();

    // Verify WEL is set (Status Bit 1)
    spi_cs_assert();
    spi_xfer(0x05);  // READ_STATUS
    uint8_t status = spi_xfer(0x00);
    spi_cs_deassert();

    if (!(status & 0x02)) {
      uart_puts("FAIL T1: WEL not set\r\n");
      return 1;
    }

    // Pulse Reset
    REG32(GPIO_REG_OUTPUT_VAL) &= ~GPIO_FLASH_RST_BIT;  // Assert
    for (volatile int i = 0; i < 100; i++);
    REG32(GPIO_REG_OUTPUT_VAL) |= GPIO_FLASH_RST_BIT;  // Deassert
    for (volatile int i = 0; i < 100; i++);

    // Verify WEL is cleared
    spi_cs_assert();
    spi_xfer(0x05);  // READ_STATUS
    status = spi_xfer(0x00);
    spi_cs_deassert();

    if (status & 0x02) {
      uart_puts("FAIL T1: WEL still set after reset\r\n");
      return 1;
    }
    uart_puts("T1: OK\r\n");
  }

  {
    // ===== Test 2: Read JEDEC ID =====
    uart_puts("T2: JEDEC ID\r\n");

    spi_cs_assert();
    spi_xfer(0x9F);  // READ_ID command
    uint8_t mfr = spi_xfer(0x00);
    uint8_t id1 = spi_xfer(0x00);
    uint8_t id2 = spi_xfer(0x00);
    spi_cs_deassert();

    uart_puts("  MFR=");
    print_hex8(mfr);
    uart_puts(" ID1=");
    print_hex8(id1);
    uart_puts(" ID2=");
    print_hex8(id2);
    uart_puts("\r\n");

    if (mfr != 0x01 || id1 != 0x02 || id2 != 0x20) {
      uart_puts("FAIL T2: wrong JEDEC ID\r\n");
      return 1;
    }
    uart_puts("T2: OK\r\n");
  }

  {
    // ===== Test 3: Sector Erase + Verify =====
    uart_puts("T3: Erase+Verify\r\n");

    // Write Enable
    spi_cs_assert();
    spi_xfer(0x06);  // WREN
    spi_cs_deassert();
    uint8_t result;
    if (!poll_status(0x2, &result)) {
      uart_puts("FAIL T3: WREN failed: ");
      print_hex8(result);
      uart_puts("\r\n");
      return 1;
    }

    // Sector Erase at address 0x000000
    spi_cs_assert();
    spi_xfer(0xD8);  // SECTOR_ERASE
    spi_xfer(0x10);  // Addr[23:16]
    spi_xfer(0x10);  // Addr[15:8]
    spi_xfer(0x10);  // Addr[7:0]
    spi_cs_deassert();

    // Poll until the write is done...
    int poll_count = 0;
    {
      spi_cs_assert();
      spi_xfer(0x05);
      do {
        result = spi_xfer(0x00);
        poll_count++;
      } while ((result & 0x3) && poll_count < 0x7FFFFFFF);
      spi_cs_deassert();
    }

    if (result != 0) {
      uart_puts("FAIL T3: Erase did not finish: ");
      print_hex8(result);
      uart_puts("\r\n");
      return 1;
    }

    // Read back - should be 0xFF
    spi_cs_assert();
    spi_xfer(0x03);  // READ
    spi_xfer(0x00);
    spi_xfer(0x00);
    spi_xfer(0x00);
    uint8_t d0 = spi_xfer(0x00);
    uint8_t d1 = spi_xfer(0x00);
    uint8_t d2 = spi_xfer(0x00);
    uint8_t d3 = spi_xfer(0x00);
    spi_cs_deassert();

    uart_puts("  Read: ");
    print_hex8(d0);
    uart_puts(" ");
    print_hex8(d1);
    uart_puts(" ");
    print_hex8(d2);
    uart_puts(" ");
    print_hex8(d3);
    uart_puts("\r\n");

    if (d0 != 0xFF || d1 != 0xFF || d2 != 0xFF || d3 != 0xFF) {
      uart_puts("FAIL T3: erase failed\r\n");
      return 1;
    }
    uart_puts("T3: OK\r\n");
  }

  {
    // ===== Test 4: Page Program + Read =====
    uart_puts("T4: Program+Read\r\n");

    {
      // Read back at address 0x000000
      spi_cs_assert();
      spi_xfer(0x03);  // READ
      spi_xfer(0x10);  // Addr[23:16]
      spi_xfer(0x10);  // Addr[15:8]
      spi_xfer(0x10);  // Addr[7:0]
      uint8_t d0 = spi_xfer(0x00);
      uint8_t d1 = spi_xfer(0x00);
      uint8_t d2 = spi_xfer(0x00);
      uint8_t d3 = spi_xfer(0x00);
      spi_cs_deassert();

      if (d0 != 0xFF || d1 != 0xFF || d2 != 0xFF || d3 != 0xFF) {
        uart_puts("FAIL T4: Program target not erased?\r\n");
        return 1;
      }
    }

    // Write Enable
    spi_cs_assert();
    spi_xfer(0x06);  // WREN
    spi_cs_deassert();

    uint8_t result;
    if (!poll_status(0x2, &result)) {
      uart_puts("FAIL T4: WREN failed: ");
      print_hex8(result);
      uart_puts("\r\n");
      return 1;
    }

    // Page Program at address 0x000000
    spi_cs_assert();
    spi_xfer(0x02);  // PAGE_PROGRAM
    spi_xfer(0x10);  // Addr[23:16]
    spi_xfer(0x10);  // Addr[15:8]
    spi_xfer(0x10);  // Addr[7:0]
    spi_xfer(0xDE);
    spi_xfer(0xAD);
    spi_xfer(0xBE);
    spi_xfer(0xEF);
    spi_cs_deassert();

    if (!poll_status(0x0, nullptr)) {
      uart_puts("FAIL T4: Write did not finish?\r\n");
      return 1;
    }

    // Read back at address 0x000000
    spi_cs_assert();
    spi_xfer(0x03);  // READ
    spi_xfer(0x10);  // Addr[23:16]
    spi_xfer(0x10);  // Addr[15:8]
    spi_xfer(0x10);  // Addr[7:0]
    uint8_t d0 = spi_xfer(0x00);
    uint8_t d1 = spi_xfer(0x00);
    uint8_t d2 = spi_xfer(0x00);
    uint8_t d3 = spi_xfer(0x00);
    spi_cs_deassert();

    uart_puts("  Read: ");
    print_hex8(d0);
    uart_puts(" ");
    print_hex8(d1);
    uart_puts(" ");
    print_hex8(d2);
    uart_puts(" ");
    print_hex8(d3);
    uart_puts("\r\n");

    if (d0 != 0xDE || d1 != 0xAD || d2 != 0xBE || d3 != 0xEF) {
      uart_puts("FAIL T4: data mismatch\r\n");
      return 1;
    }
    uart_puts("T4: OK\r\n");
  }

  uart_puts("TEST PASSED\r\n");
  return 0;
}
