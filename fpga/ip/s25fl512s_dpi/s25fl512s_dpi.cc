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

#include <cstdint>
#include <cstring>
#include <iostream>

#include "svdpi.h"

// S25FL512S: 64MB NOR flash.
#define FLASH_SIZE (64 * 1024 * 1024)
// 256kB sectors
#define SECTOR_SIZE (256 * 1024)

enum FlashState {
  STATE_IDLE,
  STATE_CMD,
  STATE_ADDR,
  STATE_DUMMY,
  STATE_DATA_READ,
  STATE_DATA_WRITE,
  STATE_READ_STATUS,
  STATE_READ_ID,
};

enum FlashCmd {
  CMD_READ_ID = 0x9F,
  CMD_READ_STATUS = 0x05,
  CMD_WRITE_ENABLE = 0x06,
  CMD_WRITE_DISABLE = 0x04,
  CMD_READ = 0x03,
  CMD_FAST_READ = 0x0B,
  CMD_PAGE_PROGRAM = 0x02,
  CMD_SECTOR_ERASE = 0xD8,
  CMD_BULK_ERASE = 0xC7,
};

struct S25fl512sState {
  uint8_t memory[FLASH_SIZE];
  uint8_t status_reg;  // bit 0: WIP, bit 1: WEL

  // SPI state
  uint8_t sck_prev;
  uint8_t csb_prev;
  uint8_t shift_in;
  uint8_t shift_out;
  uint8_t miso_bit;  // persistent MISO output, updated on falling edges only
  int bit_count_in;
  int bit_count_out;

  // Deferred byte processing: received bytes are processed on the falling edge
  // after reception, so the last MISO bit of the current byte is output
  // correctly before the new shift_out is loaded.
  uint8_t pending_byte;
  bool pending_byte_valid;

  // Command state
  FlashState state;
  uint8_t cmd;
  uint32_t addr;
  int addr_bytes;
  int dummy_bytes;
  int id_index;
};

static void flash_reset(S25fl512sState* ctx) {
  ctx->sck_prev = 0;
  ctx->csb_prev = 1;
  ctx->shift_in = 0;
  ctx->shift_out = 0;
  ctx->miso_bit = 0;
  ctx->bit_count_in = 0;
  ctx->bit_count_out = 0;
  ctx->pending_byte = 0;
  ctx->pending_byte_valid = false;
  ctx->state = STATE_IDLE;
  ctx->cmd = 0;
  ctx->addr = 0;
  ctx->addr_bytes = 0;
  ctx->dummy_bytes = 0;
  ctx->id_index = 0;
  ctx->status_reg = 0;
}

static void flash_cs_deassert(S25fl512sState* ctx) {
  // Complete pending operations on CS deassert
  switch (ctx->cmd) {
    case CMD_SECTOR_ERASE:
      if (ctx->state == STATE_ADDR || ctx->addr_bytes == 3) {
        uint32_t sector_base = ctx->addr & ~(SECTOR_SIZE - 1);
        if (ctx->status_reg & 0x02) {  // WEL set
          uint32_t end = sector_base + SECTOR_SIZE;
          if (end > FLASH_SIZE) end = FLASH_SIZE;
          if (sector_base < FLASH_SIZE) {
            memset(&ctx->memory[sector_base], 0xFF, end - sector_base);
          }
          ctx->status_reg &= ~0x02;  // Clear WEL
        }
      }
      break;
    case CMD_BULK_ERASE:
      if (ctx->status_reg & 0x02) {  // WEL set
        memset(ctx->memory, 0xFF, FLASH_SIZE);
        ctx->status_reg &= ~0x02;  // Clear WEL
      }
      break;
    case CMD_PAGE_PROGRAM:
      ctx->status_reg &= ~0x02;  // Clear WEL
      break;
    default:
      break;
  }
  ctx->state = STATE_IDLE;
  ctx->cmd = 0;
  ctx->addr = 0;
  ctx->addr_bytes = 0;
  ctx->dummy_bytes = 0;
  ctx->bit_count_in = 0;
  ctx->bit_count_out = 0;
  ctx->id_index = 0;
}

static void flash_process_cmd(S25fl512sState* ctx, uint8_t cmd) {
  ctx->cmd = cmd;
  switch (cmd) {
    case CMD_WRITE_ENABLE:
      ctx->status_reg |= 0x02;  // Set WEL
      ctx->state = STATE_IDLE;
      break;
    case CMD_WRITE_DISABLE:
      ctx->status_reg &= ~0x02;  // Clear WEL
      ctx->state = STATE_IDLE;
      break;
    case CMD_READ_STATUS:
      ctx->state = STATE_READ_STATUS;
      ctx->shift_out = ctx->status_reg;
      ctx->bit_count_out = 0;
      break;
    case CMD_READ_ID:
      ctx->state = STATE_READ_ID;
      ctx->id_index = 0;
      ctx->shift_out = 0x01;  // Manufacturer ID (Spansion)
      ctx->bit_count_out = 0;
      break;
    case CMD_READ:
      ctx->state = STATE_ADDR;
      ctx->addr = 0;
      ctx->addr_bytes = 0;
      ctx->dummy_bytes = 0;
      break;
    case CMD_FAST_READ:
      ctx->state = STATE_ADDR;
      ctx->addr = 0;
      ctx->addr_bytes = 0;
      ctx->dummy_bytes = 1;
      break;
    case CMD_PAGE_PROGRAM:
      ctx->state = STATE_ADDR;
      ctx->addr = 0;
      ctx->addr_bytes = 0;
      ctx->dummy_bytes = 0;
      break;
    case CMD_SECTOR_ERASE:
      ctx->state = STATE_ADDR;
      ctx->addr = 0;
      ctx->addr_bytes = 0;
      break;
    case CMD_BULK_ERASE:
      // Handled at CS deassert
      ctx->state = STATE_IDLE;
      break;
    default:
      ctx->state = STATE_IDLE;
      break;
  }
}

static void flash_receive_byte(S25fl512sState* ctx, uint8_t byte) {
  switch (ctx->state) {
    case STATE_IDLE:
      flash_process_cmd(ctx, byte);
      break;
    case STATE_ADDR:
      ctx->addr = (ctx->addr << 8) | byte;
      ctx->addr_bytes++;
      if (ctx->addr_bytes == 3) {
        if (ctx->dummy_bytes > 0) {
          ctx->state = STATE_DUMMY;
        } else if (ctx->cmd == CMD_READ) {
          ctx->state = STATE_DATA_READ;
          if (ctx->addr < FLASH_SIZE) {
            ctx->shift_out = ctx->memory[ctx->addr];
          } else {
            ctx->shift_out = 0xFF;
          }
          ctx->bit_count_out = 0;
        } else if (ctx->cmd == CMD_PAGE_PROGRAM) {
          ctx->state = STATE_DATA_WRITE;
        } else if (ctx->cmd == CMD_SECTOR_ERASE) {
          // Erase happens at CS deassert
          ctx->state = STATE_IDLE;
        }
      }
      break;
    case STATE_DUMMY:
      ctx->dummy_bytes--;
      if (ctx->dummy_bytes == 0) {
        ctx->state = STATE_DATA_READ;
        if (ctx->addr < FLASH_SIZE) {
          ctx->shift_out = ctx->memory[ctx->addr];
        } else {
          ctx->shift_out = 0xFF;
        }
        ctx->bit_count_out = 0;
      }
      break;
    case STATE_DATA_WRITE:
      if (ctx->status_reg & 0x02) {  // WEL set
        if (ctx->addr < FLASH_SIZE) {
          // Page program can only clear bits (AND operation)
          ctx->memory[ctx->addr] &= byte;
        }
        ctx->addr++;
        // Wrap within 256-byte page
        if ((ctx->addr & 0xFF) == 0) {
          ctx->addr -= 256;
        }
      }
      break;
    case STATE_DATA_READ:
      // During read, incoming bytes are dummy/ignored
      ctx->addr++;
      if (ctx->addr < FLASH_SIZE) {
        ctx->shift_out = ctx->memory[ctx->addr];
      } else {
        ctx->shift_out = 0xFF;
      }
      ctx->bit_count_out = 0;
      break;
    case STATE_READ_STATUS:
      // Continuous read of status register
      ctx->shift_out = ctx->status_reg;
      ctx->bit_count_out = 0;
      break;
    case STATE_READ_ID:
      ctx->id_index++;
      if (ctx->id_index == 1) {
        ctx->shift_out = 0x02;  // Device ID byte 1
      } else if (ctx->id_index == 2) {
        ctx->shift_out = 0x20;  // Device ID byte 2
      } else {
        ctx->shift_out = 0x00;
      }
      ctx->bit_count_out = 0;
      break;
    default:
      break;
  }
}

extern "C" {

S25fl512sState* s25fl512s_dpi_init() {
  S25fl512sState* ctx = new S25fl512sState();
  memset(ctx->memory, 0xFF, FLASH_SIZE);
  ctx->status_reg = 0;
  flash_reset(ctx);
  std::cout << "DPI: S25FL512S Flash Model Initialized (" << FLASH_SIZE / 1024
            << " KB)" << std::endl;
  return ctx;
}

void s25fl512s_dpi_close(S25fl512sState* ctx) {
  if (ctx) {
    delete ctx;
  }
  std::cout << "DPI: S25FL512S Flash Model Closed" << std::endl;
}

void s25fl512s_dpi_reset(S25fl512sState* ctx) {
  if (ctx) {
    memset(ctx->memory, 0xFF, FLASH_SIZE);
    ctx->status_reg = 0;
    flash_reset(ctx);
  }
}

void s25fl512s_dpi_tick(S25fl512sState* ctx, unsigned char sck,
                        unsigned char csb, unsigned char mosi,
                        unsigned char rst_ni, unsigned char* miso) {
  *miso = 0;

  if (!ctx) return;

  if (!rst_ni) {
    flash_reset(ctx);
    return;
  }

  // CSB rising edge: end of transaction
  if (!ctx->csb_prev && csb) {
    // Process any pending byte before deasserting
    if (ctx->pending_byte_valid) {
      flash_receive_byte(ctx, ctx->pending_byte);
      ctx->pending_byte_valid = false;
    }
    flash_cs_deassert(ctx);
    ctx->csb_prev = csb;
    ctx->sck_prev = sck;
    return;
  }

  // CSB falling edge: start of new command
  if (ctx->csb_prev && !csb) {
    ctx->state = STATE_IDLE;
    ctx->bit_count_in = 0;
    ctx->bit_count_out = 0;
    ctx->shift_in = 0;
  }

  // SCK rising edge: output MISO, then process any deferred received byte.
  // Processing is deferred from the falling edge so the last MISO bit of the
  // current output byte is driven correctly before shift_out is overwritten.
  if (!csb && !ctx->sck_prev && sck) {
    // First, drive MISO from current shift_out (before any overwrite)
    if (ctx->state == STATE_DATA_READ || ctx->state == STATE_READ_STATUS ||
        ctx->state == STATE_READ_ID) {
      ctx->miso_bit = (ctx->shift_out >> 7) & 1;
      ctx->shift_out <<= 1;
      ctx->bit_count_out++;
    }

    // Then, process any pending received byte
    if (ctx->pending_byte_valid) {
      flash_receive_byte(ctx, ctx->pending_byte);
      ctx->pending_byte_valid = false;
    }
  }

  // SCK falling edge: capture MOSI, defer byte processing to next rising edge
  if (!csb && ctx->sck_prev && !sck) {
    ctx->shift_in = (ctx->shift_in << 1) | (mosi & 1);
    ctx->bit_count_in++;

    if (ctx->bit_count_in == 8) {
      ctx->pending_byte = ctx->shift_in;
      ctx->pending_byte_valid = true;
      ctx->bit_count_in = 0;
      ctx->shift_in = 0;
    }
  }

  // Always drive MISO from persistent miso_bit
  *miso = ctx->miso_bit;

  ctx->csb_prev = csb;
  ctx->sck_prev = sck;
}

}  // extern "C"
