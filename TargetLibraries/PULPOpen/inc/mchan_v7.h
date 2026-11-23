/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __MCHAN_V7_H__
#define __MCHAN_V7_H__

#include "assert.h"

// Requires to have MCHAN_BASE_ADDR, MCHAN_EVENT defined outside of header
#ifndef MCHAN_BASE_ADDR
#error "[mchan_v7.h] MCHAN_BASE_ADDR not defined!"
#endif

#if !defined(MCHAN_EVENT) && !defined(MCHAN_POLLED)
#error "[mchan_v7.h] Nor MCHAN_EVENT nor MCHAN_POLLED defined!"
#endif

#if defined(MCHAN_EVENT) && defined(MCHAN_POLLED)
#error "[mchan_v7.h] Define either MCHAN_EVENT or MCHAN_POLLED, not both!"
#endif

#if defined(MCHAN_EVENT) && !defined(MCHAN_EVENT_BIT)
#error                                                                         \
    "[mchan_v7.h] MCHAN_EVENT_BIT should be defined when using events as signalization!"
#endif

#if !defined(MCHAN_VERSION)
#define MCHAN_VERSION 7
#elif MCHAN_VERSION != 7
#error "[mchan_v7.h] Illegal MCHAN_VERSION. Supported only 7"
#endif

#include "pmsis.h"

#define MCHAN_TRANSFER_LEN_SIZE (17)
#define MCHAN_CHANNEL_ID_MAX (15)

#define MCHAN_CMD_FLAG_DIRECTION_LOC2EXT (0 << (MCHAN_TRANSFER_LEN_SIZE + 0))
#define MCHAN_CMD_FLAG_DIRECTION_EXT2LOC (1 << (MCHAN_TRANSFER_LEN_SIZE + 0))
#define MCHAN_CMD_FLAG_INCREMENTAL (1 << (MCHAN_TRANSFER_LEN_SIZE + 1))
#define MCHAN_CMD_FLAG_2D_TRANSFER_EXTERNAL (1 << (MCHAN_TRANSFER_LEN_SIZE + 2))
#define MCHAN_CMD_FLAG_EVENT_ENABLE (1 << (MCHAN_TRANSFER_LEN_SIZE + 3))
#define MCHAN_CMD_FLAG_INTERRUPT_ENABLE (1 << (MCHAN_TRANSFER_LEN_SIZE + 4))
#define MCHAN_CMD_FLAG_BROADCAST_FINISH (1 << (MCHAN_TRANSFER_LEN_SIZE + 5))
#define MCHAN_CMD_FLAG_2D_TRANSFER_LOCAL (1 << (MCHAN_TRANSFER_LEN_SIZE + 6))

static volatile uint32_t *const cmd_ptr =
    (volatile uint32_t *const)(MCHAN_BASE_ADDR + 0x0);
static volatile uint32_t *const status_ptr =
    (volatile uint32_t *const)(MCHAN_BASE_ADDR + 0x4);

static void mchan_transfer_1d(uint32_t cmd, void *loc, void *ext) {
  // TODO: assert flags are set correctly
  *cmd_ptr = (uint32_t)cmd;
  *cmd_ptr = (uint32_t)loc;
  *cmd_ptr = (uint32_t)ext;
}

static void mchan_transfer_2d_loc_strided(uint32_t cmd, void *loc, void *ext,
                                          uint32_t loc_size_1d,
                                          uint32_t loc_stride_2d) {
  // TODO: assert flags are set correctly
  *cmd_ptr = (uint32_t)cmd;
  *cmd_ptr = (uint32_t)loc;
  *cmd_ptr = (uint32_t)ext;
  *cmd_ptr = (uint32_t)loc_size_1d;
  *cmd_ptr = (uint32_t)loc_stride_2d;
}

static void mchan_transfer_2d_ext_strided(uint32_t cmd, void *loc, void *ext,
                                          uint32_t ext_size_1d,
                                          uint32_t ext_stride_2d) {
  // TODO: assert flags are set correctly
  *cmd_ptr = (uint32_t)cmd;
  *cmd_ptr = (uint32_t)loc;
  *cmd_ptr = (uint32_t)ext;
  *cmd_ptr = (uint32_t)ext_size_1d;
  *cmd_ptr = (uint32_t)ext_stride_2d;
}

static void mchan_transfer_2d_loc_strided_ext_strided(
    uint32_t cmd, void *loc, void *ext, uint32_t loc_size_1d,
    uint32_t loc_stride_2d, uint32_t ext_size_1d, uint32_t ext_stride_2d) {
  // TODO: assert flags are set correctly
  *cmd_ptr = (uint32_t)cmd;
  *cmd_ptr = (uint32_t)loc;
  *cmd_ptr = (uint32_t)ext;
  *cmd_ptr = (uint32_t)ext_size_1d;
  *cmd_ptr = (uint32_t)ext_stride_2d;
  *cmd_ptr = (uint32_t)loc_size_1d;
  *cmd_ptr = (uint32_t)loc_stride_2d;
}

// Transfers longer than the 17-bit length field allows are split into several
// commands on the currently allocated channel; waiting on that channel waits
// for all of them.
#define MCHAN_MAX_TRANSFER_LEN ((1 << MCHAN_TRANSFER_LEN_SIZE) - 1)

static void mchan_transfer_1d_chunked(uint32_t cmd_flags, uint32_t len,
                                      void *loc, void *ext) {
  char *l = (char *)loc, *e = (char *)ext;
  const uint32_t chunk = 1 << (MCHAN_TRANSFER_LEN_SIZE - 1);
  while (len > 0) {
    uint32_t n = len > chunk ? chunk : len;
    mchan_transfer_1d(cmd_flags | n, l, e);
    l += n;
    e += n;
    len -= n;
  }
}

static void mchan_transfer_2d_ext_strided_chunked(uint32_t cmd_flags,
                                                  uint32_t len, void *loc,
                                                  void *ext,
                                                  uint32_t ext_size_1d,
                                                  uint32_t ext_stride_2d) {
  char *l = (char *)loc, *e = (char *)ext;
  const uint32_t rows = MCHAN_MAX_TRANSFER_LEN / ext_size_1d;
  const uint32_t chunk = rows * ext_size_1d;
  while (len > 0) {
    uint32_t n = len > chunk ? chunk : len;
    mchan_transfer_2d_ext_strided(cmd_flags | n, l, e, ext_size_1d,
                                  ext_stride_2d);
    l += n;
    e += (n / ext_size_1d) * ext_stride_2d;
    len -= n;
  }
}

static uint32_t mchan_channel_alloc() { return *cmd_ptr; }

static void mchan_channel_free(uint32_t channel_id) {
  assert(channel_id <= MCHAN_CHANNEL_ID_MAX);
  *status_ptr = 1 << channel_id;
}

static uint32_t mchan_channel_is_busy(uint32_t channel_id) {
  assert(channel_id <= MCHAN_CHANNEL_ID_MAX);
  return *status_ptr & (1 << channel_id);
}

static void mchan_channel_wait(uint32_t channel_id) {
  assert(channel_id <= MCHAN_CHANNEL_ID_MAX);
#if defined(MCHAN_EVENT)
  while (mchan_channel_is_busy(channel_id))
    eu_evt_maskWaitAndClr(1 << MCHAN_EVENT_BIT);
#elif defined(MCHAN_POLLED)
  while (mchan_channel_is_busy(channel_id))
    ;
#endif
}

#endif // __MCHAN_V7_H__
