// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <new>

#include "warp.h"

namespace cuju {
CUJU_HOST_DEVICE constexpr size_t WorkAreaRound(size_t bytes) { return (bytes + 15) / 16 * 16; }

#if !defined(CUJU_WORK_AREA_BYTES)
#define CUJU_WORK_AREA_BYTES 12288
#endif
constexpr size_t WORK_AREA_BYTES = CUJU_WORK_AREA_BYTES;
struct alignas(16) WorkArea {
    unsigned char bytes[WORK_AREA_BYTES];
};

#if CUJU_WARP_LANES
struct WarpWorkState {
    unsigned char *base;
    uint32_t used;
};
__device__ __forceinline__ WarpWorkState &WarpWork() {
    __shared__ WarpWorkState states[32];
    return states[threadIdx.x >> 5];
}
#endif

CUJU_HOST_DEVICE void WarpWorkInitialize(WorkArea *area) {
#if CUJU_WARP_LANES
    if (Lane() == 0) {
        WarpWork().base = area->bytes;
        WarpWork().used = 0;
    }
    __syncwarp();
#else
    (void)area;
#endif
}

template <class T>
class WarpLocal {
public:
    CUJU_HOST_DEVICE WarpLocal() {
#if CUJU_WARP_LANES
        constexpr uint32_t SIZE = (uint32_t)WorkAreaRound(sizeof(T));
        static_assert(alignof(T) <= 16, "WarpLocal: alignment of at most 16");
        WarpWorkState &state = WarpWork();
        __syncwarp();
        offset_ = state.used;
        if (WarpNested() || offset_ + SIZE > WORK_AREA_BYTES) __trap();
        __syncwarp();
        if (Lane() == 0) state.used = offset_ + SIZE;
        __syncwarp();
        value_ = new (state.base + offset_) T();
#endif
    }
    CUJU_HOST_DEVICE ~WarpLocal() {
#if CUJU_WARP_LANES
        __syncwarp();
        if (Lane() == 0) WarpWork().used = offset_;
        __syncwarp();
#endif
    }
    WarpLocal(const WarpLocal &) = delete;
    WarpLocal &operator=(const WarpLocal &) = delete;

#if CUJU_WARP_LANES
    CUJU_HOST_DEVICE T &operator*() { return *value_; }

private:
    T *value_;
    uint32_t offset_;
#else
    CUJU_HOST_DEVICE T &operator*() { return value_; }

private:
    T value_{};
#endif
};

}  // namespace cuju
