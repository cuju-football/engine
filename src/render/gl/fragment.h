// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The fragment interface: one call per 4x4 block, as llvmpipe's fragment shader functions, plus the
// per-lane helpers that every program shares (interpolation, depth, the RGBA16F, 10-bit and
// overlay stores). The arithmetic follows llvmpipe's JIT code (Mesa, MIT licence,
// src/gallium/drivers/llvmpipe/lp_state_fs.c, lp_bld_interp.c, lp_bld_depth.c).
//
// Lanes. llvmpipe shades a block as two halves of eight lanes. Lane i of half h is pixel
// (kLaneX[i], kLaneY[i]) of the block with kLaneY including the 2h offset, so lanes come in 2x2
// quads (0,0), (2,0), (0,2), (2,2), lane = 4 * quad + x + 2y inside the quad. That is the lane
// order of sampler.h, so arrays of 16 lanes go to SampleQuads with quads = 4. Per-lane arrays in
// this interface are SoA with 16 lanes per channel: v[c * 16 + lane].
//
// Coverage. FragmentBlock::coverage has bit 4 * row + col for pixel (col, row) of the block, row
// counted in memory rows. Lane masks have bit l for lane l.

#ifndef CUJU_RENDER_GL_FRAGMENT_H
#define CUJU_RENDER_GL_FRAGMENT_H

#include <cstdint>

#include "../fpenv.h"
#include "../numerics.h"
#include "program.h"

namespace cuju::render::gl {
constexpr int kLaneX[16] = {0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3};
constexpr int kLaneY[16] = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};

struct FragmentBlock {
    int x = 0;
    int y = 0;
    std::uint32_t coverage = 0xFFFF;
    bool frontfacing = true;
    const float (*a0)[4] = nullptr;
    const float (*dadx)[4] = nullptr;
    const float (*dady)[4] = nullptr;
    const DrawState *state = nullptr;
    std::uint8_t *color[kMaxColorBuffers] = {};
    int color_stride[kMaxColorBuffers] = {};
    std::uint8_t *depth = nullptr;
    int depth_stride = 0;
};

inline std::uint16_t LaneMaskFromCoverage(std::uint32_t coverage) {
    std::uint16_t m = 0;
    for (int l = 0; l < 16; ++l) {
        if (coverage & (1u << (4 * kLaneY[l] + kLaneX[l]))) {
            m = static_cast<std::uint16_t>(m | (1u << l));
        }
    }
    return m;
}

void LaneCoordinates(const FragmentBlock &b, float *X, float *Y);

// The plane value fma(dady, Y, fma(dadx, X, a0)) of component c of a slot, per lane.
void InterpolatePlane(const FragmentBlock &b, int slot, int c, const float *X, const float *Y,
                      float *out);

void PerspectiveOneOverW(const FragmentBlock &b, const float *X, const float *Y, float *one_over_w);
void InterpolatePerspective(const FragmentBlock &b, int slot, int c, const float *X, const float *Y,
                            const float *one_over_w, float *out);

void FragCoord(const FragmentBlock &b, const float *X, const float *Y, float *fx, float *fy);

void InterpolateDepth(const FragmentBlock &b, const float *X, const float *Y, float *z);
void DepthToD16(const float *z, std::uint16_t *d16);
std::uint16_t DepthTest(const FragmentBlock &b, const std::uint16_t *d16, std::uint16_t lanes);
void DepthWrite(const FragmentBlock &b, const std::uint16_t *d16, std::uint16_t lanes);

void LoadRgba16f(const FragmentBlock &b, int buffer, float *rgba);
void StoreRgba16f(const FragmentBlock &b, int buffer, const float *rgba, std::uint16_t lanes);
void BlendAddRgba16f(const FragmentBlock &b, int buffer, const float *rgba, std::uint16_t lanes);

inline std::uint16_t Unorm16Ftz(float c) {
    return static_cast<std::uint16_t>(Bits(ftz::Add(ftz::Mul(c, FromBits(0x3F7FFF00u)), 128.0f)) &
                                      0xFFFFu);
}
inline std::uint32_t Pack10(std::uint32_t r16, std::uint32_t g16, std::uint32_t b16,
                            std::uint32_t a16) {
    const std::uint32_t a2 = (((a16 >> 2) * 3u) + 8192u) >> 14;
    return (b16 >> 6) | ((g16 >> 6) << 10) | ((r16 >> 6) << 20) | ((a2 & 3u) << 30);
}
inline std::uint32_t Expand10To16(std::uint32_t v10) { return (v10 << 6) | (v10 >> 4); }
inline std::uint32_t Expand2To16(std::uint32_t v2) { return v2 * 0x5555u; }
void Store10(const FragmentBlock &b, int buffer, const std::uint32_t *words, std::uint16_t lanes);
void Load10(const FragmentBlock &b, int buffer, std::uint32_t *words);

inline std::uint32_t Mul16(std::uint32_t x, std::uint32_t y) {
    const std::uint32_t t = x * y;
    return (t + (t >> 16) + 0x8000u) >> 16;
}
inline std::uint32_t AddSat16(std::uint32_t a, std::uint32_t b) {
    const std::uint32_t s = a + b;
    return s > 0xFFFFu ? 0xFFFFu : s;
}

}

#endif
