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
    int x = 0;  // block origin in memory coordinates, multiples of 4
    int y = 0;
    std::uint32_t coverage = 0xFFFF;
    bool frontfacing = true;
    // Setup planes per slot: slot 0 is the position (x, y, z, 1/w) with a0[0][0] = 0, slots
    // 1 to NumInputs() the fragment inputs. Values at pixel centre (X + 0.5, Y + 0.5) are
    // a0 + dadx * X + dady * Y.
    const float (*a0)[4] = nullptr;
    const float (*dadx)[4] = nullptr;
    const float (*dady)[4] = nullptr;
    const DrawState *state = nullptr;
    // Block origin in each color buffer and in the depth buffer, null when absent.
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

// Lane coordinates X = lane_x + x and Y = lane_y + y as floats, exact.
void LaneCoordinates(const FragmentBlock &b, float *X, float *Y);

// The plane value fma(dady, Y, fma(dadx, X, a0)) of component c of a slot, per lane. FTZ.
void InterpolatePlane(const FragmentBlock &b, int slot, int c, const float *X, const float *Y,
                      float *out);

// A perspective input: the plane of slot (already times 1/w) times fdiv(1, w) with w the plane of
// slot 0 component 3, per lane. `one_over_w` is fdiv(1, w) from PerspectiveOneOverW.
void PerspectiveOneOverW(const FragmentBlock &b, const float *X, const float *Y, float *one_over_w);
void InterpolatePerspective(const FragmentBlock &b, int slot, int c, const float *X, const float *Y,
                            const float *one_over_w, float *out);

// gl_FragCoord: x = X + 0.5, y = Y + 0.5 in a framebuffer object, H - (Y + 0.5) in the window.
void FragCoord(const FragmentBlock &b, const float *X, const float *Y, float *fx, float *fy);

// Depth. z = plane of slot 0 component 2 at the lane plus a0[0][0], which setup sets to 0.
void InterpolateDepth(const FragmentBlock &b, const float *X, const float *Y, float *z);
// D16 of the fragment shaders: clamp max(min(z, 1), 0), so NaN gives 1, then
// bits(fl(fl(z * 0x3F7FFF00) + 128)) & 0xFFFF.
void DepthToD16(const float *z, std::uint16_t *d16);
// The depth test of the draw state's function on the stored D16 values (unsigned compare), for
// the lanes in `lanes`. Returns the lanes that pass. Without a depth test all lanes pass.
std::uint16_t DepthTest(const FragmentBlock &b, const std::uint16_t *d16, std::uint16_t lanes);
// Stores d16 for the lanes in `lanes` when the depth mask allows writes.
void DepthWrite(const FragmentBlock &b, const std::uint16_t *d16, std::uint16_t lanes);

// RGBA16F color buffers. Values are SoA, 16 lanes per channel r, g, b, a.
void LoadRgba16f(const FragmentBlock &b, int buffer, float *rgba);
// Writes the lanes in `lanes` as round-toward-zero halves. The other lanes are read and written
// back through float, as the JIT's blend of partial blocks does, which keeps every value except
// that a signaling NaN half comes back quiet.
void StoreRgba16f(const FragmentBlock &b, int buffer, const float *rgba, std::uint16_t lanes);
// Blend ONE, ONE: fl(src + dst) in IEEE arithmetic, then round toward zero, for `lanes`; the other
// lanes as StoreRgba16f.
void BlendAddRgba16f(const FragmentBlock &b, int buffer, const float *rgba, std::uint16_t lanes);

// The 10-bit default framebuffer, B in bits 0 to 9, G 10 to 19, R 20 to 29, X2 in 30 to 31.
// unorm16 of a channel already clamped to [0, 1]: bits(fl(fl(c * 0x3F7FFF00) + 128)) & 0xFFFF.
inline std::uint16_t Unorm16Ftz(float c) {
    return static_cast<std::uint16_t>(Bits(ftz::Add(ftz::Mul(c, FromBits(0x3F7FFF00u)), 128.0f)) &
                                      0xFFFFu);
}
// The word for 16-bit channels: each 10-bit channel is u16 >> 6, the two top bits
// ((a16 >> 2) * 3 + 8192) >> 14.
inline std::uint32_t Pack10(std::uint32_t r16, std::uint32_t g16, std::uint32_t b16,
                            std::uint32_t a16) {
    const std::uint32_t a2 = (((a16 >> 2) * 3u) + 8192u) >> 14;
    return (b16 >> 6) | ((g16 >> 6) << 10) | ((r16 >> 6) << 20) | ((a2 & 3u) << 30);
}
// A stored channel expanded to 16 bits as the JIT's partial blocks and blends read it.
inline std::uint32_t Expand10To16(std::uint32_t v10) { return (v10 << 6) | (v10 >> 4); }
inline std::uint32_t Expand2To16(std::uint32_t v2) { return v2 * 0x5555u; }
// Writes words for the lanes in `lanes`. `words` holds 16 values in lane order.
void Store10(const FragmentBlock &b, int buffer, const std::uint32_t *words, std::uint16_t lanes);
void Load10(const FragmentBlock &b, int buffer, std::uint32_t *words);

// The overlay's 16-bit fixed point blend: x * y / 65535 as (t + (t >> 16) + 0x8000) >> 16 with
// t = x * y in 32 bits, and the saturating add.
inline std::uint32_t Mul16(std::uint32_t x, std::uint32_t y) {
    const std::uint32_t t = x * y;
    return (t + (t >> 16) + 0x8000u) >> 16;
}
inline std::uint32_t AddSat16(std::uint32_t a, std::uint32_t b) {
    const std::uint32_t s = a + b;
    return s > 0xFFFFu ? 0xFFFFu : s;
}

}  // namespace cuju::render::gl

#endif  // CUJU_RENDER_GL_FRAGMENT_H
