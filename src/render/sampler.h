// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Texture sampling as llvmpipe's SoA sampler code computes it for GRF's sampler states. The
// algorithms follow Mesa's gallivm, lp_bld_sample_soa.c and lp_bld_sample.c (MIT licence,
// Copyright VMware, Inc. and others).
//
// Lanes come in 2x2 quads, lane x + 2y of a quad at pixel (x, y) of the quad, y growing with the
// row. A 4x4 block of llvmpipe is four quads in the order of its lanes: (0, 0), (2, 0), (0, 2),
// (2, 2). Level of detail, anisotropy and the major axis are per quad, from its lanes 0, 1 and 2.
// Results are per lane and never depend on the other quads of a call. All functions run under
// FTZ and DAZ, so call them inside an FpScope(FpMode::kFtzDaz).

#ifndef CUJU_RENDER_SAMPLER_H
#define CUJU_RENDER_SAMPLER_H

#include <cstdint>

#include "texture.h"

namespace cuju::render {

enum class SamplerKind : std::uint8_t {
    // LINEAR_MIPMAP_LINEAR, anisotropy 16, REPEAT on power-of-two sizes, every level: albedo,
    // normal, specular and illumination maps. A texfunc.
    kAnisoTrilinearRepeat,
    // LINEAR, anisotropy 16, REPEAT, level 0 only: the ambient pass's noise. Inline code.
    kAnisoLinearRepeat,
    // LINEAR, anisotropy 16, CLAMP_TO_EDGE, level 0 only: the overlay. Inline code.
    kAnisoLinearClamp,
    // NEAREST, CLAMP_TO_EDGE, level 0: G-buffer, accumulation and depth reads. A texfunc.
    kNearestClamp,
    // NEAREST, REPEAT, level 0: units without a texfunc. Inline code.
    kNearestRepeat,
    // LINEAR, CLAMP_TO_EDGE, explicit level 0 of a one-level view: the mipmap blit. A texfunc.
    kLinearClampLod0,
};

// Formats without alpha return alpha exactly 1, through the swizzle of their GL_RGB view, even
// where the filtered alpha would be NaN.
inline bool ForcesAlphaOne(Format format) {
    return format != Format::R8G8B8A8_SRGB && format != Format::R8G8B8A8_UNORM &&
           format != Format::R16G16B16A16_FLOAT;
}

// Samples `quads` quads. s and t hold 4 * quads coordinates in lane order. The result is SoA:
// out[c * 4 * quads + lane] for channel c = r, g, b, a. A Z16 texture reads as (d, d, d, 1).
void SampleQuads(SamplerKind kind, const Texture &texture, int quads, const float *s,
                 const float *t, float *out);

// SampleQuads with kAnisoTrilinearRepeat for the channels in `channels` only, bit c for channel c.
// Channels outside the mask may be left unwritten; the others hold the bits SampleQuads gives.
void SampleAnisoTrilinearQuads(const Texture &texture, int quads, const float *s, const float *t,
                               unsigned channels, float *out);

// The same for the kinds that read one level, on a level given apart from its texture, as the
// mipmap blit's one-level view does. kAnisoTrilinearRepeat needs SampleQuads.
void SampleLevelQuads(SamplerKind kind, Format format, const TextureLevel &level, int quads,
                      const float *s, const float *t, float *out);

// The shadow map lookup of textureProj with LEQUAL compare: LINEAR, anisotropy 16, CLAMP_TO_EDGE
// on level 0 of a Z16 texture. `ref` is the projected r, clamped inside. The result is per lane,
// returned by llvmpipe in r, g and b with alpha 1.
void SampleShadowQuads(const Texture &texture, int quads, const float *s, const float *t,
                       const float *ref, float *out);

}  // namespace cuju::render

#endif  // CUJU_RENDER_SAMPLER_H
