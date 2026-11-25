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
    kAnisoTrilinearRepeat,
    kAnisoLinearRepeat,
    kAnisoLinearClamp,
    kNearestClamp,
    kNearestRepeat,
    kLinearClampLod0,
};

inline bool ForcesAlphaOne(Format format) {
    return format != Format::R8G8B8A8_SRGB && format != Format::R8G8B8A8_UNORM &&
           format != Format::R16G16B16A16_FLOAT;
}

void SampleQuads(SamplerKind kind, const Texture &texture, int quads, const float *s,
                 const float *t, float *out);

void SampleAnisoTrilinearQuads(const Texture &texture, int quads, const float *s, const float *t,
                               unsigned channels, float *out);

void SampleLevelQuads(SamplerKind kind, Format format, const TextureLevel &level, int quads,
                      const float *s, const float *t, float *out);

void SampleShadowQuads(const Texture &texture, int quads, const float *s, const float *t,
                       const float *ref, float *out);

}

#endif
