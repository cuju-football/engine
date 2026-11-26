// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports GRF's overlay shaders (Apache-2.0) as llvmpipe's JIT compiles them (Mesa, MIT licence,
// the fragment shader and the blend of normalized formats in
// src/gallium/drivers/llvmpipe/lp_state_fs.c and gallivm/lp_bld_blend_aos.c).

#include "overlay.h"

#include "../fpenv.h"
#include "../gl/context.h"
#include "../gl/fragment.h"
#include "../sampler.h"
#include "quad_vs.h"

namespace cuju::render::shaders {
namespace {
constexpr int kLanes = 16;

inline std::uint32_t StoreUnorm16(float c) {
    return gl::Unorm16Ftz(ftz::MinPs(ftz::MaxPs(c, 0.0f), 1.0f));
}

}  // namespace

OverlayProgram::OverlayProgram()
    : gl::Program({{"model", gl::kMat4, 1},
                   {"projection", gl::kMat4, 1},
                   {"ourTexture", gl::kSampler2D, 1}},
                  {{"vertex", 0}}, {gl::Interp::kPerspective}) {}

void OverlayProgram::RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                                 float *clip, float *inputs) const {
    const float *v = attribs[0];
    OrthoQuadVertices(state.UniformPtr(kModel), state.UniformPtr(kProjection), count, v, clip);
    for (int i = 0; i < count; ++i) {
        inputs[4 * i + 0] = v[4 * i + 0];
        inputs[4 * i + 1] = v[4 * i + 1];
        inputs[4 * i + 2] = 0.0f;
        inputs[4 * i + 3] = 0.0f;
    }
}

void OverlayProgram::ShadeBlock(const gl::FragmentBlock &b) const {
    const Texture *tex = b.state->SamplerTexture(kOurTexture);
    if (tex == nullptr || tex->levels.empty()) {
        return;
    }
    float X[kLanes], Y[kLanes], oow[kLanes], s[kLanes], t[kLanes];
    gl::LaneCoordinates(b, X, Y);
    gl::PerspectiveOneOverW(b, X, Y, oow);
    gl::InterpolatePerspective(b, 1, 0, X, Y, oow, s);
    gl::InterpolatePerspective(b, 1, 1, X, Y, oow, t);
    float rgba[4 * kLanes];
    SampleQuads(SamplerKind::kAnisoLinearClamp, *tex, 4, s, t, rgba);

    const gl::FragmentOps &ops = b.state->ops;
    const bool blend = ops.blend && ops.blend_src == gl::GL_SRC_ALPHA &&
                       ops.blend_dst == gl::GL_ONE_MINUS_SRC_ALPHA;
    std::uint32_t dst[kLanes];
    if (blend) {
        gl::Load10(b, 0, dst);
    }
    std::uint32_t words[kLanes];
    for (int l = 0; l < kLanes; ++l) {
        const std::uint32_t r = StoreUnorm16(rgba[l]);
        const std::uint32_t g = StoreUnorm16(rgba[kLanes + l]);
        const std::uint32_t bl = StoreUnorm16(rgba[2 * kLanes + l]);
        const std::uint32_t a = StoreUnorm16(rgba[3 * kLanes + l]);
        if (!blend) {
            words[l] = gl::Pack10(r, g, bl, a);
            continue;
        }
        const std::uint32_t w = dst[l];
        const std::uint32_t db = gl::Expand10To16(w & 0x3FFu);
        const std::uint32_t dg = gl::Expand10To16((w >> 10) & 0x3FFu);
        const std::uint32_t dr = gl::Expand10To16((w >> 20) & 0x3FFu);
        const std::uint32_t da = gl::Expand2To16(w >> 30);
        const std::uint32_t inv = 0xFFFFu - a;
        words[l] = gl::Pack10(gl::AddSat16(gl::Mul16(r, a), gl::Mul16(dr, inv)),
                              gl::AddSat16(gl::Mul16(g, a), gl::Mul16(dg, inv)),
                              gl::AddSat16(gl::Mul16(bl, a), gl::Mul16(db, inv)),
                              gl::AddSat16(gl::Mul16(a, a), gl::Mul16(da, inv)));
    }
    gl::Store10(b, 0, words, gl::LaneMaskFromCoverage(b.coverage));
}

CUJU_GL_REGISTER_PROGRAM(kOverlay, OverlayProgram);

}  // namespace cuju::render::shaders
