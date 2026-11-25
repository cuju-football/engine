// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports GRF's postprocess shaders (Apache-2.0) as llvmpipe's JIT compiles them (Mesa, MIT licence,
// the fragment shader and 10-bit store of src/gallium/drivers/llvmpipe/lp_state_fs.c).

#include "postprocess.h"

#include "../fpenv.h"
#include "../gl/fragment.h"
#include "../lanes.h"
#include "../sampler.h"
#include "quad_vs.h"

namespace cuju::render::shaders {
namespace {
constexpr int kLanes = 16;

const float kHundredth = FromBits(0x3C23D70Au);
const float kFogBias = FromBits(0x3E23D70Au);
const float kFogRed = FromBits(0x3F59999Au);
const float kFogBlue = FromBits(0x3F666666u);
const float kSkyDepth = FromBits(0x3F7FBE77u);
const float kSsaoWeight = FromBits(0x3F19999Au);
const float kSsaoBase = FromBits(0x3ECCCCCDu);
const float kSaturation = FromBits(0x3F733333u);
const float kLumaB = FromBits(0x3D93A92Au);
const float kLumaG = FromBits(0x3F372474u);
const float kLumaR = FromBits(0x3E59999Au);
const float kContrastKeep = FromBits(0x3F333333u);
const float kPi = FromBits(0x40490FDBu);
const float kContrastBias = FromBits(0x3E99999Au);

inline std::uint32_t StoreUnorm16(float c) {
    return gl::Unorm16Ftz(ftz::MinPs(ftz::MaxPs(c, 0.0f), 1.0f));
}

}

PostprocessProgram::PostprocessProgram()
    : gl::Program({{"orthoViewMatrix", gl::kMat4, 1},
                   {"orthoProjectionMatrix", gl::kMat4, 1},
                   {"map_accumulation", gl::kSampler2D, 1},
                   {"map_modifier", gl::kSampler2D, 1},
                   {"map_depth", gl::kSampler2D, 1},
                   {"contextWidth", gl::kFloat, 1},
                   {"contextHeight", gl::kFloat, 1},
                   {"contextX", gl::kFloat, 1},
                   {"contextY", gl::kFloat, 1},
                   {"cameraClip", gl::kVec2, 1},
                   {"fogScale", gl::kFloat, 1}},
                  {{"position", 0}}, {}) {}

void PostprocessProgram::RunVertices(const gl::DrawState &state, int count,
                                     const float *const *attribs, float *clip, float *) const {
    OrthoQuadVertices(state.UniformPtr(kOrthoViewMatrix), state.UniformPtr(kOrthoProjectionMatrix),
                      count, attribs[0], clip);
}

namespace {
using lanes::F8;
using lanes::I8;

inline F8 FetchChannel(const lanes::NearestLevel &l, I8 x, I8 y, F8 s, F8 t, int c) {
    using namespace lanes;
    if (l.format == Format::R16G16B16A16_FLOAT) {
        const I8 offset = AddI(ShlI(AddI(MulLoI(y, SplatI(l.w)), x), 3), SplatI(2 * (c & 2)));
        const I8 pair = Gather32(l.data, offset);
        return HalvesToFloatI((c & 1) ? SrlI(pair, 16) : AndI(pair, SplatI(0xFFFF)));
    }
    F8 rgba[4];
    FetchNearest(l, s, t, rgba);
    return rgba[c];
}

inline void FetchRgb(const lanes::NearestLevel &l, F8 s, F8 t, F8 *rgb) {
    F8 rgba[4];
    lanes::FetchNearest(l, s, t, rgba);
    rgb[0] = rgba[0];
    rgb[1] = rgba[1];
    rgb[2] = rgba[2];
}

}

void PostprocessProgram::ShadeBlock(const gl::FragmentBlock &b) const {
    using namespace lanes;
    const gl::DrawState &st = *b.state;
    const Texture *accum_tex = st.SamplerTexture(kMapAccumulation);
    const Texture *modifier_tex = st.SamplerTexture(kMapModifier);
    const Texture *depth_tex = st.SamplerTexture(kMapDepth);
    if (accum_tex == nullptr || modifier_tex == nullptr || depth_tex == nullptr) {
        return;
    }
    const NearestLevel accum = NearestLevelOf(*accum_tex);
    const NearestLevel modifier = NearestLevelOf(*modifier_tex);
    const NearestLevel depth = NearestLevelOf(*depth_tex);
    const float cw = st.Uniform(kContextWidth);
    const float ch = st.Uniform(kContextHeight);
    const F8 neg_cx = Splat(-st.Uniform(kContextX));
    const F8 neg_cy = Splat(-st.Uniform(kContextY));
    const F8 neg_clip_near = Splat(-st.Uniform(kCameraClip, 0));
    const F8 clip_far = Splat(st.Uniform(kCameraClip, 1));
    const float fog_scale = st.Uniform(kFogScale);
    const F8 wpos_scale = Splat(st.target.window ? -1.0f : 1.0f);
    const F8 wpos_offset = Splat(st.target.window ? static_cast<float>(st.target.height) : 0.0f);
    const F8 zero = Splat(0.0f), half = Splat(0.5f), one = Splat(1.0f);

    const F8 up = Splat(ftz::Div(1.0f, ch));
    const F8 right = Splat(ftz::Div(1.0f, cw));
    const F8 down = Splat(ftz::Div(-1.0f, ch));
    const F8 left = Splat(ftz::Div(-1.0f, cw));
    const float inv_w = ftz::Div(1.0f, static_cast<float>(modifier_tex->Width(0)));
    const float inv_h = ftz::Div(1.0f, static_cast<float>(modifier_tex->Height(0)));
    static const float kOffsets[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
    F8 ox[4], oy[4];
    for (int i = 0; i < 4; ++i) {
        ox[i] = Splat(ftz::Mul(kOffsets[i], inv_w));
        oy[i] = Splat(ftz::Mul(kOffsets[i], inv_h));
    }
    const F8 fog_keep_scale = Splat(ftz::Add(1.0f, -fog_scale));
    const F8 fog_bias = Splat(-ftz::Mul(kFogBias, fog_scale));
    const std::uint32_t alpha_bits = gl::Pack10(0, 0, 0, StoreUnorm16(0.0f));

    float X[kLanes], Y[kLanes];
    gl::LaneCoordinates(b, X, Y);
    alignas(32) std::uint32_t words[kLanes];
    for (int h = 0; h < 2; ++h) {
        const int o = 8 * h;
        const F8 fx = Add(Load(X + o), half);
        const F8 fy = Add(Mul(Add(Load(Y + o), half), wpos_scale), wpos_offset);
        const F8 tx = Div(Add(fx, neg_cx), Splat(cw));
        const F8 ty = Div(Add(fy, neg_cy), Splat(ch));

        F8 base[3];
        FetchRgb(accum, tx, ty, base);
        const I8 mx = NearestIndex(tx, modifier.fw, modifier.w);
        const I8 my = NearestIndex(ty, modifier.fh, modifier.h);
        const F8 mod_r = FetchChannel(modifier, mx, my, tx, ty, 0);

        const F8 blur = Olt(zero, mod_r);
        if (MoveMask(blur) != 0) {
            F8 t0[3], t1[3], sum[3];
            FetchRgb(accum, tx, Add(ty, up), t0);
            FetchRgb(accum, Add(tx, right), ty, t1);
            for (int c = 0; c < 3; ++c) sum[c] = Add(t0[c], t1[c]);
            FetchRgb(accum, tx, Add(ty, down), t0);
            for (int c = 0; c < 3; ++c) sum[c] = Add(sum[c], t0[c]);
            FetchRgb(accum, Add(tx, left), ty, t1);
            for (int c = 0; c < 3; ++c) sum[c] = Add(sum[c], t1[c]);
            const F8 keep = Add(one, Neg(mod_r));
            for (int c = 0; c < 3; ++c) {
                const F8 blurred = Add(Mul(base[c], keep), Mul(Mul(sum[c], Splat(0.25f)), mod_r));
                base[c] = Select(blur, blurred, base[c]);
            }
        }

        I8 sx[4], sy[4];
        F8 fsx[4], fsy[4];
        for (int i = 0; i < 4; ++i) {
            fsx[i] = Add(tx, ox[i]);
            fsy[i] = Add(ty, oy[i]);
            sx[i] = NearestIndex(fsx[i], modifier.fw, modifier.w);
            sy[i] = NearestIndex(fsy[i], modifier.fh, modifier.h);
        }
        F8 ssao = FetchChannel(modifier, sx[0], sy[0], fsx[0], fsy[0], 1);
        for (int k = 1; k < 16; ++k) {
            const int i = k >> 2, j = k & 3;
            ssao = Add(ssao, FetchChannel(modifier, sx[i], sy[j], fsx[i], fsy[j], 1));
        }
        ssao = Div(ssao, Splat(16.0f));

        F8 drgba[4];
        FetchNearest(depth, tx, ty, drgba);
        const F8 d = drgba[0];
        F8 r = Mul(base[0], ssao);
        F8 g = Mul(base[1], ssao);
        F8 bl = Mul(base[2], ssao);
        const F8 frag_depth = Div(clip_far, Add(d, neg_clip_near));
        const F8 f0 = Add(Mul(Mul(frag_depth, Splat(kHundredth)), fog_keep_scale), fog_bias);
        const F8 fog = MinPs(MaxPs(MinPs(f0, Splat(0.25f)), zero), one);
        const F8 keep = Add(one, Neg(fog));
        const F8 fog_rg = Mul(Splat(kFogRed), fog);
        const F8 fog_b = Mul(Splat(kFogBlue), fog);
        r = Add(Mul(r, keep), fog_rg);
        g = Add(Mul(g, keep), fog_rg);
        bl = Add(Mul(bl, keep), fog_b);
        const F8 sky = Olt(Splat(kSkyDepth), d);
        r = Select(sky, Splat(kFogRed), r);
        g = Select(sky, Splat(kFogRed), g);
        bl = Select(sky, Splat(kFogBlue), bl);
        const F8 sat =
            Mul(Splat(kSaturation), Add(Splat(kSsaoBase), Mul(ssao, Splat(kSsaoWeight))));
        const F8 luma =
            Add(Add(Mul(Splat(kLumaB), bl), Mul(Splat(kLumaG), g)), Mul(Splat(kLumaR), r));
        const F8 grey = Mul(luma, Add(one, Neg(sat)));
        const F8 c[3] = {Add(grey, Mul(r, sat)), Add(grey, Mul(g, sat)), Add(grey, Mul(bl, sat))};
        I8 u16[3];
        for (int k = 0; k < 3; ++k) {
            const F8 kept = Mul(c[k], Splat(kContrastKeep));
            const F8 clamped = MinPs(MaxPs(c[k], zero), one);
            const F8 cosv = Cos(Mul(clamped, Splat(kPi)));
            const F8 curve = Add(Neg(Mul(cosv, half)), half);
            const F8 out = Add(kept, Mul(curve, Splat(kContrastBias)));
            const F8 store = MinPs(MaxPs(MinPs(MaxPs(out, zero), one), zero), one);
            u16[k] = lanes::Unorm16(store);
        }
        const I8 w =
            OrI(OrI(SrlI(u16[2], 6), ShlI(SrlI(u16[1], 6), 10)),
                OrI(ShlI(SrlI(u16[0], 6), 20), SplatI(static_cast<std::int32_t>(alpha_bits))));
        StoreI(reinterpret_cast<std::int32_t *>(words + o), w);
    }
    gl::Store10(b, 0, words, gl::LaneMaskFromCoverage(b.coverage));
}

CUJU_GL_REGISTER_PROGRAM(kPostprocess, PostprocessProgram);

}
