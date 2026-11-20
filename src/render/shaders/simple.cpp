// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports GRF's simple shaders (Apache-2.0) as llvmpipe's JIT compiles them (Mesa, MIT licence).

#include "simple.h"

#include <cstring>

#include "../fpenv.h"
#include "../gl/fragment.h"
#include "../lanes.h"
#include "../sampler.h"

namespace cuju::render::shaders {

SimpleProgram::SimpleProgram()
    : gl::Program(
          {{"modelMatrix", gl::kMat4, 1},
           {"viewMatrix", gl::kMat4, 1},
           {"projectionMatrix", gl::kMat4, 1},
           {"map_albedo", gl::kSampler2D, 1},
           {"map_normal", gl::kSampler2D, 1},
           {"map_specular", gl::kSampler2D, 1},
           {"map_illumination", gl::kSampler2D, 1},
           {"materialparams", gl::kVec3, 1},
           {"materialbools", gl::kVec3, 1}},
          {{"position", 0}, {"normal", 1}, {"texcoord", 2}, {"tangent", 3}, {"bitangent", 4}},
          {gl::Interp::kPerspective, gl::Interp::kPerspective, gl::Interp::kPerspective}) {}

namespace {

// m * v for the upper 3x3 of a column-major matrix, ((m0r * x + m1r * y) + m2r * z).
inline void Mul3(const float *m, const float *v, float *out) {
    for (int r = 0; r < 3; ++r) {
        float s = ftz::Mul(m[0 * 3 + r], v[0]);
        s = ftz::Add(s, ftz::Mul(m[1 * 3 + r], v[1]));
        out[r] = ftz::Add(s, ftz::Mul(m[2 * 3 + r], v[2]));
    }
}

}  // namespace

// frag_position = V * (M * pos) with pos.w = 1 folded away, gl_Position = P * frag_position. The
// normal matrix transpose(inverse(mat3(M))) is the uniform-only cofactor form with nine divides by
// the determinant, applied to normal, tangent and bitangent. Fragment inputs, packed as the JIT
// packs the varyings: (normal, s), (t, tangent), (bitangent, 0).
void SimpleProgram::RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                                float *clip, float *inputs) const {
    const float *M = state.UniformPtr(kModelMatrix);
    const float *V = state.UniformPtr(kViewMatrix);
    const float *P = state.UniformPtr(kProjectionMatrix);
    const float a0 = M[0], a1 = M[1], a2 = M[2];
    const float b0 = M[4], b1 = M[5], b2 = M[6];
    const float c0 = M[8], c1 = M[9], c2 = M[10];
    const float k77 = ftz::Add(ftz::Mul(b1, c2), -ftz::Mul(c1, b2));
    const float k81 = ftz::Add(ftz::Mul(b0, c2), -ftz::Mul(c0, b2));
    const float k85 = ftz::Add(ftz::Mul(b0, c1), -ftz::Mul(c0, b1));
    const float k90 = ftz::Add(ftz::Mul(a1, c2), -ftz::Mul(c1, a2));
    const float k95 = ftz::Add(ftz::Mul(a0, c2), -ftz::Mul(c0, a2));
    const float k99 = ftz::Add(ftz::Mul(a0, c1), -ftz::Mul(c0, a1));
    const float k104 = ftz::Add(ftz::Mul(a1, b2), -ftz::Mul(b1, a2));
    const float k108 = ftz::Add(ftz::Mul(a0, b2), -ftz::Mul(b0, a2));
    const float k113 = ftz::Add(ftz::Mul(a0, b1), -ftz::Mul(b0, a1));
    const float det = ftz::Add(ftz::Add(ftz::Mul(a0, k77), -ftz::Mul(a1, k81)), ftz::Mul(a2, k85));
    // N, column-major: column 0 multiplies x, column 1 y, column 2 z.
    float N[9];
    N[0] = ftz::Div(k77, det);
    N[3] = ftz::Div(-k90, det);
    N[6] = ftz::Div(k104, det);
    N[1] = ftz::Div(-k81, det);
    N[4] = ftz::Div(k95, det);
    N[7] = ftz::Div(-k108, det);
    N[2] = ftz::Div(k85, det);
    N[5] = ftz::Div(-k99, det);
    N[8] = ftz::Div(k113, det);

    const float *pos = attribs[0];
    const float *nrm = attribs[1];
    const float *tex = attribs[2];
    const float *tan = attribs[3];
    const float *bit = attribs[4];
    for (int i = 0; i < count; ++i) {
        const float x = pos[4 * i + 0], y = pos[4 * i + 1], z = pos[4 * i + 2];
        float m[4], e[4];
        for (int r = 0; r < 4; ++r) {
            float s = ftz::Mul(M[0 * 4 + r], x);
            s = ftz::Add(s, ftz::Mul(M[1 * 4 + r], y));
            s = ftz::Add(s, ftz::Mul(M[2 * 4 + r], z));
            m[r] = ftz::Add(s, M[3 * 4 + r]);
        }
        for (int r = 0; r < 4; ++r) {
            float s = ftz::Mul(V[0 * 4 + r], m[0]);
            s = ftz::Add(s, ftz::Mul(V[1 * 4 + r], m[1]));
            s = ftz::Add(s, ftz::Mul(V[2 * 4 + r], m[2]));
            e[r] = ftz::Add(s, ftz::Mul(V[3 * 4 + r], m[3]));
        }
        for (int r = 0; r < 4; ++r) {
            float s = ftz::Mul(P[0 * 4 + r], e[0]);
            s = ftz::Add(s, ftz::Mul(P[1 * 4 + r], e[1]));
            s = ftz::Add(s, ftz::Mul(P[2 * 4 + r], e[2]));
            clip[4 * i + r] = ftz::Add(s, ftz::Mul(P[3 * 4 + r], e[3]));
        }
        float *in = inputs + 12 * i;
        Mul3(N, nrm + 4 * i, in + 0);
        in[3] = tex[4 * i + 0];
        in[4] = tex[4 * i + 1];
        Mul3(N, tan + 4 * i, in + 5);
        Mul3(N, bit + 4 * i, in + 8);
        in[11] = 0.0f;
    }
}

namespace {

// Quads of a lane mask, bit q for quad q.
inline unsigned QuadMask(std::uint16_t lanes) {
    unsigned q = 0;
    for (int k = 0; k < 4; ++k) {
        if ((lanes >> (4 * k)) & 0xFu) {
            q |= 1u << k;
        }
    }
    return q;
}

// texture() of one unit for the quads in `quads` and the channels in `channels`, out[c * 16 +
// lane]; lanes of other quads are 0. GRF's material textures are mipmapped and sampled by
// llvmpipe's anisotropic trilinear texfunc. An incomplete texture samples Mesa's fallback, a 1x1
// RGBA8 texel (0, 0, 0, 255), as the inline nearest code of the variants without a texfunc for that
// unit.
void SampleUnit(const Texture *tex, unsigned quads, const float *s, const float *t,
                unsigned channels, float *out) {
    if (tex == nullptr || tex->levels.size() < 2) {
        for (int l = 0; l < 16; ++l) {
            out[0 * 16 + l] = 0.0f;
            out[1 * 16 + l] = 0.0f;
            out[2 * 16 + l] = 0.0f;
            out[3 * 16 + l] = 1.0f;
        }
        return;
    }
    if (quads == 0xFu) {
        SampleAnisoTrilinearQuads(*tex, 4, s, t, channels, out);
        return;
    }
    float cs[16], ct[16], co[64];
    int n = 0;
    for (int q = 0; q < 4; ++q) {
        if (quads & (1u << q)) {
            std::memcpy(cs + 4 * n, s + 4 * q, 4 * sizeof(float));
            std::memcpy(ct + 4 * n, t + 4 * q, 4 * sizeof(float));
            ++n;
        }
    }
    SampleAnisoTrilinearQuads(*tex, n, cs, ct, channels, co);
    std::memset(out, 0, 64 * sizeof(float));
    int k = 0;
    for (int q = 0; q < 4; ++q) {
        if (quads & (1u << q)) {
            for (int c = 0; c < 4; ++c) {
                if (!(channels & (1u << c))) {
                    continue;
                }
                std::memcpy(out + c * 16 + 4 * q, co + c * 4 * n + 4 * k, 4 * sizeof(float));
            }
            ++k;
        }
    }
}

}  // namespace

// fs157: early D16 LESS test, albedo with discard on alpha < 0.12 (olt), the materialbools
// branches, bump mapping, normalize as v * fdiv(1, sqrt(dot)) with the dot in reverse component
// order, three RGBA16F outputs stored round toward zero, then the depth store for the lanes left.
void SimpleProgram::ShadeBlock(const gl::FragmentBlock &b) const {
    float X[16], Y[16], z[16];
    std::uint16_t d16[16];
    gl::LaneCoordinates(b, X, Y);
    gl::InterpolateDepth(b, X, Y, z);
    gl::DepthToD16(z, d16);
    const std::uint16_t pass = gl::DepthTest(b, d16, gl::LaneMaskFromCoverage(b.coverage));
    // Lanes outside the mask are rewritten unchanged by the JIT's store, so nothing to do.
    if (pass == 0) {
        return;
    }
    const gl::DrawState &st = *b.state;

    alignas(32) float oow[16];
    alignas(32) float v[11][16];  // normal xyz, s, t, tangent xyz, bitangent xyz
    gl::PerspectiveOneOverW(b, X, Y, oow);
    for (int c = 0; c < 4; ++c) {
        gl::InterpolatePerspective(b, 1, c, X, Y, oow, v[c]);
        gl::InterpolatePerspective(b, 2, c, X, Y, oow, v[4 + c]);
    }
    for (int c = 0; c < 3; ++c) {
        gl::InterpolatePerspective(b, 3, c, X, Y, oow, v[8 + c]);
    }
    const float *s = v[3];
    const float *t = v[4];

    alignas(32) float albedo[64];
    SampleUnit(st.SamplerTexture(kMapAlbedo), QuadMask(pass), s, t, 15u, albedo);
    using namespace lanes;
    const F8 discard = Splat(FromBits(0x3DF5C28Fu));
    const unsigned keep = static_cast<unsigned>(MoveMask(Olt(Load(albedo + 48), discard)) |
                                                MoveMask(Olt(Load(albedo + 56), discard)) << 8);
    const std::uint16_t exec = static_cast<std::uint16_t>(~keep & pass);
    if (exec == 0) {
        return;
    }
    const unsigned quads = QuadMask(exec);
    const F8 one = Splat(1.0f), two = Splat(2.0f), minus_one = Splat(-1.0f);

    alignas(32) float bx[16], by[16], bz[16];
    if (ftz::Oeq(st.Uniform(kMaterialBools, 0), 1.0f)) {
        float n[64];
        SampleUnit(st.SamplerTexture(kMapNormal), quads, s, t, 7u, n);
        for (int h = 0; h < 16; h += 8) {
            const F8 x = Add(Mul(Load(n + h), two), minus_one);
            const F8 y = Add(Mul(Load(n + 16 + h), two), minus_one);
            const F8 w = Add(Mul(Load(n + 32 + h), two), minus_one);
            const F8 len2 = Add(Add(Mul(w, w), Mul(y, y)), Mul(x, x));
            const F8 inv = Div(one, Sqrt(len2));
            Store(bx + h, Mul(x, inv));
            Store(by + h, Mul(y, inv));
            Store(bz + h, Mul(w, inv));
        }
    } else {
        for (int h = 0; h < 16; h += 8) {
            Store(bx + h, Splat(0.0f));
            Store(by + h, Splat(0.0f));
            Store(bz + h, one);
        }
    }
    alignas(32) float spec[16], illum[16];
    const float spec_amount = st.Uniform(kMaterialParams, 1);
    if (ftz::Oeq(st.Uniform(kMaterialBools, 1), 1.0f)) {
        float m[64];
        SampleUnit(st.SamplerTexture(kMapSpecular), quads, s, t, 1u, m);
        for (int h = 0; h < 16; h += 8) {
            Store(spec + h, Mul(Load(m + h), Splat(spec_amount)));
        }
    } else {
        for (int h = 0; h < 16; h += 8) {
            Store(spec + h, Splat(spec_amount));
        }
    }
    if (ftz::Oeq(st.Uniform(kMaterialBools, 2), 1.0f)) {
        float m[64];
        SampleUnit(st.SamplerTexture(kMapIllumination), quads, s, t, 1u, m);
        std::memcpy(illum, m, sizeof(illum));
    } else {
        const F8 self_illumination = Splat(st.Uniform(kMaterialParams, 2));
        for (int h = 0; h < 16; h += 8) {
            Store(illum + h, self_illumination);
        }
    }

    alignas(32) float out0[64], out1[64], out2[64];
    const F8 shininess = Splat(st.Uniform(kMaterialParams, 0));
    for (int h = 0; h < 16; h += 8) {
        const F8 bxh = Load(bx + h), byh = Load(by + h), bzh = Load(bz + h);
        F8 bn[3];
        for (int c = 0; c < 3; ++c) {
            const F8 tb = Add(Mul(bxh, Load(v[5 + c] + h)), Mul(byh, Load(v[8 + c] + h)));
            bn[c] = Add(tb, Mul(bzh, Load(v[c] + h)));
        }
        const F8 len2 = Add(Add(Mul(bn[2], bn[2]), Mul(bn[1], bn[1])), Mul(bn[0], bn[0]));
        const F8 inv = Div(one, Sqrt(len2));
        for (int c = 0; c < 3; ++c) {
            Store(out0 + c * 16 + h, Load(albedo + c * 16 + h));
            Store(out1 + c * 16 + h, Mul(bn[c], inv));
            Store(out2 + c * 16 + h, Splat(0.0f));
        }
        Store(out0 + 48 + h, Load(spec + h));
        Store(out1 + 48 + h, shininess);
        Store(out2 + 48 + h, Load(illum + h));
    }
    gl::StoreRgba16f(b, 0, out0, exec);
    gl::StoreRgba16f(b, 1, out1, exec);
    gl::StoreRgba16f(b, 2, out2, exec);
    gl::DepthWrite(b, d16, exec);
}

CUJU_GL_REGISTER_PROGRAM(kSimple, SimpleProgram);

}  // namespace cuju::render::shaders
