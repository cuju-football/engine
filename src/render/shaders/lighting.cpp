// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports GRF's lighting shaders (Apache-2.0) as llvmpipe's JIT compiles them (Mesa, MIT licence).
// pow is lp_build_pow over lp_build_log2 and lp_build_exp2 (Mesa's gallivm, lp_bld_arit.c, MIT
// licence), as polynomials.h has it for one lane.

#include "lighting.h"

#include "../fpenv.h"
#include "../gl/fragment.h"
#include "../lanes.h"
#include "../numerics.h"
#include "../sampler.h"

namespace cuju::render::shaders {

namespace {

using lanes::F8;
using lanes::I8;

constexpr int kLanes = 16;
constexpr int kShadowTaps = 9;

inline F8 K(std::uint32_t bits) { return lanes::Splat(FromBits(bits)); }

// lp_build_log2 with its special cases: NaN below zero, -inf at zero, +inf at +inf.
F8 Log2(F8 x) {
    using namespace lanes;
    const I8 b = AsInt(x);
    const F8 e = ToFloat(SubI(SrlI(AndI(b, SplatI(0x7F800000)), 23), SplatI(127)));
    const F8 m = AsFloat(OrI(AndI(b, SplatI(0x7FFFFF)), SplatI(0x3F800000)));
    const F8 one = Splat(1.0f);
    const F8 y = Div(Sub(m, one), Add(m, one));
    const F8 z = Mul(y, y);
    const F8 z2 = Mul(z, z);
    const F8 p_odd = Fma(z2, K(0x3ED03D59u), K(0x3F13D321u));
    const F8 p_even = Fma(z2, K(0x3ECE8316u), K(0x3F7637F9u));
    const F8 p_hi = Fma(z2, p_odd, K(0x4038AA3Bu));
    const F8 p = Fma(p_even, z, p_hi);
    F8 r = Fma(y, p, e);
    r = Select(Uge(x, K(0x7F800000u)), K(0x7F800000u), r);
    r = Select(Ueq(x, Splat(0.0f)), K(0xFF800000u), r);
    return Select(Ult(x, Splat(0.0f)), K(0x7FC00000u), r);
}

// lp_build_exp2: the input clamped to [-126.99999, 128], 2^floor times a polynomial of the
// fraction.
F8 Exp2(F8 x) {
    using namespace lanes;
    const F8 v = MaxPs(K(0xC2FDFFFFu), MinPs(Splat(128.0f), x));
    const F8 i = Floor(v);
    const F8 f = Sub(v, i);
    const F8 scale = AsFloat(ShlI(AddI(CvttPs2Dq(i), SplatI(127)), 23));
    const F8 f2 = Mul(f, f);
    const F8 q_hi = Fma(f2, K(0x3AF61905u), K(0x3D64AA23u));
    const F8 q_lo = Fma(f2, K(0x3C134806u), K(0x3E75EAD4u));
    const F8 q_odd = Fma(f2, q_hi, K(0x3F31727Bu));
    const F8 q_even = Fma(f2, q_lo, Splat(1.0f));
    return Mul(scale, Fma(q_odd, f, q_even));
}

// GLSL pow: zero for a zero base, else exp2(log2(x) * y).
F8 Pow(F8 x, F8 y) {
    using namespace lanes;
    return Select(Oeq(x, Splat(0.0f)), Splat(0.0f), Exp2(Mul(Log2(x), y)));
}

// mat4 * (x, y, z, 1) row r, column-major, ((m0r * x + m4r * y) + m8r * z) + m12r.
inline F8 Affine(const float *m, int r, F8 x, F8 y, F8 z) {
    using namespace lanes;
    return Add(Add(Add(Mul(Splat(m[r]), x), Mul(Splat(m[4 + r]), y)), Mul(Splat(m[8 + r]), z)),
               Splat(m[12 + r]));
}

// v * (1 / sqrt(dot(v, v))) with the dot product in reverse component order.
inline void Normalize(const F8 *v, F8 *out) {
    using namespace lanes;
    const F8 inv =
        Div(Splat(1.0f), Sqrt(Add(Add(Mul(v[2], v[2]), Mul(v[1], v[1])), Mul(v[0], v[0]))));
    for (int c = 0; c < 3; ++c) {
        out[c] = Mul(v[c], inv);
    }
}

inline F8 Dot3(const F8 *a, const F8 *b) {
    using namespace lanes;
    return Add(Add(Mul(a[2], b[2]), Mul(a[1], b[1])), Mul(a[0], b[0]));
}

}  // namespace

LightingProgram::LightingProgram()
    : gl::Program({{"modelMatrix", gl::kMat4, 1},
                   {"viewMatrix", gl::kMat4, 1},
                   {"projectionMatrix", gl::kMat4, 1},
                   {"map_albedo", gl::kSampler2D, 1},
                   {"map_normal", gl::kSampler2D, 1},
                   {"map_depth", gl::kSampler2D, 1},
                   {"map_shadow", gl::kSampler2DShadow, 1},
                   {"inverseProjectionViewMatrix", gl::kMat4, 1},
                   {"lightViewProjectionMatrix", gl::kMat4, 1},
                   {"contextWidth", gl::kFloat, 1},
                   {"contextHeight", gl::kFloat, 1},
                   {"contextX", gl::kFloat, 1},
                   {"contextY", gl::kFloat, 1},
                   {"has_shadow", gl::kBool, 1},
                   {"cameraPosition", gl::kVec3, 1},
                   {"lightPosition", gl::kVec3, 1},
                   {"lightColor", gl::kVec3, 1},
                   {"lightRadius", gl::kFloat, 1}},
                  {{"position", 0}}, {}) {}

// gl_Position = projectionMatrix * (viewMatrix * modelMatrix * position): the uniform-only product
// V * M per call, element (j, r) the column-major ((V0r * Mj0 + V1r * Mj1) + V2r * Mj2) + V3r *
// Mj3, then that product times the vec4 position and the projection times the result, all separate
// roundings under FTZ and DAZ. The varyings modelViewMatrix and modelViewPosition have no reader
// and are dropped by the linker.
void LightingProgram::RunVertices(const gl::DrawState &state, int count,
                                  const float *const *attribs, float *clip, float *) const {
    using ftz::Add;
    using ftz::Mul;
    const float *M = state.UniformPtr(kModelMatrix);
    const float *V = state.UniformPtr(kViewMatrix);
    const float *P = state.UniformPtr(kProjectionMatrix);
    float C[16];
    for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < 4; ++r) {
            float s = Mul(V[0 * 4 + r], M[j * 4 + 0]);
            s = Add(s, Mul(V[1 * 4 + r], M[j * 4 + 1]));
            s = Add(s, Mul(V[2 * 4 + r], M[j * 4 + 2]));
            s = Add(s, Mul(V[3 * 4 + r], M[j * 4 + 3]));
            C[j * 4 + r] = s;
        }
    }
    const float *pos = attribs[0];
    for (int i = 0; i < count; ++i) {
        const float *p = pos + 4 * i;
        float m[4];
        for (int r = 0; r < 4; ++r) {
            float s = Mul(C[0 * 4 + r], p[0]);
            s = Add(s, Mul(C[1 * 4 + r], p[1]));
            s = Add(s, Mul(C[2 * 4 + r], p[2]));
            m[r] = Add(s, Mul(C[3 * 4 + r], p[3]));
        }
        for (int r = 0; r < 4; ++r) {
            float s = Mul(P[0 * 4 + r], m[0]);
            s = Add(s, Mul(P[1 * 4 + r], m[1]));
            s = Add(s, Mul(P[2 * 4 + r], m[2]));
            clip[4 * i + r] = Add(s, Mul(P[3 * 4 + r], m[3]));
        }
    }
}

// fs158, one half of eight lanes at a time as the JIT code runs it. G-buffer fetches are nearest
// with clamp to edge, the shadow map is textureProj with a LEQUAL compare. Shader arithmetic is
// separate fmul and fadd in the IR's operand order, divides are IEEE, pow is the log2 and exp2
// polynomials. The colour is added to both targets with blend ONE, ONE in IEEE arithmetic.
void LightingProgram::ShadeBlock(const gl::FragmentBlock &b) const {
    using namespace lanes;
    const gl::DrawState &st = *b.state;
    const Texture *albedo_tex = st.SamplerTexture(kMapAlbedo);
    const Texture *normal_tex = st.SamplerTexture(kMapNormal);
    const Texture *depth_tex = st.SamplerTexture(kMapDepth);
    const bool has_shadow = st.UniformBits(kHasShadow) != 0;
    const Texture *shadow_tex = has_shadow ? st.SamplerTexture(kMapShadow) : nullptr;
    if (!albedo_tex || !normal_tex || !depth_tex || (has_shadow && !shadow_tex)) {
        return;
    }
    const NearestLevel albedo = NearestLevelOf(*albedo_tex);
    const NearestLevel normal = NearestLevelOf(*normal_tex);
    const NearestLevel depth = NearestLevelOf(*depth_tex);
    const float *I = st.UniformPtr(kInverseProjectionViewMatrix);
    const float *L = st.UniformPtr(kLightViewProjectionMatrix);
    const float cw = st.Uniform(kContextWidth);
    const float ch = st.Uniform(kContextHeight);
    const F8 neg_cx = Splat(-st.Uniform(kContextX));
    const F8 neg_cy = Splat(-st.Uniform(kContextY));
    const float *cam = st.UniformPtr(kCameraPosition);
    const float *light = st.UniformPtr(kLightPosition);
    const float *color = st.UniformPtr(kLightColor);
    const F8 radius = Splat(st.Uniform(kLightRadius));
    const bool window = st.target.window;
    const F8 wpos_scale = Splat(window ? -1.0f : 1.0f);
    const F8 wpos_offset = Splat(window ? static_cast<float>(st.target.height) : 0.0f);
    const F8 zero = Splat(0.0f), half = Splat(0.5f), one = Splat(1.0f), two = Splat(2.0f);
    const F8 minus_one = Splat(-1.0f);

    float X[kLanes], Y[kLanes];
    gl::LaneCoordinates(b, X, Y);
    float out0[4 * kLanes], out1[4 * kLanes] = {};
    F8 world[2][3];
    for (int h = 0; h < 2; ++h) {
        const int o = 8 * h;
        const F8 fx = Add(Load(X + o), half);
        const F8 fy = Add(Mul(Add(Load(Y + o), half), wpos_scale), wpos_offset);
        const F8 tx = Div(Add(fx, neg_cx), Splat(cw));
        const F8 ty = Div(Add(fy, neg_cy), Splat(ch));
        F8 d[4], ns[4], base[4];
        FetchNearest(depth, tx, ty, d);
        FetchNearest(normal, tx, ty, ns);
        FetchNearest(albedo, tx, ty, base);

        const F8 px = Add(Mul(tx, two), minus_one);
        const F8 py = Add(Mul(ty, two), minus_one);
        const F8 pz = Add(Mul(d[0], two), minus_one);
        const F8 ww = Affine(I, 3, px, py, pz);
        F8 *w = world[h];
        for (int c = 0; c < 3; ++c) {
            w[c] = Div(Affine(I, c, px, py, pz), ww);
        }

        F8 l0[3], ld[3], n[3], e0[3], e[3], refl[3];
        for (int c = 0; c < 3; ++c) {
            l0[c] = Add(Splat(light[c]), Neg(w[c]));
        }
        const F8 llen2 = Add(Add(Mul(l0[2], l0[2]), Mul(l0[1], l0[1])), Mul(l0[0], l0[0]));
        const F8 linv = Div(one, Sqrt(llen2));
        for (int c = 0; c < 3; ++c) {
            ld[c] = Mul(l0[c], linv);
        }
        Normalize(ns, n);

        const F8 ndotl = Dot3(ld, n);
        const F8 scatter = Pow(Add(Mul(ndotl, half), half), K(0x3FD9999Au));  // 1.7
        const F8 nd = MaxPs(ndotl, zero);
        const F8 fall = Div(MaxPs(Add(radius, Neg(Sqrt(llen2))), zero), radius);
        const F8 fall2 = Mul(fall, fall);
        const F8 resulting =
            Add(Mul(nd, K(0x3F19999Au)), Mul(scatter, K(0x3ECCCCCDu)));  // 0.6, 0.4
        const F8 lit = Mul(Mul(resulting, fall2), two);

        for (int c = 0; c < 3; ++c) {
            e0[c] = Add(w[c], Neg(Splat(cam[c])));
        }
        Normalize(e0, e);
        const F8 k2 = Mul(Dot3(n, e), two);
        for (int c = 0; c < 3; ++c) {
            refl[c] = Add(e[c], Neg(Mul(k2, n[c])));
        }
        const F8 rdotl = MaxPs(Dot3(refl, ld), zero);
        const F8 spec = Mul(Mul(Pow(rdotl, Mul(ns[3], Splat(128.0f))), base[3]), fall2);
        for (int c = 0; c < 3; ++c) {
            const F8 cc = Splat(color[c]);
            Store(out0 + c * kLanes + o, Add(Mul(base[c], Mul(lit, cc)), Mul(spec, cc)));
        }
        Store(out0 + 3 * kLanes + o, zero);
    }

    if (has_shadow) {
        // textureProj of lightViewProjectionMatrix * world plus the Poisson offsets / 1500 and the
        // depth offset, folded into constants; offsets of zero add nothing.
        const float c1 = FromBits(0x39AEC33Eu);  // 0.5 / 1500
        const float c2 = FromBits(0x3A2EC33Eu);  // 1 / 1500
        const float dx[kShadowTaps] = {c1, -c1, c1, -c1, c2, 0.0f, -c2, 0.0f, 0.0f};
        const float dy[kShadowTaps] = {c1, c1, -c1, -c1, 0.0f, c2, 0.0f, -c2, 0.0f};
        const F8 ninth = K(0x3DE38E39u);
        F8 q[2][4], shaded[2];
        for (int h = 0; h < 2; ++h) {
            for (int r = 0; r < 4; ++r) {
                q[h][r] = Affine(L, r, world[h][0], world[h][1], world[h][2]);
            }
            q[h][2] = Add(q[h][2], K(0xB951B717u));  // -0.0002
            shaded[h] = one;
        }
        float ps[kLanes], pt[kLanes], pr[kLanes], lit[kLanes];
        for (int k = 0; k < kShadowTaps; ++k) {
            for (int h = 0; h < 2; ++h) {
                const F8 x = dx[k] != 0.0f ? Add(q[h][0], Splat(dx[k])) : q[h][0];
                const F8 y = dy[k] != 0.0f ? Add(q[h][1], Splat(dy[k])) : q[h][1];
                const F8 rq = Div(one, q[h][3]);
                Store(ps + 8 * h, Mul(x, rq));
                Store(pt + 8 * h, Mul(y, rq));
                Store(pr + 8 * h, Mul(q[h][2], rq));
            }
            SampleShadowQuads(*shadow_tex, 4, ps, pt, pr, lit);
            for (int h = 0; h < 2; ++h) {
                shaded[h] = Add(shaded[h], Neg(Mul(Add(one, Neg(Load(lit + 8 * h))), ninth)));
            }
        }
        for (int h = 0; h < 2; ++h) {
            const F8 sh = Add(Mul(shaded[h], Splat(0.75f)), Splat(0.25f));
            for (int c = 0; c < 3; ++c) {
                float *p = out0 + c * kLanes + 8 * h;
                Store(p, Mul(Load(p), sh));
            }
        }
    }

    const std::uint16_t lanes_mask = gl::LaneMaskFromCoverage(b.coverage);
    if (st.ops.blend) {
        gl::BlendAddRgba16f(b, 0, out0, lanes_mask);
        gl::BlendAddRgba16f(b, 1, out1, lanes_mask);
    } else {
        gl::StoreRgba16f(b, 0, out0, lanes_mask);
        gl::StoreRgba16f(b, 1, out1, lanes_mask);
    }
}

CUJU_GL_REGISTER_PROGRAM(kLighting, LightingProgram);

}  // namespace cuju::render::shaders
