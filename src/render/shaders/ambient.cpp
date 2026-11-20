// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports GRF's ambient shaders (Apache-2.0) as llvmpipe's JIT compiles them (Mesa, MIT licence).

#include "ambient.h"

#include "../fpenv.h"
#include "../gl/fragment.h"
#include "../lanes.h"
#include "../numerics.h"
#include "../sampler.h"
#include "quad_vs.h"

namespace cuju::render::shaders {

namespace {

using lanes::F8;
using lanes::I8;

constexpr int kLanes = 16;

// The red channel of a kNearestClamp fetch.
inline F8 FetchRed(const lanes::NearestLevel &l, F8 s, F8 t) {
    F8 rgba[4];
    lanes::FetchNearest(l, s, t, rgba);
    return rgba[0];
}

// NIR's fmin and fmax where neither operand is a constant: minps(a, b), or a when b is NaN, so
// the result is NaN only when both are.
inline F8 MinNan(F8 a, F8 b) { return lanes::Select(lanes::IsNan(b), a, lanes::MinPs(a, b)); }
inline F8 MaxNan(F8 a, F8 b) { return lanes::Select(lanes::IsNan(b), a, lanes::MaxPs(a, b)); }
inline F8 Clamp01(F8 x) {
    return lanes::MinPs(lanes::MaxPs(x, lanes::Splat(0.0f)), lanes::Splat(1.0f));
}

// The linear depth cameraClip.y / (d - cameraClip.x), as d + -x then an IEEE divide.
inline F8 LinearDepth(F8 d, F8 neg_clip_x, F8 clip_y) {
    return lanes::Div(clip_y, lanes::Add(d, neg_clip_x));
}

// mat4 * (x, y, z, 1) row r, column-major, ((m0r * x + m4r * y) + m8r * z) + m12r.
inline F8 Affine(const float *m, int r, F8 x, F8 y, F8 z) {
    using namespace lanes;
    return Add(Add(Add(Mul(Splat(m[r]), x), Mul(Splat(m[4 + r]), y)), Mul(Splat(m[8 + r]), z)),
               Splat(m[12 + r]));
}

}  // namespace

AmbientProgram::AmbientProgram()
    : gl::Program({{"orthoViewMatrix", gl::kMat4, 1},
                   {"orthoProjectionMatrix", gl::kMat4, 1},
                   {"map_albedo", gl::kSampler2D, 1},
                   {"map_normal", gl::kSampler2D, 1},
                   {"map_depth", gl::kSampler2D, 1},
                   {"map_aux", gl::kSampler2D, 1},
                   {"map_noise", gl::kSampler2D, 1},
                   {"viewMatrix", gl::kMat4, 1},
                   {"projectionMatrix", gl::kMat4, 1},
                   {"inverseProjectionViewMatrix", gl::kMat4, 1},
                   {"contextWidth", gl::kFloat, 1},
                   {"contextHeight", gl::kFloat, 1},
                   {"contextX", gl::kFloat, 1},
                   {"contextY", gl::kFloat, 1},
                   {"cameraClip", gl::kVec2, 1},
                   {"SSAO_kernel[0]", gl::kVec3, kKernelSize}},
                  {{"position", 0}}, {}) {}

void AmbientProgram::RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                                 float *clip, float *) const {
    OrthoQuadVertices(state.UniformPtr(kOrthoViewMatrix), state.UniformPtr(kOrthoProjectionMatrix),
                      count, attribs[0], clip);
}

// fs159, one half of eight lanes at a time as the JIT code runs it. Every fetch but the noise is
// nearest with clamp to edge; the noise is linear, anisotropic, repeat, on its one level. Shader
// arithmetic is separate fmul and fadd in the IR's operand order, divides are IEEE.
void AmbientProgram::ShadeBlock(const gl::FragmentBlock &b) const {
    using namespace lanes;
    const gl::DrawState &st = *b.state;
    const Texture *albedo_tex = st.SamplerTexture(kMapAlbedo);
    const Texture *normal_tex = st.SamplerTexture(kMapNormal);
    const Texture *depth_tex = st.SamplerTexture(kMapDepth);
    const Texture *aux_tex = st.SamplerTexture(kMapAux);
    const Texture *noise_tex = st.SamplerTexture(kMapNoise);
    if (!albedo_tex || !normal_tex || !depth_tex || !aux_tex || !noise_tex) {
        return;
    }
    const NearestLevel albedo = NearestLevelOf(*albedo_tex);
    const NearestLevel normal = NearestLevelOf(*normal_tex);
    const NearestLevel depth = NearestLevelOf(*depth_tex);
    const NearestLevel aux = NearestLevelOf(*aux_tex);
    const float *V = st.UniformPtr(kViewMatrix);
    const float *P = st.UniformPtr(kProjectionMatrix);
    const float *I = st.UniformPtr(kInverseProjectionViewMatrix);
    const float cw = st.Uniform(kContextWidth);
    const float ch = st.Uniform(kContextHeight);
    const F8 neg_cx = Splat(-st.Uniform(kContextX));
    const F8 neg_cy = Splat(-st.Uniform(kContextY));
    const F8 neg_clip_x = Splat(-st.Uniform(kCameraClip, 0));
    const F8 clip_y = Splat(st.Uniform(kCameraClip, 1));
    // gl_FragCoord.y through Mesa's window position transform, (1, 0) in a framebuffer object.
    const bool window = st.target.window;
    const F8 wpos_scale = Splat(window ? -1.0f : 1.0f);
    const F8 wpos_offset = Splat(window ? static_cast<float>(st.target.height) : 0.0f);

    const F8 zero = Splat(0.0f), half = Splat(0.5f), one = Splat(1.0f), two = Splat(2.0f);
    const F8 minus_one = Splat(-1.0f);
    const F8 c0_7 = Splat(FromBits(0x3F333333u));
    const F8 c0_3 = Splat(FromBits(0x3E99999Au));
    const F8 tint[3] = {Splat(FromBits(0x3E0A3D71u)),   // 0.9 * 0.15
                        Splat(FromBits(0x3E19999Au)),   // 0.15
                        Splat(FromBits(0x3E3851ECu))};  // 1.2 * 0.15
    const F8 radius = Splat(FromBits(0x3E3851ECu));     // 0.18
    const F8 min_delta = Splat(FromBits(0x3727C5ACu));  // 0.00001
    const float milli = FromBits(0x3A83126Fu);          // 0.001
    const float noise_sx = ftz::Div(cw, 4.0f);
    const float noise_sy = ftz::Div(ch, 4.0f);
    // GetEdge's offsets: scalar divides, then * 0.001.
    const float ow1 = ftz::Mul(cw, milli);
    const float oh1 = ftz::Mul(ch, milli);
    const F8 dxm = Splat(ftz::Mul(ftz::Div(-1.0f, ow1), milli));
    const F8 dym = Splat(ftz::Mul(ftz::Div(-1.0f, oh1), milli));
    const F8 dxp = Splat(ftz::Mul(ftz::Div(1.0f, ow1), milli));
    const F8 dyp = Splat(ftz::Mul(ftz::Div(1.0f, oh1), milli));

    float X[kLanes], Y[kLanes];
    gl::LaneCoordinates(b, X, Y);
    float out0[4 * kLanes], out1[4 * kLanes];
    // What each half keeps from its setup for the kernel and the end.
    struct Half {
        F8 tx, ty, d, n0[3], col[3], vpx, vpy, vpz, vn[3], tg[3], bt[3], occlusion;
    };
    Half halves[2];
    for (int h = 0; h < 2; ++h) {
        Half &q = halves[h];
        const int o = 8 * h;
        const F8 fx = Add(Load(X + o), half);
        const F8 fy = Add(Mul(Add(Load(Y + o), half), wpos_scale), wpos_offset);
        const F8 tx = Div(Add(fx, neg_cx), Splat(cw));
        const F8 ty = Div(Add(fy, neg_cy), Splat(ch));

        const F8 d = FetchRed(depth, tx, ty);
        F8 base[4], n0[4];
        FetchNearest(albedo, tx, ty, base);
        FetchNearest(normal, tx, ty, n0);

        // World position from the inverse projection-view matrix.
        const F8 px = Add(Mul(tx, two), minus_one);
        const F8 py = Add(Mul(ty, two), minus_one);
        const F8 pz = Add(Mul(d, two), minus_one);
        const F8 ww = Affine(I, 3, px, py, pz);
        const F8 wx = Div(Affine(I, 0, px, py, pz), ww);
        const F8 wy = Div(Affine(I, 1, px, py, pz), ww);
        const F8 wz = Div(Affine(I, 2, px, py, pz), ww);

        // The blueish tint of the desaturated base colour.
        const F8 avg = Mul(Div(Add(Add(base[0], base[1]), base[2]), Splat(3.0f)), c0_7);
        F8 col[3];
        for (int c = 0; c < 3; ++c) {
            col[c] = Mul(Add(avg, Mul(base[c], c0_3)), tint[c]);
        }

        // View position and view normal.
        const F8 vpx = Affine(V, 0, wx, wy, wz);
        const F8 vpy = Affine(V, 1, wx, wy, wz);
        const F8 vpz = Affine(V, 2, wx, wy, wz);
        F8 vn[3];
        for (int r = 0; r < 3; ++r) {
            vn[r] = Add(Add(Mul(Splat(V[r]), n0[0]), Mul(Splat(V[4 + r]), n0[1])),
                        Mul(Splat(V[8 + r]), n0[2]));
        }

        // The noise, tiled every 4 pixels, then the tangent frame: Gram-Schmidt of the random
        // vector against the view normal and the cross product.
        float ns[8], nt[8], noise[32];
        Store(ns, Mul(tx, Splat(noise_sx)));
        Store(nt, Mul(ty, Splat(noise_sy)));
        SampleQuads(SamplerKind::kAnisoLinearRepeat, *noise_tex, 2, ns, nt, noise);
        F8 rv[3];
        for (int c = 0; c < 3; ++c) {
            rv[c] = Add(Neg(Mul(Load(noise + 8 * c), two)), minus_one);
        }
        const F8 rdn = Add(Add(Mul(rv[2], vn[2]), Mul(rv[1], vn[1])), Mul(rv[0], vn[0]));
        F8 u[3];
        for (int c = 0; c < 3; ++c) {
            u[c] = Add(rv[c], Neg(Mul(vn[c], rdn)));
        }
        const F8 inv = Div(one, Sqrt(Add(Add(Mul(u[2], u[2]), Mul(u[1], u[1])), Mul(u[0], u[0]))));
        const F8 tg[3] = {Mul(u[0], inv), Mul(u[1], inv), Mul(u[2], inv)};
        const F8 bt[3] = {Add(Mul(vn[1], tg[2]), Neg(Mul(vn[2], tg[1]))),
                          Add(Mul(vn[2], tg[0]), Neg(Mul(vn[0], tg[2]))),
                          Add(Mul(vn[0], tg[1]), Neg(Mul(vn[1], tg[0])))};
        q.tx = tx;
        q.ty = ty;
        q.d = d;
        q.vpx = vpx;
        q.vpy = vpy;
        q.vpz = vpz;
        for (int c = 0; c < 3; ++c) {
            q.n0[c] = n0[c];
            q.col[c] = col[c];
            q.vn[c] = vn[c];
            q.tg[c] = tg[c];
            q.bt[c] = bt[c];
        }
        q.occlusion = zero;
    }

    // Occlusion over the kernel. Speed: the two halves step through the kernel together, several
    // entries at a time, each stage for all of them before the next, so that the processor
    // overlaps their independent chains; each lane still runs its operations, and adds its terms,
    // in kernel order.
    constexpr int kStep = 2;
    constexpr int kChains = 2 * kStep;
    static_assert(kKernelSize % kStep == 0, "the kernel steps kStep entries at a time");
    for (int i = 0; i < kKernelSize; i += kStep) {
        F8 kx[kStep], ky[kStep], kz[kStep];
        for (int e = 0; e < kStep; ++e) {
            const float *kv = st.UniformPtr(kSsaoKernel + i + e);
            kx[e] = Splat(kv[0]);
            ky[e] = Splat(kv[1]);
            kz[e] = Splat(kv[2]);
        }
        // Chain c is half c / kStep at kernel entry i + c % kStep.
        F8 spx[kChains], spy[kChains], spz[kChains], su[kChains], sv[kChains], lin[kChains];
        F8 term[kChains];
        for (int c = 0; c < kChains; ++c) {
            const Half &q = halves[c / kStep];
            const int e = c % kStep;
            const F8 sx = Add(Add(Mul(q.tg[0], kx[e]), Mul(q.bt[0], ky[e])), Mul(q.vn[0], kz[e]));
            const F8 sy = Add(Add(Mul(q.tg[1], kx[e]), Mul(q.bt[1], ky[e])), Mul(q.vn[1], kz[e]));
            const F8 sz = Add(Add(Mul(q.tg[2], kx[e]), Mul(q.bt[2], ky[e])), Mul(q.vn[2], kz[e]));
            spx[c] = Add(q.vpx, Mul(sx, radius));
            spy[c] = Add(q.vpy, Mul(sy, radius));
            spz[c] = Add(q.vpz, Mul(sz, radius));
        }
        for (int c = 0; c < kChains; ++c) {
            const F8 ow = Affine(P, 3, spx[c], spy[c], spz[c]);
            su[c] = Add(Mul(Div(Affine(P, 0, spx[c], spy[c], spz[c]), ow), half), half);
            sv[c] = Add(Mul(Div(Affine(P, 1, spx[c], spy[c], spz[c]), ow), half), half);
        }
        for (int c = 0; c < kChains; ++c) {
            lin[c] = LinearDepth(FetchRed(depth, su[c], sv[c]), neg_clip_x, clip_y);
        }
        for (int c = 0; c < kChains; ++c) {
            const F8 dist = Mul(Abs(Add(Neg(halves[c / kStep].vpz), Neg(lin[c]))), half);
            const F8 falloff = Clamp01(Div(Add(radius, Neg(dist)), radius));
            const F8 inside = OneIf(Olt(lin[c], Neg(spz[c])));
            term[c] = Mul(falloff, inside);
        }
        for (int c = 0; c < kChains; ++c) {
            Half &q = halves[c / kStep];
            q.occlusion = Add(q.occlusion, term[c]);
        }
    }

    for (int h = 0; h < 2; ++h) {
        const Half &q = halves[h];
        const int o = 8 * h;
        const F8 tx = q.tx, ty = q.ty, d = q.d;
        const F8 *n0 = q.n0, *col = q.col;
        const F8 ssao = Clamp01(
            Add(Mul(Add(one, Neg(Div(q.occlusion, Splat(32.0f)))), Splat(1.5f)), Splat(-0.5f)));

        F8 ax[4];
        FetchNearest(aux, tx, ty, ax);
        const F8 illum = Add(one, ax[3]);
        for (int c = 0; c < 3; ++c) {
            Store(out0 + c * kLanes + o, Mul(Clamp01(Mul(col[c], illum)), ssao));
        }
        Store(out0 + 3 * kLanes + o, one);

        // GetEdge: the eight neighbours one pixel away, round the clock from the lower left.
        const F8 xm = Add(tx, dxm), ym = Add(ty, dym), xp = Add(tx, dxp), yp = Add(ty, dyp);
        const F8 tap_s[8] = {xm, tx, xp, xp, xp, tx, xm, xm};
        const F8 tap_t[8] = {ym, ym, ym, ty, yp, yp, yp, ty};
        F8 tap_depth[8], tap_dot[8];
        for (int j = 0; j < 8; ++j) {
            F8 nk[4];
            tap_depth[j] = LinearDepth(FetchRed(depth, tap_s[j], tap_t[j]), neg_clip_x, clip_y);
            FetchNearest(normal, tap_s[j], tap_t[j], nk);
            tap_dot[j] = Add(Add(Mul(nk[2], n0[2]), Mul(nk[1], n0[1])), Mul(nk[0], n0[0]));
        }
        const F8 d0 = LinearDepth(d, neg_clip_x, clip_y);
        F8 result[4];
        for (int j = 0; j < 4; ++j) {
            const F8 d1 = Abs(Add(tap_depth[j], Neg(d0)));
            const F8 d2 = Abs(Add(d0, Neg(tap_depth[j + 4])));
            const F8 min_d = MaxPs(MinNan(d1, d2), min_delta);
            const F8 depth_edge = Oge(MaxNan(d1, d2), Mul(min_d, Splat(35.0f)));
            const F8 normal_edge = Oge(Abs(Add(tap_dot[j], Neg(tap_dot[j + 4]))), half);
            result[j] = OneIf(Or(normal_edge, depth_edge));
        }
        Store(out1 + o,
              Mul(Add(Add(result[0], result[1]), Add(result[2], result[3])), Splat(0.25f)));
        Store(out1 + kLanes + o, ssao);
        Store(out1 + 2 * kLanes + o, zero);
        Store(out1 + 3 * kLanes + o, zero);
    }

    const std::uint16_t lanes_mask = gl::LaneMaskFromCoverage(b.coverage);
    gl::StoreRgba16f(b, 0, out0, lanes_mask);
    gl::StoreRgba16f(b, 1, out1, lanes_mask);
}

CUJU_GL_REGISTER_PROGRAM(kAmbient, AmbientProgram);

}  // namespace cuju::render::shaders
