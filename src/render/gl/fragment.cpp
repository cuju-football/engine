// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports llvmpipe's per-pixel helpers (Mesa, MIT licence,
// src/gallium/drivers/llvmpipe/lp_state_fs.c, src/gallium/auxiliary/gallivm/lp_bld_interp.c and
// lp_bld_depth.c).

#include "fragment.h"

#include <cstring>

#include "../lanes.h"
#include "pipeline.h"

namespace cuju::render::gl {
void LaneCoordinates(const FragmentBlock &b, float *X, float *Y) {
    const float bx = static_cast<float>(b.x);
    const float by = static_cast<float>(b.y);
    for (int l = 0; l < 16; ++l) {
        X[l] = static_cast<float>(kLaneX[l]) + bx;
        Y[l] = static_cast<float>(kLaneY[l]) + by;
    }
}

void InterpolatePlane(const FragmentBlock &b, int slot, int c, const float *X, const float *Y,
                      float *out) {
    using namespace lanes;
    const F8 a0 = Splat(b.a0[slot][c]);
    const F8 dx = Splat(b.dadx[slot][c]);
    const F8 dy = Splat(b.dady[slot][c]);
    for (int h = 0; h < 16; h += 8) {
        Store(out + h, Fma(dy, Load(Y + h), Fma(dx, Load(X + h), a0)));
    }
}

void PerspectiveOneOverW(const FragmentBlock &b, const float *X, const float *Y,
                         float *one_over_w) {
    using namespace lanes;
    float w[16];
    InterpolatePlane(b, 0, 3, X, Y, w);
    for (int h = 0; h < 16; h += 8) {
        Store(one_over_w + h, Div(Splat(1.0f), Load(w + h)));
    }
}

void InterpolatePerspective(const FragmentBlock &b, int slot, int c, const float *X, const float *Y,
                            const float *one_over_w, float *out) {
    using namespace lanes;
    InterpolatePlane(b, slot, c, X, Y, out);
    for (int h = 0; h < 16; h += 8) {
        Store(out + h, Mul(Load(out + h), Load(one_over_w + h)));
    }
}

void FragCoord(const FragmentBlock &b, const float *X, const float *Y, float *fx, float *fy) {
    const bool window = b.state->target.window;
    const float h = static_cast<float>(b.state->target.height);
    for (int l = 0; l < 16; ++l) {
        fx[l] = ftz::Add(X[l], 0.5f);
        const float ry = ftz::Add(Y[l], 0.5f);
        fy[l] = window ? ftz::Fma(ry, -1.0f, h) : ry;
    }
}

void InterpolateDepth(const FragmentBlock &b, const float *X, const float *Y, float *z) {
    using namespace lanes;
    InterpolatePlane(b, 0, 2, X, Y, z);
    const F8 a = Splat(b.a0[0][0]);
    for (int h = 0; h < 16; h += 8) {
        Store(z + h, Add(Load(z + h), a));
    }
}

void DepthToD16(const float *z, std::uint16_t *d16) {
    using namespace lanes;
    for (int h = 0; h < 16; h += 8) {
        alignas(32) std::int32_t v[8];
        StoreI(v, lanes::Unorm16(MaxPs(MinPs(Load(z + h), Splat(1.0f)), Splat(0.0f))));
        for (int l = 0; l < 8; ++l) {
            d16[h + l] = static_cast<std::uint16_t>(v[l]);
        }
    }
}

namespace {
bool DepthPasses(std::uint32_t func, std::uint32_t frag, std::uint32_t stored) {
    switch (func) {
        case 0x0200:
            return false;
        case 0x0201:
            return frag < stored;
        case 0x0202:
            return frag == stored;
        case 0x0203:
            return frag <= stored;
        case 0x0204:
            return frag > stored;
        case 0x0205:
            return frag != stored;
        case 0x0206:
            return frag >= stored;
        default:
            return true;
    }
}

inline std::uint8_t *LanePtr(std::uint8_t *base, int stride, int lane, int bpp) {
    return base + kLaneY[lane] * stride + kLaneX[lane] * bpp;
}

}

#if CUJU_RENDER_LANES_AVX2

namespace {
inline __m128i LoadDepth8(const std::uint8_t *base, int stride, int h) {
    const std::uint8_t *row0 = base + (2 * h) * stride;
    const __m128i r0 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(row0));
    const __m128i r1 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(row0 + stride));
    return _mm_unpacklo_epi32(r0, r1);
}

inline unsigned Mask16Bits(__m128i m) {
    return static_cast<unsigned>(_mm_movemask_epi8(_mm_packs_epi16(m, _mm_setzero_si128())));
}

}

std::uint16_t DepthTest(const FragmentBlock &b, const std::uint16_t *d16, std::uint16_t lanes) {
    const FragmentOps &ops = b.state->ops;
    if (!ops.depth_test || b.depth == nullptr) {
        return lanes;
    }
    unsigned pass = 0;
    for (int h = 0; h < 2; ++h) {
        if (((lanes >> (8 * h)) & 0xFFu) == 0) {
            continue;
        }
        const __m128i stored = LoadDepth8(b.depth, b.depth_stride, h);
        const __m128i frag = _mm_loadu_si128(reinterpret_cast<const __m128i *>(d16 + 8 * h));
        const __m128i eq = _mm_cmpeq_epi16(frag, stored);
        const __m128i le = _mm_cmpeq_epi16(_mm_min_epu16(frag, stored), frag);
        const __m128i ones = _mm_set1_epi16(-1);
        __m128i m;
        switch (ops.depth_func) {
            case 0x0200:
                m = _mm_setzero_si128();
                break;
            case 0x0201:
                m = _mm_andnot_si128(eq, le);
                break;
            case 0x0202:
                m = eq;
                break;
            case 0x0203:
                m = le;
                break;
            case 0x0204:
                m = _mm_xor_si128(le, ones);
                break;
            case 0x0205:
                m = _mm_xor_si128(eq, ones);
                break;
            case 0x0206:
                m = _mm_or_si128(_mm_xor_si128(le, ones), eq);
                break;
            default:
                m = ones;
                break;
        }
        pass |= Mask16Bits(m) << (8 * h);
    }
    return static_cast<std::uint16_t>(pass & lanes);
}

void DepthWrite(const FragmentBlock &b, const std::uint16_t *d16, std::uint16_t lanes) {
    const FragmentOps &ops = b.state->ops;
    if (!ops.depth_test || !ops.depth_write || b.depth == nullptr) {
        return;
    }
    for (int h = 0; h < 2; ++h) {
        const unsigned bits = (lanes >> (8 * h)) & 0xFFu;
        if (bits == 0) {
            continue;
        }
        std::uint8_t *row0 = b.depth + (2 * h) * b.depth_stride;
        std::uint8_t *row1 = row0 + b.depth_stride;
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(d16 + 8 * h));
        if (bits != 0xFFu) {
            const __m128i sel = _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128);
            const __m128i m =
                _mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(static_cast<short>(bits)), sel), sel);
            v = _mm_blendv_epi8(LoadDepth8(b.depth, b.depth_stride, h), v, m);
        }
        const __m128i rows = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 1, 2, 0));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(row0), rows);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(row1), _mm_unpackhi_epi64(rows, rows));
    }
}

#else

std::uint16_t DepthTest(const FragmentBlock &b, const std::uint16_t *d16, std::uint16_t lanes) {
    const FragmentOps &ops = b.state->ops;
    if (!ops.depth_test || b.depth == nullptr) {
        return lanes;
    }
    std::uint16_t pass = 0;
    for (int l = 0; l < 16; ++l) {
        if (!(lanes & (1u << l))) {
            continue;
        }
        std::uint16_t stored;
        std::memcpy(&stored, LanePtr(b.depth, b.depth_stride, l, 2), 2);
        if (DepthPasses(ops.depth_func, d16[l], stored)) {
            pass = static_cast<std::uint16_t>(pass | (1u << l));
        }
    }
    return pass;
}

void DepthWrite(const FragmentBlock &b, const std::uint16_t *d16, std::uint16_t lanes) {
    const FragmentOps &ops = b.state->ops;
    if (!ops.depth_test || !ops.depth_write || b.depth == nullptr) {
        return;
    }
    for (int l = 0; l < 16; ++l) {
        if (lanes & (1u << l)) {
            std::memcpy(LanePtr(b.depth, b.depth_stride, l, 2), &d16[l], 2);
        }
    }
}

#endif

#if CUJU_RENDER_LANES_F16C

namespace {
inline void LoadHalves8(const std::uint8_t *base, int stride, int h, __m128i *rgba) {
    const std::uint8_t *row0 = base + (2 * h) * stride;
    const std::uint8_t *row1 = row0 + stride;
    const __m128i p0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row0));
    const __m128i p2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row0 + 16));
    const __m128i p1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row1));
    const __m128i p3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row1 + 16));
    const __m128i t0 = _mm_unpacklo_epi16(p0, p1);
    const __m128i t1 = _mm_unpackhi_epi16(p0, p1);
    const __m128i t2 = _mm_unpacklo_epi16(p2, p3);
    const __m128i t3 = _mm_unpackhi_epi16(p2, p3);
    const __m128i u0 = _mm_unpacklo_epi16(t0, t1);
    const __m128i u1 = _mm_unpackhi_epi16(t0, t1);
    const __m128i v0 = _mm_unpacklo_epi16(t2, t3);
    const __m128i v1 = _mm_unpackhi_epi16(t2, t3);
    rgba[0] = _mm_unpacklo_epi64(u0, v0);
    rgba[1] = _mm_unpackhi_epi64(u0, v0);
    rgba[2] = _mm_unpacklo_epi64(u1, v1);
    rgba[3] = _mm_unpackhi_epi64(u1, v1);
}

inline void StoreHalves8(std::uint8_t *base, int stride, int h, const __m128i *rgba) {
    std::uint8_t *row0 = base + (2 * h) * stride;
    std::uint8_t *row1 = row0 + stride;
    const __m128i rg_lo = _mm_unpacklo_epi16(rgba[0], rgba[1]);
    const __m128i rg_hi = _mm_unpackhi_epi16(rgba[0], rgba[1]);
    const __m128i ba_lo = _mm_unpacklo_epi16(rgba[2], rgba[3]);
    const __m128i ba_hi = _mm_unpackhi_epi16(rgba[2], rgba[3]);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(row0), _mm_unpacklo_epi32(rg_lo, ba_lo));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(row1), _mm_unpackhi_epi32(rg_lo, ba_lo));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(row0 + 16), _mm_unpacklo_epi32(rg_hi, ba_hi));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(row1 + 16), _mm_unpackhi_epi32(rg_hi, ba_hi));
}

inline __m128i Requiet(__m128i x) {
    const __m128i nan =
        _mm_cmpgt_epi16(_mm_and_si128(x, _mm_set1_epi16(0x7FFF)), _mm_set1_epi16(0x7C00));
    return _mm_or_si128(x, _mm_and_si128(nan, _mm_set1_epi16(0x200)));
}

inline __m128i LaneMask8(unsigned bits) {
    const __m128i b = _mm_set1_epi16(static_cast<short>(bits));
    const __m128i sel = _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128);
    return _mm_cmpeq_epi16(_mm_and_si128(b, sel), sel);
}

inline __m128i Select16(__m128i mask, __m128i t, __m128i f) { return _mm_blendv_epi8(f, t, mask); }

}

void StoreRgba16f(const FragmentBlock &b, int buffer, const float *rgba, std::uint16_t lanes) {
    std::uint8_t *base = b.color[buffer];
    const int stride = b.color_stride[buffer];
    for (int h = 0; h < 2; ++h) {
        const unsigned bits = (lanes >> (8 * h)) & 0xFFu;
        __m128i v[4];
        for (int c = 0; c < 4; ++c) {
            v[c] = _mm256_cvtps_ph(_mm256_loadu_ps(rgba + c * 16 + 8 * h), _MM_FROUND_TO_ZERO);
        }
        if (bits != 0xFFu) {
            __m128i old[4];
            LoadHalves8(base, stride, h, old);
            const __m128i m = LaneMask8(bits);
            for (int c = 0; c < 4; ++c) {
                v[c] = Select16(m, v[c], Requiet(old[c]));
            }
        }
        StoreHalves8(base, stride, h, v);
    }
}

void LoadRgba16f(const FragmentBlock &b, int buffer, float *rgba) {
    for (int h = 0; h < 2; ++h) {
        __m128i v[4];
        LoadHalves8(b.color[buffer], b.color_stride[buffer], h, v);
        for (int c = 0; c < 4; ++c) {
            _mm256_storeu_ps(rgba + c * 16 + 8 * h, _mm256_cvtph_ps(v[c]));
        }
    }
}

namespace {
CUJU_GL_NOINLINE void BlendAddKernel(const FragmentBlock *b, int buffer, const float *rgba,
                                     std::uint16_t lanes) {
    std::uint8_t *base = b->color[buffer];
    const int stride = b->color_stride[buffer];
    for (int h = 0; h < 2; ++h) {
        const __m128i m = LaneMask8((lanes >> (8 * h)) & 0xFFu);
        __m128i v[4];
        LoadHalves8(base, stride, h, v);
        for (int c = 0; c < 4; ++c) {
            const __m256 sum =
                _mm256_add_ps(_mm256_loadu_ps(rgba + c * 16 + 8 * h), _mm256_cvtph_ps(v[c]));
            v[c] = Select16(m, _mm256_cvtps_ph(sum, _MM_FROUND_TO_ZERO), Requiet(v[c]));
        }
        StoreHalves8(base, stride, h, v);
    }
}

}

#else

void LoadRgba16f(const FragmentBlock &b, int buffer, float *rgba) {
    for (int l = 0; l < 16; ++l) {
        std::uint16_t h[4];
        std::memcpy(h, LanePtr(b.color[buffer], b.color_stride[buffer], l, 8), 8);
        for (int c = 0; c < 4; ++c) {
            rgba[c * 16 + l] = HalfToFloat(h[c]);
        }
    }
}

void StoreRgba16f(const FragmentBlock &b, int buffer, const float *rgba, std::uint16_t lanes) {
    for (int l = 0; l < 16; ++l) {
        std::uint8_t *p = LanePtr(b.color[buffer], b.color_stride[buffer], l, 8);
        std::uint16_t h[4];
        if (lanes & (1u << l)) {
            for (int c = 0; c < 4; ++c) {
                h[c] = HalfRz(rgba[c * 16 + l]);
            }
        } else {
            std::memcpy(h, p, 8);
            for (int c = 0; c < 4; ++c) {
                h[c] = HalfRz(HalfToFloat(h[c]));
            }
        }
        std::memcpy(p, h, 8);
    }
}

namespace {
CUJU_GL_NOINLINE void BlendAddKernel(const FragmentBlock *b, int buffer, const float *rgba,
                                     std::uint16_t lanes) {
    for (int l = 0; l < 16; ++l) {
        std::uint8_t *p = LanePtr(b->color[buffer], b->color_stride[buffer], l, 8);
        std::uint16_t h[4];
        std::memcpy(h, p, 8);
        for (int c = 0; c < 4; ++c) {
            const float dst = HalfToFloat(h[c]);
            h[c] = (lanes & (1u << l)) ? HalfRz(rgba[c * 16 + l] + dst) : HalfRz(dst);
        }
        std::memcpy(p, h, 8);
    }
}

}

#endif

void BlendAddRgba16f(const FragmentBlock &b, int buffer, const float *rgba, std::uint16_t lanes) {
    FpScope scope(FpMode::kIeee);
    BlendAddKernel(&b, buffer, rgba, lanes);
}

void Store10(const FragmentBlock &b, int buffer, const std::uint32_t *words, std::uint16_t lanes) {
    for (int l = 0; l < 16; ++l) {
        if (lanes & (1u << l)) {
            std::memcpy(LanePtr(b.color[buffer], b.color_stride[buffer], l, 4), &words[l], 4);
        }
    }
}

void Load10(const FragmentBlock &b, int buffer, std::uint32_t *words) {
    for (int l = 0; l < 16; ++l) {
        std::memcpy(&words[l], LanePtr(b.color[buffer], b.color_stride[buffer], l, 4), 4);
    }
}

}
