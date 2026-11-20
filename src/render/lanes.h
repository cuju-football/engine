// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Eight lanes of FTZ stage arithmetic, half of a 4x4 block as llvmpipe shades it. On x86-64 hosts
// with hardware FTZ and AVX2 the operations are the AVX instructions the JIT code uses, which give
// per lane the bits of the scalar ones under the same MXCSR. Elsewhere each lane runs the `ftz`
// operation of fpenv.h, so both builds give the same bits. Masks are lanes of all ones or all
// zeros, as the vector compares produce them.

#ifndef CUJU_RENDER_LANES_H
#define CUJU_RENDER_LANES_H

#include <cstdint>
#include <cstring>

#include "fpenv.h"
#include "numerics.h"
#include "sampler.h"
#include "texture.h"

#if CUJU_RENDER_HARDWARE_FTZ && defined(__AVX2__)
#define CUJU_RENDER_LANES_AVX2 1
#include <immintrin.h>
#else
#define CUJU_RENDER_LANES_AVX2 0
#endif

#if CUJU_RENDER_LANES_AVX2 && (defined(__F16C__) || defined(_MSC_VER))
#define CUJU_RENDER_LANES_F16C 1
#else
#define CUJU_RENDER_LANES_F16C 0
#endif

namespace cuju::render::lanes {

#if CUJU_RENDER_LANES_AVX2

struct F8 {
    __m256 v;
};
struct I8 {
    __m256i v;
};

inline F8 Load(const float *p) { return {_mm256_loadu_ps(p)}; }
inline void Store(float *p, F8 a) { _mm256_storeu_ps(p, a.v); }
inline F8 Splat(float x) { return {_mm256_set1_ps(x)}; }
inline I8 LoadI(const std::int32_t *p) {
    return {_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p))};
}
inline void StoreI(std::int32_t *p, I8 a) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), a.v);
}
inline I8 SplatI(std::int32_t x) { return {_mm256_set1_epi32(x)}; }

inline F8 Add(F8 a, F8 b) { return {_mm256_add_ps(a.v, b.v)}; }
inline F8 Sub(F8 a, F8 b) { return {_mm256_sub_ps(a.v, b.v)}; }
inline F8 Mul(F8 a, F8 b) { return {_mm256_mul_ps(a.v, b.v)}; }
inline F8 Div(F8 a, F8 b) { return {_mm256_div_ps(a.v, b.v)}; }
inline F8 Sqrt(F8 a) { return {_mm256_sqrt_ps(a.v)}; }
inline F8 Fma(F8 a, F8 b, F8 c) { return {_mm256_fmadd_ps(a.v, b.v, c.v)}; }
inline F8 MinPs(F8 a, F8 b) { return {_mm256_min_ps(a.v, b.v)}; }
inline F8 MaxPs(F8 a, F8 b) { return {_mm256_max_ps(a.v, b.v)}; }
inline F8 Floor(F8 a) { return {_mm256_round_ps(a.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)}; }
inline F8 Ceil(F8 a) { return {_mm256_round_ps(a.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)}; }

inline F8 Olt(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)}; }
inline F8 Ole(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)}; }
inline F8 Ogt(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)}; }
inline F8 Oge(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ)}; }
inline F8 Oeq(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ)}; }
inline F8 Ult(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_NGE_UQ)}; }
inline F8 Ugt(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_NLE_UQ)}; }
inline F8 Uge(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_NLT_UQ)}; }
inline F8 Ueq(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_EQ_UQ)}; }
inline F8 IsNan(F8 a) { return {_mm256_cmp_ps(a.v, a.v, _CMP_UNORD_Q)}; }

// Bitwise operations, which no FTZ or DAZ touches.
inline F8 And(F8 a, F8 b) { return {_mm256_and_ps(a.v, b.v)}; }
inline F8 AndNot(F8 a, F8 b) { return {_mm256_andnot_ps(a.v, b.v)}; }  // ~a & b
inline F8 Or(F8 a, F8 b) { return {_mm256_or_ps(a.v, b.v)}; }
inline F8 Xor(F8 a, F8 b) { return {_mm256_xor_ps(a.v, b.v)}; }
// mask ? t : f, per lane.
inline F8 Select(F8 mask, F8 t, F8 f) { return {_mm256_blendv_ps(f.v, t.v, mask.v)}; }

inline I8 AsInt(F8 a) { return {_mm256_castps_si256(a.v)}; }
inline F8 AsFloat(I8 a) { return {_mm256_castsi256_ps(a.v)}; }
// vcvttps2dq and vcvtdq2ps.
inline I8 CvttPs2Dq(F8 a) { return {_mm256_cvttps_epi32(a.v)}; }
inline F8 ToFloat(I8 a) { return {_mm256_cvtepi32_ps(a.v)}; }
inline I8 AddI(I8 a, I8 b) { return {_mm256_add_epi32(a.v, b.v)}; }
inline I8 SubI(I8 a, I8 b) { return {_mm256_sub_epi32(a.v, b.v)}; }
inline I8 MulLoI(I8 a, I8 b) { return {_mm256_mullo_epi32(a.v, b.v)}; }
inline I8 MinI(I8 a, I8 b) { return {_mm256_min_epi32(a.v, b.v)}; }
inline I8 MaxI(I8 a, I8 b) { return {_mm256_max_epi32(a.v, b.v)}; }
inline I8 AndI(I8 a, I8 b) { return {_mm256_and_si256(a.v, b.v)}; }
inline I8 OrI(I8 a, I8 b) { return {_mm256_or_si256(a.v, b.v)}; }
inline I8 ShlI(I8 a, int n) { return {_mm256_slli_epi32(a.v, n)}; }
inline I8 SrlI(I8 a, int n) { return {_mm256_srli_epi32(a.v, n)}; }
inline I8 SraI(I8 a, int n) { return {_mm256_srai_epi32(a.v, n)}; }
inline I8 GtI(I8 a, I8 b) { return {_mm256_cmpgt_epi32(a.v, b.v)}; }
inline I8 EqI(I8 a, I8 b) { return {_mm256_cmpeq_epi32(a.v, b.v)}; }
// Lane i of the result is lane idx[i] of a.
inline F8 Permute(F8 a, I8 idx) { return {_mm256_permutevar8x32_ps(a.v, idx.v)}; }
// Gathers as loads inserted into the vector, which Zen 3 runs faster than vpgatherdd or a store and
// reload.
inline std::int32_t LoadAt(const void *base, std::int32_t offset) {
    std::int32_t x;
    std::memcpy(&x, static_cast<const std::uint8_t *>(base) + offset, 4);
    return x;
}
inline int Load16At(const void *base, std::int32_t offset) {
    std::uint16_t x;
    std::memcpy(&x, static_cast<const std::uint8_t *>(base) + offset, 2);
    return x;
}
inline __m128i Load64At(const void *base, std::int32_t offset) {
    return _mm_loadl_epi64(
        reinterpret_cast<const __m128i *>(static_cast<const std::uint8_t *>(base) + offset));
}
// The eight offsets of a vector, lane i at o[i].
struct Offsets8 {
    std::int32_t o[8];
    explicit Offsets8(I8 v) {
        const __m128i lo = _mm256_castsi256_si128(v.v);
        const __m128i hi = _mm256_extracti128_si256(v.v, 1);
        o[0] = _mm_extract_epi32(lo, 0);
        o[1] = _mm_extract_epi32(lo, 1);
        o[2] = _mm_extract_epi32(lo, 2);
        o[3] = _mm_extract_epi32(lo, 3);
        o[4] = _mm_extract_epi32(hi, 0);
        o[5] = _mm_extract_epi32(hi, 1);
        o[6] = _mm_extract_epi32(hi, 2);
        o[7] = _mm_extract_epi32(hi, 3);
    }
};
// The 32 bits at lo + offsets[i] bytes for lanes 0 to 3 and at hi + offsets[i] for lanes 4 to 7.
inline I8 Gather32x2(const void *lo, const void *hi, I8 offsets) {
    const __m128i olo = _mm256_castsi256_si128(offsets.v);
    const __m128i ohi = _mm256_extracti128_si256(offsets.v, 1);
    const __m128i a = _mm_setr_epi32(
        LoadAt(lo, _mm_extract_epi32(olo, 0)), LoadAt(lo, _mm_extract_epi32(olo, 1)),
        LoadAt(lo, _mm_extract_epi32(olo, 2)), LoadAt(lo, _mm_extract_epi32(olo, 3)));
    const __m128i b = _mm_setr_epi32(
        LoadAt(hi, _mm_extract_epi32(ohi, 0)), LoadAt(hi, _mm_extract_epi32(ohi, 1)),
        LoadAt(hi, _mm_extract_epi32(ohi, 2)), LoadAt(hi, _mm_extract_epi32(ohi, 3)));
    return {_mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1)};
}
// The 32 bits at base + offsets[i] bytes in each lane.
inline I8 Gather32(const void *base, I8 offsets) { return Gather32x2(base, base, offsets); }
// The 16 bits at base + offsets[i] bytes in each lane, zero extended.
inline I8 Gather16(const void *base, I8 offsets) {
    const Offsets8 o(offsets);
    __m128i v = _mm_cvtsi32_si128(Load16At(base, o.o[0]));
    v = _mm_insert_epi16(v, Load16At(base, o.o[1]), 1);
    v = _mm_insert_epi16(v, Load16At(base, o.o[2]), 2);
    v = _mm_insert_epi16(v, Load16At(base, o.o[3]), 3);
    v = _mm_insert_epi16(v, Load16At(base, o.o[4]), 4);
    v = _mm_insert_epi16(v, Load16At(base, o.o[5]), 5);
    v = _mm_insert_epi16(v, Load16At(base, o.o[6]), 6);
    v = _mm_insert_epi16(v, Load16At(base, o.o[7]), 7);
    return {_mm256_cvtepu16_epi32(v)};
}
#if CUJU_RENDER_LANES_F16C
// The four halves of the 64-bit texels at base + offsets[i] bytes as floats, one vector per
// channel.
inline void GatherHalves4(const void *base, I8 offsets, F8 *rgba) {
    const Offsets8 o(offsets);
    __m128i p[4];
    for (int i = 0; i < 4; ++i) {
        p[i] = _mm_unpacklo_epi64(Load64At(base, o.o[2 * i]), Load64At(base, o.o[2 * i + 1]));
    }
    // p[i] holds lanes 2i and 2i + 1 as (r, g, b, a) halves.
    const __m128i t0 = _mm_unpacklo_epi16(p[0], p[1]);
    const __m128i t1 = _mm_unpackhi_epi16(p[0], p[1]);
    const __m128i t2 = _mm_unpacklo_epi16(p[2], p[3]);
    const __m128i t3 = _mm_unpackhi_epi16(p[2], p[3]);
    const __m128i u0 = _mm_unpacklo_epi16(t0, t1);
    const __m128i u1 = _mm_unpackhi_epi16(t0, t1);
    const __m128i v0 = _mm_unpacklo_epi16(t2, t3);
    const __m128i v1 = _mm_unpackhi_epi16(t2, t3);
    rgba[0] = {_mm256_cvtph_ps(_mm_unpacklo_epi64(u0, v0))};
    rgba[1] = {_mm256_cvtph_ps(_mm_unpackhi_epi64(u0, v0))};
    rgba[2] = {_mm256_cvtph_ps(_mm_unpacklo_epi64(u1, v1))};
    rgba[3] = {_mm256_cvtph_ps(_mm_unpackhi_epi64(u1, v1))};
}
#endif
// Bit i set when lane i of the mask is set.
inline int MoveMask(F8 a) { return _mm256_movemask_ps(a.v); }
// a in lanes 0 to 3, b in lanes 4 to 7.
inline F8 Pair(float a, float b) {
    return {_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(a)), _mm_set1_ps(b), 1)};
}
inline I8 PairI(std::int32_t a, std::int32_t b) {
    return {
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi32(a)), _mm_set1_epi32(b), 1)};
}

// Eight halves to floats, exactly, a signaling NaN coming back quiet as vcvtph2ps does.
inline F8 HalvesToFloat(const std::uint16_t *h) {
#if CUJU_RENDER_LANES_F16C
    return {_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)))};
#else
    float f[8];
    for (int i = 0; i < 8; ++i) {
        f[i] = HalfToFloat(h[i]);
    }
    return Load(f);
#endif
}
// The halves in the low 16 bits of each lane to floats, as HalvesToFloat.
inline F8 HalvesToFloatI(I8 h) {
#if CUJU_RENDER_LANES_F16C
    const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi32(h.v, h.v), 0x08);
    return {_mm256_cvtph_ps(_mm256_castsi256_si128(packed))};
#else
    std::int32_t v[8];
    std::uint16_t u[8];
    StoreI(v, h);
    for (int i = 0; i < 8; ++i) {
        u[i] = static_cast<std::uint16_t>(v[i]);
    }
    return HalvesToFloat(u);
#endif
}

#else

struct F8 {
    float v[8];
};
struct I8 {
    std::int32_t v[8];
};

#define CUJU_LANES_MAP(expr)      \
    F8 r;                         \
    for (int i = 0; i < 8; ++i) { \
        r.v[i] = (expr);          \
    }                             \
    return r
#define CUJU_LANES_MAPI(expr)     \
    I8 r;                         \
    for (int i = 0; i < 8; ++i) { \
        r.v[i] = (expr);          \
    }                             \
    return r

inline float MaskBits(bool m) { return FromBits(m ? 0xFFFFFFFFu : 0u); }

inline F8 Load(const float *p) {
    F8 r;
    std::memcpy(r.v, p, sizeof r.v);
    return r;
}
inline void Store(float *p, F8 a) { std::memcpy(p, a.v, sizeof a.v); }
inline F8 Splat(float x) { CUJU_LANES_MAP(x); }
inline I8 LoadI(const std::int32_t *p) {
    I8 r;
    std::memcpy(r.v, p, sizeof r.v);
    return r;
}
inline void StoreI(std::int32_t *p, I8 a) { std::memcpy(p, a.v, sizeof a.v); }
inline I8 SplatI(std::int32_t x) { CUJU_LANES_MAPI(x); }

inline F8 Add(F8 a, F8 b) { CUJU_LANES_MAP(ftz::Add(a.v[i], b.v[i])); }
inline F8 Sub(F8 a, F8 b) { CUJU_LANES_MAP(ftz::Sub(a.v[i], b.v[i])); }
inline F8 Mul(F8 a, F8 b) { CUJU_LANES_MAP(ftz::Mul(a.v[i], b.v[i])); }
inline F8 Div(F8 a, F8 b) { CUJU_LANES_MAP(ftz::Div(a.v[i], b.v[i])); }
inline F8 Sqrt(F8 a) { CUJU_LANES_MAP(ftz::Sqrt(a.v[i])); }
inline F8 Fma(F8 a, F8 b, F8 c) { CUJU_LANES_MAP(ftz::Fma(a.v[i], b.v[i], c.v[i])); }
inline F8 MinPs(F8 a, F8 b) { CUJU_LANES_MAP(ftz::MinPs(a.v[i], b.v[i])); }
inline F8 MaxPs(F8 a, F8 b) { CUJU_LANES_MAP(ftz::MaxPs(a.v[i], b.v[i])); }
inline F8 Floor(F8 a) { CUJU_LANES_MAP(ftz::Floor(a.v[i])); }
inline F8 Ceil(F8 a) { CUJU_LANES_MAP(ftz::Ceil(a.v[i])); }

inline F8 Olt(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Olt(a.v[i], b.v[i]))); }
inline F8 Ole(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Ole(a.v[i], b.v[i]))); }
inline F8 Ogt(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Ogt(a.v[i], b.v[i]))); }
inline F8 Oge(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Oge(a.v[i], b.v[i]))); }
inline F8 Oeq(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Oeq(a.v[i], b.v[i]))); }
inline F8 Ult(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Ult(a.v[i], b.v[i]))); }
inline F8 Ugt(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Ugt(a.v[i], b.v[i]))); }
inline F8 Uge(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Uge(a.v[i], b.v[i]))); }
inline F8 Ueq(F8 a, F8 b) { CUJU_LANES_MAP(MaskBits(ftz::Ueq(a.v[i], b.v[i]))); }
inline F8 IsNan(F8 a) { CUJU_LANES_MAP(MaskBits(a.v[i] != a.v[i])); }

inline F8 And(F8 a, F8 b) { CUJU_LANES_MAP(FromBits(Bits(a.v[i]) & Bits(b.v[i]))); }
inline F8 AndNot(F8 a, F8 b) { CUJU_LANES_MAP(FromBits(~Bits(a.v[i]) & Bits(b.v[i]))); }
inline F8 Or(F8 a, F8 b) { CUJU_LANES_MAP(FromBits(Bits(a.v[i]) | Bits(b.v[i]))); }
inline F8 Xor(F8 a, F8 b) { CUJU_LANES_MAP(FromBits(Bits(a.v[i]) ^ Bits(b.v[i]))); }
inline F8 Select(F8 mask, F8 t, F8 f) {
    CUJU_LANES_MAP((Bits(mask.v[i]) & 0x80000000u) ? t.v[i] : f.v[i]);
}

inline I8 AsInt(F8 a) { CUJU_LANES_MAPI(static_cast<std::int32_t>(Bits(a.v[i]))); }
inline F8 AsFloat(I8 a) { CUJU_LANES_MAP(FromBits(static_cast<std::uint32_t>(a.v[i]))); }
inline I8 CvttPs2Dq(F8 a) { CUJU_LANES_MAPI(cuju::render::CvttPs2Dq(a.v[i])); }
inline F8 ToFloat(I8 a) { CUJU_LANES_MAP(static_cast<float>(a.v[i])); }
inline std::int32_t WrapI(std::uint32_t x) { return static_cast<std::int32_t>(x); }
inline I8 AddI(I8 a, I8 b) {
    CUJU_LANES_MAPI(WrapI(static_cast<std::uint32_t>(a.v[i]) + static_cast<std::uint32_t>(b.v[i])));
}
inline I8 SubI(I8 a, I8 b) {
    CUJU_LANES_MAPI(WrapI(static_cast<std::uint32_t>(a.v[i]) - static_cast<std::uint32_t>(b.v[i])));
}
inline I8 MulLoI(I8 a, I8 b) {
    CUJU_LANES_MAPI(WrapI(static_cast<std::uint32_t>(a.v[i]) * static_cast<std::uint32_t>(b.v[i])));
}
inline I8 MinI(I8 a, I8 b) { CUJU_LANES_MAPI(a.v[i] < b.v[i] ? a.v[i] : b.v[i]); }
inline I8 MaxI(I8 a, I8 b) { CUJU_LANES_MAPI(a.v[i] > b.v[i] ? a.v[i] : b.v[i]); }
inline I8 AndI(I8 a, I8 b) { CUJU_LANES_MAPI(a.v[i] & b.v[i]); }
inline I8 OrI(I8 a, I8 b) { CUJU_LANES_MAPI(a.v[i] | b.v[i]); }
inline I8 ShlI(I8 a, int n) { CUJU_LANES_MAPI(WrapI(static_cast<std::uint32_t>(a.v[i]) << n)); }
inline I8 SrlI(I8 a, int n) { CUJU_LANES_MAPI(WrapI(static_cast<std::uint32_t>(a.v[i]) >> n)); }
inline I8 SraI(I8 a, int n) {
    CUJU_LANES_MAPI(a.v[i] < 0 ? WrapI(~(~static_cast<std::uint32_t>(a.v[i]) >> n))
                               : WrapI(static_cast<std::uint32_t>(a.v[i]) >> n));
}
inline I8 GtI(I8 a, I8 b) { CUJU_LANES_MAPI(a.v[i] > b.v[i] ? -1 : 0); }
inline I8 EqI(I8 a, I8 b) { CUJU_LANES_MAPI(a.v[i] == b.v[i] ? -1 : 0); }
inline F8 Permute(F8 a, I8 idx) { CUJU_LANES_MAP(a.v[idx.v[i] & 7]); }
inline std::int32_t Load32At(const void *base, std::int32_t offset) {
    std::int32_t x;
    std::memcpy(&x, static_cast<const std::uint8_t *>(base) + offset, 4);
    return x;
}
inline I8 Gather32(const void *base, I8 offsets) { CUJU_LANES_MAPI(Load32At(base, offsets.v[i])); }
inline I8 Gather32x2(const void *lo, const void *hi, I8 offsets) {
    CUJU_LANES_MAPI(Load32At(i < 4 ? lo : hi, offsets.v[i]));
}
inline std::int32_t Load16At(const void *base, std::int32_t offset) {
    std::uint16_t x;
    std::memcpy(&x, static_cast<const std::uint8_t *>(base) + offset, 2);
    return x;
}
inline I8 Gather16(const void *base, I8 offsets) { CUJU_LANES_MAPI(Load16At(base, offsets.v[i])); }
inline F8 Pair(float a, float b) { CUJU_LANES_MAP(i < 4 ? a : b); }
inline I8 PairI(std::int32_t a, std::int32_t b) { CUJU_LANES_MAPI(i < 4 ? a : b); }
inline int MoveMask(F8 a) {
    int m = 0;
    for (int i = 0; i < 8; ++i) {
        m |= static_cast<int>(Bits(a.v[i]) >> 31) << i;
    }
    return m;
}

inline F8 HalvesToFloat(const std::uint16_t *h) { CUJU_LANES_MAP(HalfToFloat(h[i])); }
inline F8 HalvesToFloatI(I8 h) { CUJU_LANES_MAP(HalfToFloat(static_cast<std::uint16_t>(h.v[i]))); }

#undef CUJU_LANES_MAP
#undef CUJU_LANES_MAPI

#endif

// Lane operations shared by both builds.
inline F8 Neg(F8 a) { return Xor(a, Splat(FromBits(0x80000000u))); }
inline F8 Abs(F8 a) { return And(a, Splat(FromBits(0x7FFFFFFFu))); }
// 1.0 in the lanes of the mask, +0 elsewhere, as the JIT code's and with 0x3F800000.
inline F8 OneIf(F8 mask) { return And(mask, Splat(1.0f)); }
inline I8 SelectI(I8 mask, I8 t, I8 f) {
    return AsInt(Select(AsFloat(mask), AsFloat(t), AsFloat(f)));
}
inline I8 XorI(I8 a, I8 b) { return AsInt(Xor(AsFloat(a), AsFloat(b))); }

// The cosine of polynomials.h, lp_build_sin_or_cos, per lane.
inline F8 Cos(F8 a) {
    const I8 a_bits = AsInt(a);
    const F8 x_abs = Abs(a);
    const F8 scale_y = Mul(x_abs, Splat(FromBits(0x3FA2F983u)));
    const I8 emm2_add = AddI(CvttPs2Dq(scale_y), SplatI(1));
    const I8 emm2_and = AndI(emm2_add, SplatI(~1));
    const F8 y_2 = ToFloat(emm2_and);
    const I8 emm2_2 = SubI(emm2_and, SplatI(2));
    const I8 sign_bit = ShlI(AndI(XorI(emm2_2, SplatI(-1)), SplatI(4)), 29);
    const F8 poly_sin = AsFloat(EqI(AndI(emm2_2, SplatI(2)), SplatI(0)));
    const F8 x_1 = Fma(y_2, Splat(FromBits(0xBF490000u)), x_abs);
    const F8 x_2 = Fma(y_2, Splat(FromBits(0xB97DA000u)), x_1);
    const F8 x_3 = Fma(y_2, Splat(FromBits(0xB3222169u)), x_2);
    const F8 z = Mul(x_3, x_3);
    const F8 y_4 = Fma(z, Splat(FromBits(0x37CCF5CEu)), Splat(FromBits(0xBAB6061Au)));
    const F8 y_6 = Fma(y_4, z, Splat(FromBits(0x3D2AAAA5u)));
    const F8 y_7 = Mul(y_6, z);
    const F8 y_8 = Mul(y_7, z);
    const F8 y_9 = Sub(y_8, Mul(z, Splat(0.5f)));
    const F8 y_10 = Add(y_9, Splat(1.0f));
    const F8 y2_4 = Fma(z, Splat(FromBits(0xB94CA1F9u)), Splat(FromBits(0x3C08839Eu)));
    const F8 y2_6 = Fma(y2_4, z, Splat(FromBits(0xBE2AAAA3u)));
    const F8 y2_7 = Mul(y2_6, z);
    const F8 y2_9 = Fma(y2_7, x_3, x_3);
    const F8 y = Xor(Select(poly_sin, y2_9, y_10), AsFloat(sign_bit));
    const F8 clamped = MaxPs(MinPs(y, Splat(1.0f)), Splat(-1.0f));
    const F8 finite =
        AsFloat(XorI(EqI(AndI(a_bits, SplatI(0x7F800000)), SplatI(0x7F800000)), SplatI(-1)));
    return Select(finite, clamped, Splat(FromBits(0x7FC00000u)));
}

// unorm16 of a channel already clamped to [0, 1], gl::Unorm16Ftz per lane, in the low 16 bits.
inline I8 Unorm16(F8 c) {
    return AndI(AsInt(Add(Mul(c, Splat(FromBits(0x3F7FFF00u))), Splat(128.0f))), SplatI(0xFFFF));
}

// Texels of 16 bits at (x, y) of a level of width w >= 2 with rows of `stride` bytes: one 32-bit
// gather of the pair at min(x, w - 2), which never reads past the row, and the half of it that
// holds x. `next` is the texel at x + 1 clamped to the row, the pair's upper half.
inline void Gather16Pair(const std::uint8_t *data, int stride, int w, I8 x, I8 y, I8 &at_x,
                         I8 &next) {
    const I8 xm = MinI(x, SplatI(w - 2));
    const I8 pair = Gather32(data, AddI(MulLoI(y, SplatI(stride)), ShlI(xm, 1)));
    const I8 lo = AndI(pair, SplatI(0xFFFF));
    next = SrlI(pair, 16);
    at_x = SelectI(EqI(x, xm), lo, next);
}

// Level 0 of a texture for kNearestClamp fetches of eight lanes.
struct NearestLevel {
    const Texture *texture;
    const std::uint8_t *data;
    int w;
    int h;
    float fw;
    float fh;
    Format format;
};

inline NearestLevel NearestLevelOf(const Texture &t) {
    const TextureLevel &l = t.levels[0];
    return NearestLevel{&t,
                        l.texels.data(),
                        l.width,
                        l.height,
                        static_cast<float>(l.width),
                        static_cast<float>(l.height),
                        t.format};
}

// The texel index of NEAREST with CLAMP_TO_EDGE: s * size truncated, then clamped to the edge.
inline I8 NearestIndex(F8 s, float fsize, int size) {
    return MaxI(MinI(CvttPs2Dq(Mul(s, Splat(fsize))), SplatI(size - 1)), SplatI(0));
}

// kNearestClamp for eight lanes, the four channels as SampleQuads returns them: Z16 as (d, d, d, 1)
// with d = u16 * fl(1 / 65535), RGBA16F halves widened exactly, other formats through SampleQuads.
inline void FetchNearest(const NearestLevel &l, F8 s, F8 t, F8 *rgba) {
    if (l.format != Format::Z16_UNORM && l.format != Format::R16G16B16A16_FLOAT) {
        float sv[8], tv[8], out[32];
        Store(sv, s);
        Store(tv, t);
        SampleQuads(SamplerKind::kNearestClamp, *l.texture, 2, sv, tv, out);
        for (int c = 0; c < 4; ++c) {
            rgba[c] = Load(out + 8 * c);
        }
        return;
    }
    const I8 x = NearestIndex(s, l.fw, l.w);
    const I8 y = NearestIndex(t, l.fh, l.h);
    const I8 texel = AddI(MulLoI(y, SplatI(l.w)), x);
    if (l.format == Format::Z16_UNORM) {
        rgba[0] = Mul(ToFloat(Gather16(l.data, ShlI(texel, 1))), Splat(FromBits(0x37800080u)));
        rgba[1] = rgba[0];
        rgba[2] = rgba[0];
        rgba[3] = Splat(1.0f);
        return;
    }
    const I8 offset = ShlI(texel, 3);
#if CUJU_RENDER_LANES_F16C
    GatherHalves4(l.data, offset, rgba);
#else
    const I8 rg = Gather32(l.data, offset);
    const I8 ba = Gather32(l.data, AddI(offset, SplatI(4)));
    rgba[0] = HalvesToFloatI(AndI(rg, SplatI(0xFFFF)));
    rgba[1] = HalvesToFloatI(SrlI(rg, 16));
    rgba[2] = HalvesToFloatI(AndI(ba, SplatI(0xFFFF)));
    rgba[3] = HalvesToFloatI(SrlI(ba, 16));
#endif
}

}  // namespace cuju::render::lanes

#endif  // CUJU_RENDER_LANES_H
