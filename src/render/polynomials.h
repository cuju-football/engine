// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The transcendental functions of llvmpipe's fragment shaders, operation for operation as the JIT
// code computes them under FTZ and DAZ: log2, exp2 and pow from lp_build_log2, lp_build_exp2 and
// lp_build_pow, the Cephes sine and cosine of lp_build_sin_or_cos, and the piecewise linear log2
// of the level of detail. The algorithms and constants are Mesa's gallivm, lp_bld_arit.c (MIT
// licence, Copyright VMware, Inc. and others).

#ifndef CUJU_RENDER_POLYNOMIALS_H
#define CUJU_RENDER_POLYNOMIALS_H

#include <cstdint>

#include "fpenv.h"
#include "numerics.h"

namespace cuju::render {

// lp_build_fast_log2: exponent - 128 plus the mantissa read as a float in [1, 2).
inline float FastLog2(float x) {
    const std::uint32_t b = Bits(x);
    const float e = static_cast<float>(static_cast<std::int32_t>((b >> 23) & 0xFFu) - 128);
    const float m = FromBits((b & 0x7FFFFFu) | 0x3F800000u);
    return ftz::Add(e, m);
}

// lp_build_log2, with its special cases: NaN below zero, -inf at zero, +inf at +inf.
inline float Log2(float x) {
    const std::uint32_t b = Bits(x);
    const float e = static_cast<float>(static_cast<std::int32_t>((b & 0x7F800000u) >> 23) - 127);
    const float m = FromBits((b & 0x7FFFFFu) | 0x3F800000u);
    const float y = ftz::Div(ftz::Sub(m, 1.0f), ftz::Add(m, 1.0f));
    const float z = ftz::Mul(y, y);
    const float z2 = ftz::Mul(z, z);
    const float p_odd = ftz::Fma(z2, FromBits(0x3ED03D59u), FromBits(0x3F13D321u));
    const float p_even = ftz::Fma(z2, FromBits(0x3ECE8316u), FromBits(0x3F7637F9u));
    const float p_hi = ftz::Fma(z2, p_odd, FromBits(0x4038AA3Bu));
    const float p = ftz::Fma(p_even, z, p_hi);
    float r = ftz::Fma(y, p, e);
    if (ftz::Uge(x, FromBits(0x7F800000u))) {
        r = FromBits(0x7F800000u);
    }
    if (ftz::Ueq(x, 0.0f)) {
        r = FromBits(0xFF800000u);
    }
    if (ftz::Ult(x, 0.0f)) {
        r = FromBits(0x7FC00000u);
    }
    return r;
}

// lp_build_exp2: the input clamped to [-126.99999, 128], 2^floor times a polynomial of the
// fraction.
inline float Exp2(float x) {
    const float v = ftz::MaxPs(FromBits(0xC2FDFFFFu), ftz::MinPs(128.0f, x));
    const float i = ftz::Floor(v);
    const float f = ftz::Sub(v, i);
    const std::uint32_t ipart = static_cast<std::uint32_t>(CvttPs2Dq(i));
    const float scale = FromBits((ipart + 127u) << 23);
    const float f2 = ftz::Mul(f, f);
    const float q_hi = ftz::Fma(f2, FromBits(0x3AF61905u), FromBits(0x3D64AA23u));
    const float q_lo = ftz::Fma(f2, FromBits(0x3C134806u), FromBits(0x3E75EAD4u));
    const float q_odd = ftz::Fma(f2, q_hi, FromBits(0x3F31727Bu));
    const float q_even = ftz::Fma(f2, q_lo, 1.0f);
    const float q = ftz::Fma(q_odd, f, q_even);
    return ftz::Mul(scale, q);
}

// GLSL pow as the JIT code computes it: zero for a zero base, else exp2(log2(x) * y).
inline float Pow(float x, float y) {
    if (ftz::Oeq(x, 0.0f)) {
        return 0.0f;
    }
    return Exp2(ftz::Mul(Log2(x), y));
}

namespace detail {

// lp_build_sin_or_cos. The result is clamped to [-1, 1] and is NaN for an infinite or NaN input.
inline float SinOrCos(float a, bool cos) {
    const std::uint32_t a_bits = Bits(a);
    const float x_abs = FromBits(a_bits & 0x7FFFFFFFu);
    const float scale_y = ftz::Mul(x_abs, FromBits(0x3FA2F983u));
    const std::uint32_t emm2_add = static_cast<std::uint32_t>(CvttPs2Dq(scale_y)) + 1u;
    const std::uint32_t emm2_and = emm2_add & ~1u;
    const float y_2 = static_cast<float>(static_cast<std::int32_t>(emm2_and));
    const std::uint32_t emm2_2 = cos ? emm2_and - 2u : emm2_and;
    const std::uint32_t sign_bit =
        cos ? ((~emm2_2) & 4u) << 29 : (a_bits ^ (emm2_add << 29)) & 0x80000000u;
    const bool poly_sin = (emm2_2 & 2u) == 0;
    const float x_1 = ftz::Fma(y_2, FromBits(0xBF490000u), x_abs);
    const float x_2 = ftz::Fma(y_2, FromBits(0xB97DA000u), x_1);
    const float x_3 = ftz::Fma(y_2, FromBits(0xB3222169u), x_2);
    const float z = ftz::Mul(x_3, x_3);
    const float y_4 = ftz::Fma(z, FromBits(0x37CCF5CEu), FromBits(0xBAB6061Au));
    const float y_6 = ftz::Fma(y_4, z, FromBits(0x3D2AAAA5u));
    const float y_7 = ftz::Mul(y_6, z);
    const float y_8 = ftz::Mul(y_7, z);
    const float y_9 = ftz::Sub(y_8, ftz::Mul(z, 0.5f));
    const float y_10 = ftz::Add(y_9, 1.0f);
    const float y2_4 = ftz::Fma(z, FromBits(0xB94CA1F9u), FromBits(0x3C08839Eu));
    const float y2_6 = ftz::Fma(y2_4, z, FromBits(0xBE2AAAA3u));
    const float y2_7 = ftz::Mul(y2_6, z);
    const float y2_9 = ftz::Fma(y2_7, x_3, x_3);
    const float y = FromBits((poly_sin ? Bits(y2_9) : Bits(y_10)) ^ sign_bit);
    const float clamped = ftz::MaxPs(ftz::MinPs(y, 1.0f), -1.0f);
    return (a_bits & 0x7F800000u) != 0x7F800000u ? clamped : FromBits(0x7FC00000u);
}

}  // namespace detail

inline float Cos(float a) { return detail::SinOrCos(a, true); }
inline float Sin(float a) { return detail::SinOrCos(a, false); }

}  // namespace cuju::render

#endif  // CUJU_RENDER_POLYNOMIALS_H
