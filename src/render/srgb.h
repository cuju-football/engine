// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// sRGB conversions of llvmpipe under FTZ and DAZ: the per texel decode polynomial of the samplers
// and the encode of the mipmap blit, which rests on vrsqrtps. The algorithms and constants are
// Mesa's gallivm, lp_bld_format_srgb.c (MIT licence, Copyright VMware, Inc. and others).

#ifndef CUJU_RENDER_SRGB_H
#define CUJU_RENDER_SRGB_H

#include <cstdint>

#include "fpenv.h"
#include "numerics.h"

namespace cuju::render {

// lp_build_srgb_to_linear on the raw byte as a float in [0, 255], evaluated in Estrin order.
inline float SrgbDecode(float x) {
    const float lin = ftz::Mul(x, FromBits(0x39A32D56u));
    const float x2 = ftz::Mul(x, x);
    const float odd = ftz::Fma(x2, FromBits(0x329C092Bu), FromBits(0x37456107u));
    const float even = ftz::Fma(x2, FromBits(0x3732EE61u), FromBits(0x3B16BB99u));
    const float pw = ftz::Fma(odd, x, even);
    return ftz::Ule(x, 15.0f) ? lin : pw;
}

inline float SrgbDecodeByte(std::uint8_t v) { return SrgbDecode(static_cast<float>(v)); }

// vrsqrtps of AMD Zen 3 cores, which the reference ran on. The result depends on the parity of
// the exponent and the upper 12 bits of the mantissa only, and it carries 12 mantissa bits.
// Defined in mipmap.cpp.
float RsqrtZen3(float x);

// lp_build_linear_to_srgb for an 8-bit destination: three chained reciprocal square roots with no
// Newton step, then the rounding conversion. The input is the colour already clamped by the
// fragment shader output.
inline std::int32_t SrgbEncodeByte(float v) {
    const float x = ftz::MaxPs(ftz::MinPs(v, 1.0f), 0.0f);
    const float r1 = RsqrtZen3(x);
    const float s = ftz::Mul(x, r1);
    const float x15 = ftz::Mul(s, x);
    const float r2 = RsqrtZen3(x15);
    const float r3 = RsqrtZen3(r2);
    const float pw = ftz::Add(ftz::Mul(FromBits(0x4336D4C8u), r3),
                              ftz::Fma(FromBits(0x42B00F45u), s, FromBits(0xC17CF5C2u)));
    const float lin = ftz::Mul(x, FromBits(0x454DE99Au));
    return CvtPs2Dq(ftz::Ule(x, FromBits(0x3B4D2E1Cu)) ? lin : pw);
}

}  // namespace cuju::render

#endif  // CUJU_RENDER_SRGB_H
