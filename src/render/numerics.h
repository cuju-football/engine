// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Scalar operations as llvmpipe's JIT code computes them on x86-64 with AVX2, FMA and F16C.
// The algorithms follow Mesa's gallivm (MIT licence, Copyright VMware, Inc. and others).

#ifndef CUJU_RENDER_NUMERICS_H
#define CUJU_RENDER_NUMERICS_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace cuju::render {
inline std::uint32_t Bits(float x) {
    std::uint32_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

inline float FromBits(std::uint32_t b) {
    float x;
    std::memcpy(&x, &b, sizeof x);
    return x;
}

inline float Fma(float a, float b, float c) { return std::fma(a, b, c); }

// minps and maxps return the second operand on NaN and on equal operands, including signed zeros.
inline float MinPs(float a, float b) { return a < b ? a : b; }
inline float MaxPs(float a, float b) { return a > b ? a : b; }

inline float Floor(float x) { return std::floor(x); }
inline float Ceil(float x) { return std::ceil(x); }

inline std::int32_t CvttPs2Dq(float x) {
    if (!(x > -2147483904.0f && x < 2147483648.0f)) {
        return INT32_MIN;
    }
    return static_cast<std::int32_t>(x);
}

inline std::int32_t CvtPs2Dq(float x) {
    if (!(x >= -2147483648.0f && x < 2147483648.0f)) {
        return INT32_MIN;
    }
    return static_cast<std::int32_t>(std::nearbyint(x));
}

inline std::uint16_t HalfRz(float x) {
    const std::uint32_t b = Bits(x);
    const std::uint16_t sign = static_cast<std::uint16_t>((b >> 16) & 0x8000u);
    const std::uint32_t a = b & 0x7FFFFFFFu;
    if (a > 0x7F800000u) {
        return static_cast<std::uint16_t>(sign | 0x7E00u | ((a & 0x7FFFFFu) >> 13));
    }
    if (a == 0x7F800000u) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (a >= 0x477FF000u) {
        return static_cast<std::uint16_t>(sign | 0x7BFFu);
    }
    if (a >= 0x38800000u) {
        return static_cast<std::uint16_t>(sign | ((a - 0x38000000u) >> 13));
    }
    const std::uint32_t e = a >> 23;
    if (e == 0) {
        return sign;
    }
    const std::uint32_t shift = 126u - e;
    const std::uint32_t m = (a & 0x7FFFFFu) | 0x800000u;
    return static_cast<std::uint16_t>(sign | (shift > 24u ? 0u : (m >> shift)));
}

inline float HalfToFloat(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t e = (h >> 10) & 0x1Fu;
    const std::uint32_t m = h & 0x3FFu;
    if (e == 0x1Fu) {
        return FromBits(sign | 0x7F800000u | (m << 13) | (m != 0 ? 0x400000u : 0u));
    }
    if (e != 0) {
        return FromBits(sign | ((e + 112u) << 23) | (m << 13));
    }
    if (m == 0) {
        return FromBits(sign);
    }
    const float v = static_cast<float>(m) * 5.9604644775390625e-8f;
    return sign ? -v : v;
}

inline std::uint16_t Unorm16(float x) {
    const float t = x * FromBits(0x3F7FFF00u);
    return static_cast<std::uint16_t>(Bits(t + 128.0f) & 0xFFFFu);
}

inline std::uint16_t DepthD16(float z) { return Unorm16(MaxPs(MinPs(z, 1.0f), 0.0f)); }

inline float Unorm8ToFloat(std::uint8_t v) { return static_cast<float>(v) * FromBits(0x3B808081u); }
inline float D16ToFloat(std::uint16_t v) { return static_cast<float>(v) * FromBits(0x37800080u); }

}

#endif
