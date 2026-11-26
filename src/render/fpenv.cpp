// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include "fpenv.h"

#include <cfenv>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#define CUJU_RENDER_X86 1
#else
#define CUJU_RENDER_X86 0
#endif

#if defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>
#endif

namespace cuju::render {
namespace {
#if CUJU_RENDER_X86
constexpr std::uint32_t kFtz = 0x8000u;
constexpr std::uint32_t kDaz = 0x0040u;
constexpr std::uint32_t kRounding = 0x6000u;
#endif

#if !CUJU_RENDER_X86 && (defined(__aarch64__) || defined(_M_ARM64))
constexpr std::uint64_t kFpcrFz = 1ull << 24;
constexpr std::uint64_t kFpcrRounding = 3ull << 22;

std::uint64_t ReadFpcr() {
#if defined(_MSC_VER) && !defined(__clang__)
    return static_cast<std::uint64_t>(_ReadStatusReg(ARM64_FPCR));
#else
    std::uint64_t v;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(v));
    return v;
#endif
}

void WriteFpcr(std::uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
    _WriteStatusReg(ARM64_FPCR, static_cast<__int64>(v));
#else
    __asm__ __volatile__("msr fpcr, %0" : : "r"(v));
#endif
}
#endif

}  // namespace

FpScope::FpScope(FpMode mode) {
#if CUJU_RENDER_X86
    const std::uint32_t csr = _mm_getcsr();
    saved_ = csr;
    std::uint32_t next = csr & ~(kFtz | kDaz | kRounding);
#if CUJU_RENDER_HARDWARE_FTZ
    if (mode == FpMode::kFtzDaz) {
        next |= kFtz | kDaz;
    }
#else
    (void)mode;
#endif
    _mm_setcsr(next);
#elif defined(__aarch64__) || defined(_M_ARM64)
    (void)mode;
    const std::uint64_t fpcr = ReadFpcr();
    saved_ = fpcr;
    WriteFpcr(fpcr & ~(kFpcrFz | kFpcrRounding));
#else
    (void)mode;
    saved_ = static_cast<std::uint64_t>(std::fegetround());
    std::fesetround(FE_TONEAREST);
#endif
}

FpScope::~FpScope() {
#if CUJU_RENDER_X86
    _mm_setcsr(static_cast<std::uint32_t>(saved_));
#elif defined(__aarch64__) || defined(_M_ARM64)
    WriteFpcr(saved_);
#else
    std::fesetround(static_cast<int>(saved_));
#endif
}

namespace fpmodel {
namespace {
std::uint64_t DoubleBits(double x) {
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

double DoubleFromBits(std::uint64_t b) {
    double x;
    std::memcpy(&x, &b, sizeof x);
    return x;
}

float RoundToFloat(double x, bool ftz) {
    const std::uint64_t b = DoubleBits(x);
    const std::uint32_t sign = static_cast<std::uint32_t>(b >> 32) & 0x80000000u;
    const int exponent = static_cast<int>((b >> 52) & 0x7FF) - 1023;
    const std::uint64_t m = (b & ((1ull << 52) - 1)) | (1ull << 52);
    const std::uint64_t low = m & ((1ull << 29) - 1);
    std::uint64_t r24 = m >> 29;
    if (low > (1ull << 28) || (low == (1ull << 28) && (r24 & 1))) {
        ++r24;
    }
    int e24 = exponent;
    if (r24 == (1ull << 24)) {
        r24 >>= 1;
        ++e24;
    }
    if (e24 >= -126) {
        if (e24 > 127) {
            return FromBits(sign | 0x7F800000u);
        }
        return FromBits(sign | (static_cast<std::uint32_t>(e24 + 127) << 23) |
                        (static_cast<std::uint32_t>(r24) & 0x7FFFFFu));
    }
    if (ftz) {
        return FromBits(sign);
    }
    const int shift = -(exponent + 97);
    if (shift >= 64) {
        return FromBits(sign);
    }
    std::uint64_t q = m >> shift;
    const std::uint64_t rest = m & ((1ull << shift) - 1);
    const std::uint64_t half = 1ull << (shift - 1);
    if (rest > half || (rest == half && (q & 1))) {
        ++q;
    }
    return FromBits(sign | static_cast<std::uint32_t>(q));
}

}  // namespace

float Fma(float a, float b, float c, bool ftz) {
    if (ftz) {
        a = Daz(a);
        b = Daz(b);
        c = Daz(c);
    }
    const double p = static_cast<double>(a) * static_cast<double>(b);
    const double cd = c;
    if (!std::isfinite(p) || !std::isfinite(cd) || a != a || b != b) {
        const float r = static_cast<float>(p + cd);
        return X86Nan(r, a, b, c);
    }
    const double s = p + cd;
    if (s == 0) {
        return static_cast<float>(s);
    }
    const double bb = s - p;
    const double err = (p - (s - bb)) + (cd - bb);
    double odd = s;
    if (err != 0) {
        std::uint64_t sb = DoubleBits(s);
        if ((err > 0) != (s > 0)) {
            --sb;
        }
        sb |= 1;
        odd = DoubleFromBits(sb);
    }
    return RoundToFloat(odd, ftz);
}

}  // namespace fpmodel

}  // namespace cuju::render
