// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The floating-point environment of llvmpipe's stages. Fragment shading, sampling, the vertex
// shader, clipping and most of setup run with x86 FTZ and DAZ on, the RGBA16F store and blend
// window and the setup of clipped chunks run under IEEE 754. Rounding is to nearest even in both.
//
// Two implementations give the same bits. On x86-64 with SSE4.1 and FMA the operations of namespace
// `ftz` are the plain instructions, and `FpScope` sets MXCSR. Elsewhere, or with
// CUJU_RENDER_SOFT_FTZ defined, they are the software model of namespace `fpmodel`, which equals
// the hardware of an AMD Zen 3 core on every input: DAZ on every input, and a flush to signed zero
// of every result that is tiny after rounding.
//
// An `FpScope` governs the arithmetic of the functions called inside it. Open it outside the
// function whose arithmetic it governs, since a compiler may move arithmetic across the scope's
// own statements in the same function. Each thread needs its own scope.

#ifndef CUJU_RENDER_FPENV_H
#define CUJU_RENDER_FPENV_H

#include <cmath>
#include <cstdint>

#include "numerics.h"

#if !defined(CUJU_RENDER_SOFT_FTZ) && (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__SSE4_1__) || defined(__AVX__)) &&                                  \
    (defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__)))
#define CUJU_RENDER_HARDWARE_FTZ 1
#include <immintrin.h>
#else
#define CUJU_RENDER_HARDWARE_FTZ 0
#endif

namespace cuju::render {
enum class FpMode : std::uint8_t {
    kFtzDaz,
    kIeee,
};

class FpScope {
public:
    explicit FpScope(FpMode mode);
    ~FpScope();
    FpScope(const FpScope &) = delete;
    FpScope &operator=(const FpScope &) = delete;

private:
    std::uint64_t saved_;
};

constexpr bool kHardwareFtz = CUJU_RENDER_HARDWARE_FTZ != 0;

namespace fpmodel {
inline bool IsSubnormal(float x) {
    const std::uint32_t b = Bits(x);
    return (b & 0x7F800000u) == 0 && (b & 0x7FFFFFu) != 0;
}

inline float Daz(float x) {
    const std::uint32_t b = Bits(x);
    return (b & 0x7F800000u) == 0 ? FromBits(b & 0x80000000u) : x;
}

inline float FlushSubnormal(float r) {
    const std::uint32_t b = Bits(r);
    return (b & 0x7F800000u) == 0 ? FromBits(b & 0x80000000u) : r;
}

inline float CreatedNan() {
    volatile std::uint32_t bits = 0xFFC00000u;
    return FromBits(bits);
}
inline float X86Nan(float r, float a, float b) {
    return (r != r && a == a && b == b) ? CreatedNan() : r;
}
inline float X86Nan(float r, float a, float b, float c) {
    return (r != r && a == a && b == b && c == c) ? CreatedNan() : r;
}

inline float Add(float a, float b) {
    a = Daz(a);
    b = Daz(b);
    return X86Nan(FlushSubnormal(a + b), a, b);
}
inline float Sub(float a, float b) {
    a = Daz(a);
    b = Daz(b);
    return X86Nan(FlushSubnormal(a - b), a, b);
}

inline float Mul(float a, float b) {
    a = Daz(a);
    b = Daz(b);
    const float r = a * b;
    const std::uint32_t m = Bits(r) & 0x7FFFFFFFu;
    if (m == 0x00800000u) {
        const double p = static_cast<double>(a) * static_cast<double>(b);
        if (std::fabs(p) < 0x1p-126 - 0x1p-151) {
            return FromBits(Bits(r) & 0x80000000u);
        }
        return r;
    }
    return X86Nan(FlushSubnormal(r), a, b);
}

inline float Div(float a, float b) {
    a = Daz(a);
    b = Daz(b);
    return X86Nan(FlushSubnormal(a / b), a, b);
}

inline float Sqrt(float a) {
    a = Daz(a);
    const float r = std::sqrt(a);
    return (r != r && a == a) ? CreatedNan() : r;
}

float Fma(float a, float b, float c, bool ftz);

inline float MinPs(float a, float b) { return cuju::render::MinPs(Daz(a), Daz(b)); }
inline float MaxPs(float a, float b) { return cuju::render::MaxPs(Daz(a), Daz(b)); }

// roundps under DAZ, so the floor of a negative subnormal is -0 and the ceiling of a positive one
// is +0.
inline float Floor(float a) { return std::floor(Daz(a)); }
inline float Ceil(float a) { return std::ceil(Daz(a)); }

}  // namespace fpmodel

namespace ftz {
#if CUJU_RENDER_HARDWARE_FTZ

inline float Add(float a, float b) { return a + b; }
inline float Sub(float a, float b) { return a - b; }
inline float Mul(float a, float b) { return a * b; }
inline float Div(float a, float b) { return a / b; }
inline float Sqrt(float a) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(a))); }
inline float Fma(float a, float b, float c) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
inline float MinPs(float a, float b) {
    return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b)));
}
inline float MaxPs(float a, float b) {
    return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b)));
}
inline float Floor(float a) {
    const __m128 x = _mm_set_ss(a);
    return _mm_cvtss_f32(_mm_round_ss(x, x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}
inline float Ceil(float a) {
    const __m128 x = _mm_set_ss(a);
    return _mm_cvtss_f32(_mm_round_ss(x, x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}
inline float Daz(float a) { return a; }

#else

inline float Add(float a, float b) { return fpmodel::Add(a, b); }
inline float Sub(float a, float b) { return fpmodel::Sub(a, b); }
inline float Mul(float a, float b) { return fpmodel::Mul(a, b); }
inline float Div(float a, float b) { return fpmodel::Div(a, b); }
inline float Sqrt(float a) { return fpmodel::Sqrt(a); }
inline float Fma(float a, float b, float c) { return fpmodel::Fma(a, b, c, true); }
inline float MinPs(float a, float b) { return fpmodel::MinPs(a, b); }
inline float MaxPs(float a, float b) { return fpmodel::MaxPs(a, b); }
inline float Floor(float a) { return fpmodel::Floor(a); }
inline float Ceil(float a) { return fpmodel::Ceil(a); }
inline float Daz(float a) { return fpmodel::Daz(a); }

#endif

inline bool Olt(float a, float b) { return Daz(a) < Daz(b); }
inline bool Ole(float a, float b) { return Daz(a) <= Daz(b); }
inline bool Ogt(float a, float b) { return Daz(a) > Daz(b); }
inline bool Oge(float a, float b) { return Daz(a) >= Daz(b); }
inline bool Oeq(float a, float b) { return Daz(a) == Daz(b); }
inline bool Ult(float a, float b) { return !(Daz(a) >= Daz(b)); }
inline bool Ule(float a, float b) { return !(Daz(a) > Daz(b)); }
inline bool Ugt(float a, float b) { return !(Daz(a) <= Daz(b)); }
inline bool Uge(float a, float b) { return !(Daz(a) < Daz(b)); }
inline bool Ueq(float a, float b) { return !(Daz(a) < Daz(b) || Daz(a) > Daz(b)); }

}  // namespace ftz

// The operations of IEEE stages, correct on every host inside an FpScope(FpMode::kIeee).
namespace ieee {
inline float Fma(float a, float b, float c) { return std::fma(a, b, c); }

}  // namespace ieee

}  // namespace cuju::render

#endif
