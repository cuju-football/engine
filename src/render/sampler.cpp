// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports the SoA sampling of Mesa's gallivm, lp_bld_sample_soa.c and lp_bld_sample.c (MIT licence,
// Copyright VMware, Inc. and others), as its JIT code runs for GRF's sampler states.

#include "sampler.h"

#include <cstddef>
#include <cstdint>

#include "fpenv.h"
#include "lanes.h"
#include "numerics.h"
#include "polynomials.h"
#include "srgb.h"

namespace cuju::render {

namespace {

constexpr float kInv255 = 0.0039215688593685627f;  // 0x3B808081, fl(1/255)

struct Level {
    const std::uint8_t *data;
    int w;
    int h;
    std::size_t stride;
};

Level View(const TextureLevel &level, Format format) {
    return Level{level.texels.data(), level.width, level.height,
                 static_cast<std::size_t>(level.width) * BytesPerTexel(format)};
}

struct Rgba {
    float c[4];
};

inline std::uint32_t Load32(const std::uint8_t *p) {
    return static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1]) << 8 |
           static_cast<std::uint32_t>(p[2]) << 16 | static_cast<std::uint32_t>(p[3]) << 24;
}

inline std::uint16_t Load16(const std::uint8_t *p) {
    return static_cast<std::uint16_t>(p[0] | p[1] << 8);
}

// The per byte decodes as tables. Neither computation meets a subnormal, so the tables built at
// load time under any MXCSR hold the bits of the FTZ evaluation.
struct ByteDecodes {
    float srgb[256];
    float unorm[256];
};

ByteDecodes BuildByteDecodes() {
    ByteDecodes d{};
    for (int v = 0; v < 256; ++v) {
        d.srgb[v] = SrgbDecode(static_cast<float>(v));
        d.unorm[v] = ftz::Mul(static_cast<float>(v), kInv255);
    }
    return d;
}

const ByteDecodes kByteDecodes = BuildByteDecodes();

inline float Unorm8(std::uint32_t v) { return kByteDecodes.unorm[v & 0xFFu]; }
inline float Srgb8(std::uint32_t v) { return kByteDecodes.srgb[v & 0xFFu]; }

template <Format F>
Rgba Fetch(const Level &level, int x, int y) {
    const std::uint8_t *p = level.data + static_cast<std::size_t>(y) * level.stride +
                            static_cast<std::size_t>(x) * BytesPerTexel(F);
    Rgba r;
    if constexpr (F == Format::R8G8B8X8_SRGB || F == Format::R8G8B8A8_SRGB) {
        const std::uint32_t v = Load32(p);
        r.c[0] = Srgb8(v);
        r.c[1] = Srgb8(v >> 8);
        r.c[2] = Srgb8(v >> 16);
        r.c[3] = F == Format::R8G8B8A8_SRGB ? Unorm8(v >> 24) : 1.0f;
    } else if constexpr (F == Format::R8G8B8X8_UNORM || F == Format::R8G8B8A8_UNORM) {
        const std::uint32_t v = Load32(p);
        r.c[0] = Unorm8(v & 0xFFu);
        r.c[1] = Unorm8((v >> 8) & 0xFFu);
        r.c[2] = Unorm8((v >> 16) & 0xFFu);
        r.c[3] = F == Format::R8G8B8A8_UNORM ? Unorm8(v >> 24) : 1.0f;
    } else if constexpr (F == Format::R16G16B16A16_FLOAT) {
        for (int k = 0; k < 4; ++k) {
            r.c[k] = HalfToFloat(Load16(p + 2 * k));
        }
    } else {
        static_assert(F == Format::Z16_UNORM, "format without a sampler");
        const float d = ftz::Mul(static_cast<float>(Load16(p)), FromBits(0x37800080u));
        r.c[0] = d;
        r.c[1] = d;
        r.c[2] = d;
        r.c[3] = 1.0f;
    }
    return r;
}

inline float Lerp(float w, float a, float b) { return ftz::Fma(w, ftz::Sub(b, a), a); }

// Bilinear filtering of the four texels, x first on both rows, then y.
inline Rgba Filter(float fx, float fy, const Rgba &t00, const Rgba &t10, const Rgba &t01,
                   const Rgba &t11) {
    Rgba r;
    for (int k = 0; k < 4; ++k) {
        const float r0 = Lerp(fx, t00.c[k], t10.c[k]);
        const float r1 = Lerp(fx, t01.c[k], t11.c[k]);
        r.c[k] = Lerp(fy, r0, r1);
    }
    return r;
}

// REPEAT on a power-of-two size: u = s * w - 0.5, the integer parts masked.
inline void RepeatCoord(float s, int size, int &i0, int &i1, float &frac) {
    const float u = ftz::Sub(ftz::Mul(s, static_cast<float>(size)), 0.5f);
    const float fl = ftz::Floor(u);
    frac = ftz::Sub(u, fl);
    const std::uint32_t i = static_cast<std::uint32_t>(CvttPs2Dq(fl));
    const std::uint32_t mask = static_cast<std::uint32_t>(size - 1);
    i0 = static_cast<int>(i & mask);
    i1 = static_cast<int>((i + 1u) & mask);
}

// CLAMP_TO_EDGE for linear filtering: u = max(min(s * w, w) - 0.5, 0), the second texel clamped.
inline void ClampCoord(float s, int size, int &i0, int &i1, float &frac) {
    const float wf = static_cast<float>(size);
    const float u = ftz::MaxPs(ftz::Sub(ftz::MinPs(ftz::Mul(s, wf), wf), 0.5f), 0.0f);
    const float fl = ftz::Floor(u);
    frac = ftz::Sub(u, fl);
    i0 = CvttPs2Dq(fl);
    const std::int32_t next = static_cast<std::int32_t>(static_cast<std::uint32_t>(i0) + 1u);
    i1 = next < size - 1 ? next : size - 1;
}

template <Format F, bool kRepeat>
Rgba Bilinear(const Level &level, float s, float t) {
    int x0, x1, y0, y1;
    float fx, fy;
    if constexpr (kRepeat) {
        RepeatCoord(s, level.w, x0, x1, fx);
        RepeatCoord(t, level.h, y0, y1, fy);
    } else {
        ClampCoord(s, level.w, x0, x1, fx);
        ClampCoord(t, level.h, y0, y1, fy);
    }
    return Filter(fx, fy, Fetch<F>(level, x0, y0), Fetch<F>(level, x1, y0), Fetch<F>(level, x0, y1),
                  Fetch<F>(level, x1, y1));
}

// The per-quad part of anisotropic filtering: the number of probes, the major axis and the level
// of detail, from the coarse derivatives of lanes 0, 1 and 2 scaled by the base size.
struct Footprint {
    int probes;
    bool major_x;
    float lod;
};

Footprint QuadFootprint(const float *s, const float *t, int width, int height) {
    const float wf = static_cast<float>(width);
    const float hf = static_cast<float>(height);
    const float dsdx = ftz::Mul(ftz::Sub(s[1], s[0]), wf);
    const float dsdy = ftz::Mul(ftz::Sub(s[2], s[0]), wf);
    const float dtdx = ftz::Mul(ftz::Sub(t[1], t[0]), hf);
    const float dtdy = ftz::Mul(ftz::Sub(t[2], t[0]), hf);
    const float rho_x = ftz::Add(ftz::Mul(dsdx, dsdx), ftz::Mul(dtdx, dtdx));
    const float rho_y = ftz::Add(ftz::Mul(dsdy, dsdy), ftz::Mul(dtdy, dtdy));
    const float major = ftz::MaxPs(rho_x, rho_y);
    const float minor = ftz::MinPs(rho_x, rho_y);
    const float ratio = ftz::MinPs(ftz::MaxPs(ftz::Div(major, minor), 1.0f), 256.0f);
    Footprint f;
    f.probes = CvttPs2Dq(ftz::Ceil(ftz::Sqrt(ratio)));
    f.major_x = ftz::Ugt(rho_x, rho_y);
    f.lod = ftz::Mul(FastLog2(ftz::Div(major, ratio)), 0.5f);
    return f;
}

// Per lane probe directions from the fine derivatives in normalized coordinates: along x, lanes
// (0, 1) and (2, 3) take the difference of their row, along y, lanes (0, 2) and (1, 3) of their
// column.
inline void ProbeDirections(const float *v, bool major_x, float *d) {
    if (major_x) {
        d[0] = d[1] = ftz::Sub(v[1], v[0]);
        d[2] = d[3] = ftz::Sub(v[3], v[2]);
    } else {
        d[0] = d[2] = ftz::Sub(v[2], v[0]);
        d[1] = d[3] = ftz::Sub(v[3], v[1]);
    }
}

// The probe offsets: (i + (0.5 - 0.5 N)) / (N + 1) as the JIT code rounds it.
struct ProbeStep {
    float inv_n;
    float inv_n1;
    float base;
};

inline ProbeStep Probes(int n) {
    const float nf = static_cast<float>(n);
    return ProbeStep{ftz::Div(1.0f, nf), ftz::Div(1.0f, ftz::Add(nf, 1.0f)),
                     ftz::Add(ftz::Mul(nf, -0.5f), 0.5f)};
}

template <Format F>
void AnisoTrilinearQuad(const Texture &texture, const float *s, const float *t, float out[4][4]) {
    const int first = 0;
    const int last = static_cast<int>(texture.levels.size()) - 1;
    const Footprint fp =
        QuadFootprint(s, t, texture.levels[first].width, texture.levels[first].height);
    const float fl = ftz::Floor(fp.lod);
    float frac = ftz::Sub(fp.lod, fl);
    const std::int32_t ipart = CvttPs2Dq(fl);
    std::int32_t level0 = static_cast<std::int32_t>(static_cast<std::uint32_t>(ipart) + first);
    std::int32_t level1 = static_cast<std::int32_t>(static_cast<std::uint32_t>(level0) + 1u);
    if (level0 < first) {
        level0 = first;
        level1 = first;
        frac = 0.0f;
    }
    if (level0 >= last) {
        level0 = last;
        level1 = last;
        frac = 0.0f;
    }
    const Level l0 = View(texture.levels[level0], F);
    const Level l1 = View(texture.levels[level1], F);
    const bool second = ftz::Ugt(frac, 0.0f);
    const float weight = ftz::MaxPs(frac, 0.0f);
    float ds[4], dt[4];
    ProbeDirections(s, fp.major_x, ds);
    ProbeDirections(t, fp.major_x, dt);
    const ProbeStep step = Probes(fp.probes);
    float acc[4][4] = {};
    for (int i = 0; i < fp.probes; ++i) {
        const float off = ftz::Mul(ftz::Add(static_cast<float>(i), step.base), step.inv_n1);
        for (int lane = 0; lane < 4; ++lane) {
            const float ps = ftz::Add(s[lane], ftz::Mul(off, ds[lane]));
            const float pt = ftz::Add(t[lane], ftz::Mul(off, dt[lane]));
            Rgba v = Bilinear<F, true>(l0, ps, pt);
            if (second) {
                const Rgba v1 = Bilinear<F, true>(l1, ps, pt);
                for (int k = 0; k < 4; ++k) {
                    v.c[k] = Lerp(weight, v.c[k], v1.c[k]);
                }
            }
            for (int k = 0; k < 4; ++k) {
                acc[k][lane] = ftz::Add(acc[k][lane], v.c[k]);
            }
        }
    }
    for (int k = 0; k < 4; ++k) {
        for (int lane = 0; lane < 4; ++lane) {
            out[k][lane] = ftz::Mul(acc[k][lane], step.inv_n);
        }
    }
}

// AnisoTrilinearQuad and AnisoLinearQuad for two quads, eight lanes, at a time: the per quad part
// as above, the per lane part in lanes. Probes past a quad's own count and the second level of a
// quad without one are masked, so each lane gets the bits of the scalar code. Only the channels in
// kChannels are computed, bit c for channel c.
constexpr unsigned kChannelsR = 1u, kChannelsRgb = 7u, kChannelsRgba = 15u;

constexpr bool ForcesAlphaOneFormat(Format f) {
    return f != Format::R8G8B8A8_SRGB && f != Format::R8G8B8A8_UNORM &&
           f != Format::R16G16B16A16_FLOAT;
}

struct PairLevel {
    const std::uint8_t *data[2];  // lanes 0 to 3, lanes 4 to 7
    lanes::F8 wf, hf;
    lanes::I8 wmask, hmask, stride;
};

inline lanes::F8 Lerp8(lanes::F8 w, lanes::F8 a, lanes::F8 b) {
    return lanes::Fma(w, lanes::Sub(b, a), a);
}

// SrgbDecode of eight bytes as floats.
inline lanes::F8 SrgbDecode8(lanes::F8 x) {
    using namespace lanes;
    const F8 lin = Mul(x, Splat(FromBits(0x39A32D56u)));
    const F8 x2 = Mul(x, x);
    const F8 odd = Fma(x2, Splat(FromBits(0x329C092Bu)), Splat(FromBits(0x37456107u)));
    const F8 even = Fma(x2, Splat(FromBits(0x3732EE61u)), Splat(FromBits(0x3B16BB99u)));
    const F8 pw = Fma(odd, x, even);
    return Select(Ugt(x, Splat(15.0f)), pw, lin);
}

// Channel c of eight RGBA8 texels, sRGB decoded for colour channels of sRGB formats.
template <bool kSrgb>
inline lanes::F8 DecodeChannel8(lanes::I8 texels, int c) {
    using namespace lanes;
    const F8 v = ToFloat(AndI(SrlI(texels, 8 * c), SplatI(0xFF)));
    if constexpr (kSrgb) {
        if (c < 3) {
            return SrgbDecode8(v);
        }
    }
    return Mul(v, Splat(kInv255));
}

// RepeatCoord or ClampCoord per lane: the texel pair i0, i1 and the fraction.
template <bool kRepeat>
inline void Coord8(lanes::F8 s, lanes::F8 size, lanes::I8 max, lanes::I8 &i0, lanes::I8 &i1,
                   lanes::F8 &frac) {
    using namespace lanes;
    const F8 u = kRepeat ? Sub(Mul(s, size), Splat(0.5f))
                         : MaxPs(Sub(MinPs(Mul(s, size), size), Splat(0.5f)), Splat(0.0f));
    const F8 fl = Floor(u);
    frac = Sub(u, fl);
    const I8 i = CvttPs2Dq(fl);
    const I8 next = AddI(i, SplatI(1));
    if (kRepeat) {
        i0 = AndI(i, max);
        i1 = AndI(next, max);
    } else {
        i0 = i;
        i1 = MinI(next, max);
    }
}

#if CUJU_RENDER_LANES_AVX2
// The two adjacent 32-bit texels at lo + offsets[i] (lanes 0 to 3) or hi + offsets[i] (lanes 4 to
// 7): first in `a`, second in `b`.
inline void GatherPair64(const void *lo, const void *hi, lanes::I8 offsets, lanes::I8 &a,
                         lanes::I8 &b) {
    using namespace lanes;
    const Offsets8 o(offsets);
    const __m128i p0 = _mm_unpacklo_epi64(Load64At(lo, o.o[0]), Load64At(lo, o.o[1]));
    const __m128i p1 = _mm_unpacklo_epi64(Load64At(lo, o.o[2]), Load64At(lo, o.o[3]));
    const __m128i p2 = _mm_unpacklo_epi64(Load64At(hi, o.o[4]), Load64At(hi, o.o[5]));
    const __m128i p3 = _mm_unpacklo_epi64(Load64At(hi, o.o[6]), Load64At(hi, o.o[7]));
    const __m128 e0 = _mm_shuffle_ps(_mm_castsi128_ps(p0), _mm_castsi128_ps(p1), 0x88);
    const __m128 d0 = _mm_shuffle_ps(_mm_castsi128_ps(p0), _mm_castsi128_ps(p1), 0xDD);
    const __m128 e1 = _mm_shuffle_ps(_mm_castsi128_ps(p2), _mm_castsi128_ps(p3), 0x88);
    const __m128 d1 = _mm_shuffle_ps(_mm_castsi128_ps(p2), _mm_castsi128_ps(p3), 0xDD);
    a.v = _mm256_castps_si256(_mm256_insertf128_ps(_mm256_castps128_ps256(e0), e1, 1));
    b.v = _mm256_castps_si256(_mm256_insertf128_ps(_mm256_castps128_ps256(d0), d1, 1));
}
#endif

template <bool kSrgb, unsigned kChannels, bool kRepeat>
inline void Bilinear8(const PairLevel &l, lanes::F8 ps, lanes::F8 pt, lanes::F8 *v) {
    using namespace lanes;
    I8 ix0, ix1, iy0, iy1;
    F8 fx, fy;
    Coord8<kRepeat>(ps, l.wf, l.wmask, ix0, ix1, fx);
    Coord8<kRepeat>(pt, l.hf, l.hmask, iy0, iy1, fy);
    const I8 x0 = ShlI(ix0, 2);
    const I8 x1 = ShlI(ix1, 2);
    const I8 r0 = MulLoI(iy0, l.stride);
    const I8 r1 = MulLoI(iy1, l.stride);
    I8 t00, t10, t01, t11;
#if CUJU_RENDER_LANES_AVX2
    if (MoveMask(AsFloat(EqI(x1, AddI(x0, SplatI(4))))) == 0xFF) {
        GatherPair64(l.data[0], l.data[1], AddI(r0, x0), t00, t10);
        GatherPair64(l.data[0], l.data[1], AddI(r1, x0), t01, t11);
    } else
#endif
    {
        t00 = Gather32x2(l.data[0], l.data[1], AddI(r0, x0));
        t10 = Gather32x2(l.data[0], l.data[1], AddI(r0, x1));
        t01 = Gather32x2(l.data[0], l.data[1], AddI(r1, x0));
        t11 = Gather32x2(l.data[0], l.data[1], AddI(r1, x1));
    }
    for (int c = 0; c < 4; ++c) {
        if (!(kChannels & (1u << c))) {
            continue;
        }
        const F8 row0 = Lerp8(fx, DecodeChannel8<kSrgb>(t00, c), DecodeChannel8<kSrgb>(t10, c));
        const F8 row1 = Lerp8(fx, DecodeChannel8<kSrgb>(t01, c), DecodeChannel8<kSrgb>(t11, c));
        v[c] = Lerp8(fy, row0, row1);
    }
}

alignas(32) constexpr std::int32_t kPermDx0[8] = {0, 0, 2, 2, 4, 4, 6, 6};
alignas(32) constexpr std::int32_t kPermDx1[8] = {1, 1, 3, 3, 5, 5, 7, 7};
alignas(32) constexpr std::int32_t kPermDy0[8] = {0, 1, 0, 1, 4, 5, 4, 5};
alignas(32) constexpr std::int32_t kPermDy1[8] = {2, 3, 2, 3, 6, 7, 6, 7};

// FastLog2 per lane.
inline lanes::F8 FastLog2x8(lanes::F8 x) {
    using namespace lanes;
    const I8 b = AsInt(x);
    const F8 e = ToFloat(SubI(AndI(SrlI(b, 23), SplatI(0xFF)), SplatI(128)));
    const F8 m = AsFloat(OrI(AndI(b, SplatI(0x7FFFFF)), SplatI(0x3F800000)));
    return Add(e, m);
}

// The per quad part of AnisoTrilinearQuad for up to four quads, quad k in lane k: footprint, level
// pair, blend weight and probe steps.
struct QuadSetup {
    alignas(32) std::int32_t probes[8];
    alignas(32) std::int32_t major_x[8];
    alignas(32) std::int32_t level0[8];
    alignas(32) std::int32_t level1[8];
    alignas(32) std::int32_t second[8];
    alignas(32) float weight[8];
    alignas(32) float inv_n[8];
    alignas(32) float inv_n1[8];
    alignas(32) float base[8];
};

// Without trilinear, level 0 alone, as the anisotropic linear kinds sample it.
void SetupQuads(const Texture &texture, bool trilinear, int nq, const float *s, const float *t,
                QuadSetup &q) {
    using namespace lanes;
    alignas(32) float s0[8] = {}, s1[8] = {}, s2[8] = {}, t0[8] = {}, t1[8] = {}, t2[8] = {};
    for (int k = 0; k < nq; ++k) {
        s0[k] = s[4 * k];
        s1[k] = s[4 * k + 1];
        s2[k] = s[4 * k + 2];
        t0[k] = t[4 * k];
        t1[k] = t[4 * k + 1];
        t2[k] = t[4 * k + 2];
    }
    const F8 wf = Splat(static_cast<float>(texture.levels[0].width));
    const F8 hf = Splat(static_cast<float>(texture.levels[0].height));
    const F8 S0 = Load(s0), T0 = Load(t0);
    const F8 dsdx = Mul(Sub(Load(s1), S0), wf);
    const F8 dsdy = Mul(Sub(Load(s2), S0), wf);
    const F8 dtdx = Mul(Sub(Load(t1), T0), hf);
    const F8 dtdy = Mul(Sub(Load(t2), T0), hf);
    const F8 rho_x = Add(Mul(dsdx, dsdx), Mul(dtdx, dtdx));
    const F8 rho_y = Add(Mul(dsdy, dsdy), Mul(dtdy, dtdy));
    const F8 major = MaxPs(rho_x, rho_y);
    const F8 minor = MinPs(rho_x, rho_y);
    const F8 ratio = MinPs(MaxPs(Div(major, minor), Splat(1.0f)), Splat(256.0f));
    const I8 n = CvttPs2Dq(Ceil(Sqrt(ratio)));
    const F8 lod = Mul(FastLog2x8(Div(major, ratio)), Splat(0.5f));
    const F8 fl = Floor(lod);
    F8 frac = Sub(lod, fl);
    I8 level0 = CvttPs2Dq(fl);
    I8 level1 = AddI(level0, SplatI(1));
    const I8 below = GtI(SplatI(0), level0);
    level0 = SelectI(below, SplatI(0), level0);
    level1 = SelectI(below, SplatI(0), level1);
    frac = Select(AsFloat(below), Splat(0.0f), frac);
    const int last = trilinear ? static_cast<int>(texture.levels.size()) - 1 : 0;
    const I8 above = XorI(GtI(SplatI(last), level0), SplatI(-1));
    level0 = SelectI(above, SplatI(last), level0);
    level1 = SelectI(above, SplatI(last), level1);
    frac = Select(AsFloat(above), Splat(0.0f), frac);
    const F8 nf = ToFloat(n);
    StoreI(q.probes, n);
    StoreI(q.major_x, AsInt(Ugt(rho_x, rho_y)));
    StoreI(q.level0, level0);
    StoreI(q.level1, level1);
    StoreI(q.second, AsInt(Ugt(frac, Splat(0.0f))));
    Store(q.weight, MaxPs(frac, Splat(0.0f)));
    Store(q.inv_n, Div(Splat(1.0f), nf));
    Store(q.inv_n1, Div(Splat(1.0f), Add(nf, Splat(1.0f))));
    Store(q.base, Add(Mul(nf, Splat(-0.5f)), Splat(0.5f)));
}

// The per lane level parameters of quads a (lanes 0 to 3) and b (lanes 4 to 7).
inline PairLevel LevelsOf(const Texture &texture, int la, int lb) {
    using namespace lanes;
    const TextureLevel &a = texture.levels[la];
    const TextureLevel &b = texture.levels[lb];
    PairLevel l;
    l.data[0] = a.texels.data();
    l.data[1] = b.texels.data();
    l.wf = Pair(static_cast<float>(a.width), static_cast<float>(b.width));
    l.hf = Pair(static_cast<float>(a.height), static_cast<float>(b.height));
    l.wmask = PairI(a.width - 1, b.width - 1);
    l.hmask = PairI(a.height - 1, b.height - 1);
    l.stride = PairI(4 * a.width, 4 * b.width);
    return l;
}

// Probe directions of ProbeDirections for two quads of lanes.
inline lanes::F8 Directions(lanes::F8 v, lanes::F8 major_x) {
    using namespace lanes;
    const F8 along_x = Sub(Permute(v, LoadI(kPermDx1)), Permute(v, LoadI(kPermDx0)));
    const F8 along_y = Sub(Permute(v, LoadI(kPermDy1)), Permute(v, LoadI(kPermDy0)));
    return Select(major_x, along_x, along_y);
}

// Quads qa and qb of the setup (nb = 0 when qb is padding) at lanes s, t; out[c * 8 + lane].
template <Format F, unsigned kChannels, bool kRepeat>
void AnisoTrilinearPair(const Texture &texture, const QuadSetup &q, int qa, int qb, bool real_b,
                        const float *s, const float *t, float *out) {
    using namespace lanes;
    constexpr bool kSrgb = F == Format::R8G8B8X8_SRGB || F == Format::R8G8B8A8_SRGB;
    const PairLevel l0 = LevelsOf(texture, q.level0[qa], q.level0[qb]);
    const int nb = real_b ? q.probes[qb] : 0;
    const int max_probes = q.probes[qa] > nb ? q.probes[qa] : nb;
    const bool sec_a = q.second[qa] != 0, sec_b = real_b && q.second[qb] != 0;
    const F8 major_x = AsFloat(PairI(q.major_x[qa], q.major_x[qb]));
    const F8 s8 = Load(s), t8 = Load(t);
    const F8 ds8 = Directions(s8, major_x), dt8 = Directions(t8, major_x);
    const F8 base8 = Pair(q.base[qa], q.base[qb]);
    const F8 inv_n1_8 = Pair(q.inv_n1[qa], q.inv_n1[qb]);
    const I8 probes8 = PairI(q.probes[qa], nb);
    F8 acc[4] = {Splat(0.0f), Splat(0.0f), Splat(0.0f), Splat(0.0f)};
    if (sec_a || sec_b) {
        const PairLevel l1 = LevelsOf(texture, q.level1[qa], q.level1[qb]);
        const F8 weight8 = Pair(q.weight[qa], q.weight[qb]);
        const F8 second8 = AsFloat(PairI(sec_a ? -1 : 0, sec_b ? -1 : 0));
        for (int i = 0; i < max_probes; ++i) {
            const F8 active = AsFloat(GtI(probes8, SplatI(i)));
            const F8 off = Mul(Add(Splat(static_cast<float>(i)), base8), inv_n1_8);
            const F8 ps = Add(s8, Mul(off, ds8));
            const F8 pt = Add(t8, Mul(off, dt8));
            F8 v[4], v1[4];
            Bilinear8<kSrgb, kChannels, kRepeat>(l0, ps, pt, v);
            Bilinear8<kSrgb, kChannels, kRepeat>(l1, ps, pt, v1);
            for (int c = 0; c < 4; ++c) {
                if (kChannels & (1u << c)) {
                    const F8 blended = Select(second8, Lerp8(weight8, v[c], v1[c]), v[c]);
                    acc[c] = Select(active, Add(acc[c], blended), acc[c]);
                }
            }
        }
    } else {
        for (int i = 0; i < max_probes; ++i) {
            const F8 active = AsFloat(GtI(probes8, SplatI(i)));
            const F8 off = Mul(Add(Splat(static_cast<float>(i)), base8), inv_n1_8);
            const F8 ps = Add(s8, Mul(off, ds8));
            const F8 pt = Add(t8, Mul(off, dt8));
            F8 v[4];
            Bilinear8<kSrgb, kChannels, kRepeat>(l0, ps, pt, v);
            for (int c = 0; c < 4; ++c) {
                if (kChannels & (1u << c)) {
                    acc[c] = Select(active, Add(acc[c], v[c]), acc[c]);
                }
            }
        }
    }
    const F8 inv_n8 = Pair(q.inv_n[qa], q.inv_n[qb]);
    for (int c = 0; c < 4; ++c) {
        if (kChannels & (1u << c)) {
            Store(out + 8 * c, Mul(acc[c], inv_n8));
        }
    }
}

template <Format F, unsigned kChannels, bool kTrilinear, bool kRepeat>
void AnisoTrilinearQuads(const Texture &texture, int quads, const float *s, const float *t,
                         float *out) {
    const int n = 4 * quads;
    for (int g = 0; g < quads; g += 4) {
        const int ng = quads - g < 4 ? quads - g : 4;
        QuadSetup q;
        SetupQuads(texture, kTrilinear, ng, s + 4 * g, t + 4 * g, q);
        for (int k = 0; k < ng; k += 2) {
            const bool real_b = k + 1 < ng;
            alignas(32) float sv[8], tv[8], r[32];
            for (int lane = 0; lane < 8; ++lane) {
                const int src = 4 * (g + (lane < 4 || real_b ? k + lane / 4 : k)) + (lane & 3);
                sv[lane] = s[src];
                tv[lane] = t[src];
            }
            AnisoTrilinearPair<F, kChannels, kRepeat>(texture, q, k, real_b ? k + 1 : k, real_b, sv,
                                                      tv, r);
            const int nl = real_b ? 8 : 4;
            for (int c = 0; c < 4; ++c) {
                if (kChannels & (1u << c)) {
                    for (int lane = 0; lane < nl; ++lane) {
                        out[c * n + 4 * (g + k) + lane] = r[8 * c + lane];
                    }
                }
            }
        }
    }
}

template <Format F, bool kTrilinear, bool kRepeat>
void AnisoChannels(const Texture &texture, int quads, const float *s, const float *t,
                   unsigned channels, float *out) {
    constexpr bool kAlpha = !ForcesAlphaOneFormat(F);
    if (channels & 8u) {
        if constexpr (kAlpha) {
            AnisoTrilinearQuads<F, kChannelsRgba, kTrilinear, kRepeat>(texture, quads, s, t, out);
        } else {
            AnisoTrilinearQuads<F, kChannelsRgb, kTrilinear, kRepeat>(texture, quads, s, t, out);
            for (int l = 0; l < 4 * quads; ++l) {
                out[3 * 4 * quads + l] = 1.0f;
            }
        }
    } else if (channels & 6u) {
        AnisoTrilinearQuads<F, kChannelsRgb, kTrilinear, kRepeat>(texture, quads, s, t, out);
    } else {
        AnisoTrilinearQuads<F, kChannelsR, kTrilinear, kRepeat>(texture, quads, s, t, out);
    }
}

template <Format F>
void AnisoKind(SamplerKind kind, const Texture &texture, int quads, const float *s, const float *t,
               unsigned channels, float *out) {
    switch (kind) {
        case SamplerKind::kAnisoTrilinearRepeat:
            AnisoChannels<F, true, true>(texture, quads, s, t, channels, out);
            break;
        case SamplerKind::kAnisoLinearRepeat:
            AnisoChannels<F, false, true>(texture, quads, s, t, channels, out);
            break;
        default:
            AnisoChannels<F, false, false>(texture, quads, s, t, channels, out);
            break;
    }
}

template <Format F, bool kRepeat>
void AnisoLinearQuad(const Level &level, const float *s, const float *t, float out[4][4]) {
    const Footprint fp = QuadFootprint(s, t, level.w, level.h);
    float ds[4], dt[4];
    ProbeDirections(s, fp.major_x, ds);
    ProbeDirections(t, fp.major_x, dt);
    const ProbeStep step = Probes(fp.probes);
    float acc[4][4] = {};
    for (int i = 0; i < fp.probes; ++i) {
        const float off = ftz::Mul(ftz::Add(static_cast<float>(i), step.base), step.inv_n1);
        for (int lane = 0; lane < 4; ++lane) {
            const float ps = ftz::Add(s[lane], ftz::Mul(off, ds[lane]));
            const float pt = ftz::Add(t[lane], ftz::Mul(off, dt[lane]));
            const Rgba v = Bilinear<F, kRepeat>(level, ps, pt);
            for (int k = 0; k < 4; ++k) {
                acc[k][lane] = ftz::Add(acc[k][lane], v.c[k]);
            }
        }
    }
    for (int k = 0; k < 4; ++k) {
        for (int lane = 0; lane < 4; ++lane) {
            out[k][lane] = ftz::Mul(acc[k][lane], step.inv_n);
        }
    }
}

// NEAREST with CLAMP_TO_EDGE truncates s * w, then clamps to [0, w - 1].
inline int NearestClamp(float s, int size) {
    const std::int32_t i = CvttPs2Dq(ftz::Mul(s, static_cast<float>(size)));
    const std::int32_t hi = i < size - 1 ? i : size - 1;
    return hi > 0 ? hi : 0;
}

// NEAREST with REPEAT floors s * w and masks it.
inline int NearestRepeat(float s, int size) {
    const std::uint32_t i =
        static_cast<std::uint32_t>(CvttPs2Dq(ftz::Floor(ftz::Mul(s, static_cast<float>(size)))));
    return static_cast<int>(i & static_cast<std::uint32_t>(size - 1));
}

template <Format F>
void SampleTyped(SamplerKind kind, const Texture *texture, const TextureLevel &level0, int quads,
                 const float *s, const float *t, float *out) {
    const int n = 4 * quads;
    const bool alpha_one = ForcesAlphaOne(F);
    const Level base = View(level0, F);
    for (int q = 0; q < quads; ++q) {
        const float *qs = s + 4 * q;
        const float *qt = t + 4 * q;
        float r[4][4] = {};
        switch (kind) {
            case SamplerKind::kAnisoTrilinearRepeat:
                AnisoTrilinearQuad<F>(*texture, qs, qt, r);
                break;
            case SamplerKind::kAnisoLinearRepeat:
                AnisoLinearQuad<F, true>(base, qs, qt, r);
                break;
            case SamplerKind::kAnisoLinearClamp:
                AnisoLinearQuad<F, false>(base, qs, qt, r);
                break;
            case SamplerKind::kNearestClamp:
            case SamplerKind::kNearestRepeat:
                for (int lane = 0; lane < 4; ++lane) {
                    const bool clamp = kind == SamplerKind::kNearestClamp;
                    const int x =
                        clamp ? NearestClamp(qs[lane], base.w) : NearestRepeat(qs[lane], base.w);
                    const int y =
                        clamp ? NearestClamp(qt[lane], base.h) : NearestRepeat(qt[lane], base.h);
                    const Rgba v = Fetch<F>(base, x, y);
                    for (int k = 0; k < 4; ++k) {
                        r[k][lane] = v.c[k];
                    }
                }
                break;
            case SamplerKind::kLinearClampLod0:
                for (int lane = 0; lane < 4; ++lane) {
                    const Rgba v = Bilinear<F, false>(base, qs[lane], qt[lane]);
                    for (int k = 0; k < 4; ++k) {
                        r[k][lane] = v.c[k];
                    }
                }
                break;
        }
        for (int lane = 0; lane < 4; ++lane) {
            out[0 * n + 4 * q + lane] = r[0][lane];
            out[1 * n + 4 * q + lane] = r[1][lane];
            out[2 * n + 4 * q + lane] = r[2][lane];
            out[3 * n + 4 * q + lane] = alpha_one ? 1.0f : r[3][lane];
        }
    }
}

void Dispatch(SamplerKind kind, Format format, const Texture *texture, const TextureLevel &level0,
              int quads, const float *s, const float *t, float *out) {
    switch (format) {
        case Format::R8G8B8X8_SRGB:
            SampleTyped<Format::R8G8B8X8_SRGB>(kind, texture, level0, quads, s, t, out);
            break;
        case Format::R8G8B8A8_SRGB:
            SampleTyped<Format::R8G8B8A8_SRGB>(kind, texture, level0, quads, s, t, out);
            break;
        case Format::R8G8B8X8_UNORM:
            SampleTyped<Format::R8G8B8X8_UNORM>(kind, texture, level0, quads, s, t, out);
            break;
        case Format::R8G8B8A8_UNORM:
            SampleTyped<Format::R8G8B8A8_UNORM>(kind, texture, level0, quads, s, t, out);
            break;
        case Format::R16G16B16A16_FLOAT:
            SampleTyped<Format::R16G16B16A16_FLOAT>(kind, texture, level0, quads, s, t, out);
            break;
        case Format::Z16_UNORM:
            SampleTyped<Format::Z16_UNORM>(kind, texture, level0, quads, s, t, out);
            break;
        case Format::B10G10R10X2_UNORM:
            break;
    }
}

// The eight-lane anisotropic paths, for the RGBA8 formats. False for other formats and kinds.
bool AnisoLanes(SamplerKind kind, const Texture &texture, int quads, const float *s, const float *t,
                unsigned channels, float *out) {
    if (kind != SamplerKind::kAnisoTrilinearRepeat && kind != SamplerKind::kAnisoLinearRepeat &&
        kind != SamplerKind::kAnisoLinearClamp) {
        return false;
    }
    switch (texture.format) {
        case Format::R8G8B8X8_SRGB:
            AnisoKind<Format::R8G8B8X8_SRGB>(kind, texture, quads, s, t, channels, out);
            return true;
        case Format::R8G8B8A8_SRGB:
            AnisoKind<Format::R8G8B8A8_SRGB>(kind, texture, quads, s, t, channels, out);
            return true;
        case Format::R8G8B8X8_UNORM:
            AnisoKind<Format::R8G8B8X8_UNORM>(kind, texture, quads, s, t, channels, out);
            return true;
        case Format::R8G8B8A8_UNORM:
            AnisoKind<Format::R8G8B8A8_UNORM>(kind, texture, quads, s, t, channels, out);
            return true;
        default:
            return false;
    }
}

}  // namespace

void SampleQuads(SamplerKind kind, const Texture &texture, int quads, const float *s,
                 const float *t, float *out) {
    if (AnisoLanes(kind, texture, quads, s, t, 15u, out)) {
        return;
    }
    Dispatch(kind, texture.format, &texture, texture.levels[0], quads, s, t, out);
}

void SampleAnisoTrilinearQuads(const Texture &texture, int quads, const float *s, const float *t,
                               unsigned channels, float *out) {
    if (!AnisoLanes(SamplerKind::kAnisoTrilinearRepeat, texture, quads, s, t, channels, out)) {
        Dispatch(SamplerKind::kAnisoTrilinearRepeat, texture.format, &texture, texture.levels[0],
                 quads, s, t, out);
    }
}

void SampleLevelQuads(SamplerKind kind, Format format, const TextureLevel &level, int quads,
                      const float *s, const float *t, float *out) {
    if (kind != SamplerKind::kAnisoTrilinearRepeat) {
        Dispatch(kind, format, nullptr, level, quads, s, t, out);
    }
}

namespace {

// QuadFootprint and Probes for up to eight quads at once, one quad per lane, without the level of
// detail, which the shadow lookup does not use. s0, s1, s2 hold lanes 0, 1 and 2 of each quad.
struct Footprints {
    std::int32_t probes[8];
    std::int32_t major_x[8];
    float inv_n[8];
    float inv_n1[8];
    float base[8];
};

void QuadFootprints(const float *s0, const float *s1, const float *s2, const float *t0,
                    const float *t1, const float *t2, int width, int height, Footprints &f) {
    using namespace lanes;
    const F8 wf = Splat(static_cast<float>(width));
    const F8 hf = Splat(static_cast<float>(height));
    const F8 S0 = Load(s0), T0 = Load(t0);
    const F8 dsdx = Mul(Sub(Load(s1), S0), wf);
    const F8 dsdy = Mul(Sub(Load(s2), S0), wf);
    const F8 dtdx = Mul(Sub(Load(t1), T0), hf);
    const F8 dtdy = Mul(Sub(Load(t2), T0), hf);
    const F8 rho_x = Add(Mul(dsdx, dsdx), Mul(dtdx, dtdx));
    const F8 rho_y = Add(Mul(dsdy, dsdy), Mul(dtdy, dtdy));
    const F8 major = MaxPs(rho_x, rho_y);
    const F8 minor = MinPs(rho_x, rho_y);
    const F8 ratio = MinPs(MaxPs(Div(major, minor), Splat(1.0f)), Splat(256.0f));
    const I8 n = CvttPs2Dq(Ceil(Sqrt(ratio)));
    const F8 nf = ToFloat(n);
    StoreI(f.probes, n);
    StoreI(f.major_x, AsInt(Ugt(rho_x, rho_y)));
    Store(f.inv_n, Div(Splat(1.0f), nf));
    Store(f.inv_n1, Div(Splat(1.0f), Add(nf, Splat(1.0f))));
    Store(f.base, Add(Mul(nf, Splat(-0.5f)), Splat(0.5f)));
}

// Speed: the depth u16 * fl(1 / 65535) never decreases as u16 grows, so a reference r passes
// the LEQUAL compare, r <= u16 * fl(1 / 65535), for the u16 from a threshold on. The threshold
// lies within two of r * 65535, and the compares at the candidates around it give it exactly.
// Returns the threshold minus one, so that u16 passes when u16 > the result.
lanes::I8 ShadowPassBelow(lanes::F8 r) {
    using namespace lanes;
    const F8 unorm16 = Splat(FromBits(0x37800080u));
    const I8 c = CvttPs2Dq(Mul(r, Splat(65535.0f)));
    // The threshold is c + 2 less the candidates c - 1, c and c + 1 that pass, a mask of -1 each.
    I8 below = AddI(c, SplatI(1));
    for (int k = -1; k <= 1; ++k) {
        const I8 t = AddI(c, SplatI(k));
        below = AddI(below, AsInt(Ole(r, Mul(ToFloat(t), unorm16))));
    }
    return below;
}

}  // namespace

void SampleShadowQuads(const Texture &texture, int quads, const float *s, const float *t,
                       const float *ref, float *out) {
    using namespace lanes;
    const Level level = View(texture.levels[0], Format::Z16_UNORM);
    const F8 wf = Splat(static_cast<float>(level.w));
    const F8 hf = Splat(static_cast<float>(level.h));
    const I8 wmax = SplatI(level.w - 1);
    const I8 hmax = SplatI(level.h - 1);
    const F8 half = Splat(0.5f), zero = Splat(0.0f), one = Splat(1.0f);
    const int stride = static_cast<int>(level.stride);
    for (int qa = 0; qa < quads; qa += 8) {
        // The footprints of up to eight quads.
        const int nqa = quads - qa < 8 ? quads - qa : 8;
        alignas(32) float s0[8] = {}, s1[8] = {}, s2[8] = {}, t0[8] = {}, t1[8] = {}, t2[8] = {};
        for (int k = 0; k < nqa; ++k) {
            const float *qs = s + 4 * (qa + k);
            const float *qt = t + 4 * (qa + k);
            s0[k] = qs[0];
            s1[k] = qs[1];
            s2[k] = qs[2];
            t0[k] = qt[0];
            t1[k] = qt[1];
            t2[k] = qt[2];
        }
        Footprints fps;
        QuadFootprints(s0, s1, s2, t0, t1, t2, level.w, level.h, fps);
        // Two quads, eight lanes, at a time, the probes of each quad masked past its own count.
        struct Group {
            F8 s8, t8, ds8, dt8, base8, inv_n1_8, inv_n8, acc;
            I8 probes8, below;
            int q, max_probes;
            bool real_b;
        };
        Group groups[4];
        int ng = 0;
        for (int q0 = 0; q0 < nqa; q0 += 2) {
            Group &g = groups[ng++];
            g.real_b = q0 + 1 < nqa;
            const int qb = g.real_b ? q0 + 1 : q0;
            g.q = qa + q0;
            alignas(32) float sv[8], tv[8], rv[8];
            if (g.real_b) {
                std::memcpy(sv, s + 4 * g.q, sizeof sv);
                std::memcpy(tv, t + 4 * g.q, sizeof tv);
                std::memcpy(rv, ref + 4 * g.q, sizeof rv);
            } else {
                for (int lane = 0; lane < 8; ++lane) {
                    sv[lane] = s[4 * g.q + (lane & 3)];
                    tv[lane] = t[4 * g.q + (lane & 3)];
                    rv[lane] = ref[4 * g.q + (lane & 3)];
                }
            }
            const int nb = g.real_b ? fps.probes[qb] : 0;
            g.max_probes = fps.probes[q0] > nb ? fps.probes[q0] : nb;
            const F8 major_x = AsFloat(PairI(fps.major_x[q0], fps.major_x[qb]));
            g.s8 = Load(sv);
            g.t8 = Load(tv);
            g.ds8 = Directions(g.s8, major_x);
            g.dt8 = Directions(g.t8, major_x);
            g.base8 = Pair(fps.base[q0], fps.base[qb]);
            g.inv_n1_8 = Pair(fps.inv_n1[q0], fps.inv_n1[qb]);
            g.inv_n8 = Pair(fps.inv_n[q0], fps.inv_n[qb]);
            g.probes8 = PairI(fps.probes[q0], nb);
            g.below = ShadowPassBelow(MaxPs(MinPs(Load(rv), one), zero));
            g.acc = zero;
        }
        // Speed: two groups step through their probes together, each stage of a probe for both
        // before the next, so that the processor overlaps their independent chains. A group past
        // its own probes keeps its sums, as its lanes past their quad's probes do.
        for (int a = 0; a < ng; a += 2) {
            const int n = a + 1 < ng ? 2 : 1;
            Group *gs = groups + a;
            const int max_probes =
                n == 2 && gs[1].max_probes > gs[0].max_probes ? gs[1].max_probes : gs[0].max_probes;
            for (int i = 0; i < max_probes; ++i) {
                F8 fx[2], fy[2];
                I8 x0[2], y0[2], y1[2];
                for (int k = 0; k < n; ++k) {
                    const Group &g = gs[k];
                    const F8 off = Mul(Add(Splat(static_cast<float>(i)), g.base8), g.inv_n1_8);
                    const F8 ps = Add(g.s8, Mul(off, g.ds8));
                    const F8 pt = Add(g.t8, Mul(off, g.dt8));
                    // CLAMP_TO_EDGE for linear filtering: u = max(min(s * w, w) - 0.5, 0).
                    const F8 u = MaxPs(Sub(MinPs(Mul(ps, wf), wf), half), zero);
                    const F8 v = MaxPs(Sub(MinPs(Mul(pt, hf), hf), half), zero);
                    const F8 ufl = Floor(u), vfl = Floor(v);
                    fx[k] = Sub(u, ufl);
                    fy[k] = Sub(v, vfl);
                    x0[k] = CvttPs2Dq(ufl);
                    y0[k] = CvttPs2Dq(vfl);
                    y1[k] = MinI(AddI(y0[k], SplatI(1)), hmax);
                }
                I8 t00[2], t10[2], t01[2], t11[2];
                for (int k = 0; k < n; ++k) {
                    if (level.w >= 2) {
                        Gather16Pair(level.data, stride, level.w, x0[k], y0[k], t00[k], t10[k]);
                        Gather16Pair(level.data, stride, level.w, x0[k], y1[k], t01[k], t11[k]);
                    } else {
                        const I8 x1 = MinI(AddI(x0[k], SplatI(1)), wmax);
                        const I8 r0 = MulLoI(y0[k], SplatI(stride));
                        const I8 r1 = MulLoI(y1[k], SplatI(stride));
                        t00[k] = Gather16(level.data, AddI(r0, ShlI(x0[k], 1)));
                        t10[k] = Gather16(level.data, AddI(r0, ShlI(x1, 1)));
                        t01[k] = Gather16(level.data, AddI(r1, ShlI(x0[k], 1)));
                        t11[k] = Gather16(level.data, AddI(r1, ShlI(x1, 1)));
                    }
                }
                for (int k = 0; k < n; ++k) {
                    Group &g = gs[k];
                    // LEQUAL against each texel, then the weights of passing texels, selected by
                    // bit masks, so the sums along x are exact.
                    const F8 p00 = AsFloat(GtI(t00[k], g.below));
                    const F8 p10 = AsFloat(GtI(t10[k], g.below));
                    const F8 p01 = AsFloat(GtI(t01[k], g.below));
                    const F8 p11 = AsFloat(GtI(t11[k], g.below));
                    const F8 gx = Sub(one, fx[k]);
                    const F8 row0 = Add(And(p10, fx[k]), And(p00, gx));
                    const F8 row1 = Add(And(p11, fx[k]), And(p01, gx));
                    const F8 lerp = Fma(fy[k], Sub(row1, row0), row0);
                    const F8 active = AsFloat(GtI(g.probes8, SplatI(i)));
                    g.acc = Select(active, Add(g.acc, lerp), g.acc);
                }
            }
        }
        for (int k = 0; k < ng; ++k) {
            const Group &g = groups[k];
            alignas(32) float res[8];
            Store(res, Mul(g.acc, g.inv_n8));
            std::memcpy(out + 4 * g.q, res, (g.real_b ? 8 : 4) * sizeof(float));
        }
    }
}

}  // namespace cuju::render
