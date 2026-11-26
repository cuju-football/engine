// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports glGenerateMipmap of Mesa's llvmpipe (MIT licence, Copyright VMware, Inc. and others):
// util_gen_mipmap in u_gen_mipmap.c, the blit of u_blitter.c, and the blit fragment shader's
// sampling and store code as the gallivm JIT builds it.

#include "mipmap.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fpenv.h"
#include "numerics.h"
#include "sampler.h"
#include "srgb.h"

namespace cuju::render {
namespace {
constexpr std::uint16_t kRsqrtFirst[2] = {0x16a0, 0x0fff};
constexpr const char *kRsqrtSteps[2] = {
    "1431431414314314143143171431431414314311143143141431431414314314143143141431431114314314"
    "1431431414314311143143141431431114314314311431433114314331143146311431433114314331143143"
    "3114314331143143311431403114314331143143311431433114314031143143311431403114314331114336"
    "3111433331114333311143363111433331114333311143333111433331114333311143333111433331114333"
    "3111433331114333311143333111433331111111311111143111111431111111311111143111111131111114"
    "3111111131111111311111143111111131111111311111143111111131111111311111113333333333333333"
    "3333333333333333333333333333333333333333333333333333333333333330333333333333333333333330"
    "3333333333333330333333331333330113333304133333011333330413333301133333011333330413333301"
    "1333330113333301133333041333330113333301133333011333330113333301330111333301113333011133"
    "3301113333011133330111333301113333011130330111333301113333011133330111303301113333011133"
    "3301113033011133330113303301133333011330330113333301133033011333330113303301133033011333"
    "3301133033011330330113333301133033011330330113303301133301130113011301130113011301130113"
    "0113011301130113011301130113011301130113011301130113011301130113011301130113011301130110"
    "0113011330130133301301333013013030130133301301333013013330130130301301333013013030130133"
    "3013013330130130301301333013013030130133301301303013013030130133301301303013013330130130"
    "3013013030130133301301303013013030130133301301303013013030130130301301303013013330130130"
    "1301303013013030130130301301303013013030130130301301303013013030130130301301303013013030"
    "1301303013013030130130301301303013013033013030100130301301303013013030130130301301303013"
    "0130301001303013013030130130301301303010013030130130301301303010013030130130301301303010"
    "0130301301303010013030130130301301303010013030130130301001303013013030100130301301303010"
    "0130301301303010013030100130301330303030303030333030303030303030303030333030303030303030"
    "3030303330303030303030303030303030303033303030303030303030303030303030333030303030303030"
    "3030303030303030303030303030303030303033303030303030303030303030303030303030303030303030"
    "303030303030303030303030",
    "4441444744414447444144444441444744414447444144444441444444414447444144444441444444414444"
    "4441444444414444444144444441444144414444414441474144414441444147414441444144414441444144"
    "4144414441444144414441444144414441444144414441444144414141444144414441414144414443414414"
    "4341441743414414434144144341441443414414434144144341441443414414434144114341441443414414"
    "4341441143414414434144114341441443141414431414174314141443141414431414144314141443141414"
    "4314141443141414431414114314141443141414431414114314141443141411431414173141431431414314"
    "3141431731414314314143143141431431414314314143143141431431414314314143143141431431414314"
    "3141431131414314314143141431143114311434143114311431143414311431143114341431143114311431"
    "1431143114311434143114311431143114311431143114311431143114311431311143143111431431114314"
    "3111431431114311311143143111431431114311311143143111431131114314311143113111431431114311"
    "3111431131114314333111433331114333311146333111433331114333311143333111433331114333311143"
    "3331114333311143333111433331114033311143333111433331114011111114111111141111111111111114"
    "1111111111111114111111111111111411111111111111141111111111111111111111141111111111111111"
    "1111111433333333333333333333333333333333333333333333333333333333333333333333333333333333"
    "3333333333333333333333303333333333333333333333333330111433301111333011143330111433301111"
    "3330111433301111333011143330111133301114333011113330111433301111333011113330111433301111"
    "1133330111333301113333011133330411333301113333011133330111333301113333011133330111333301"
    "1133330111333301113333011133330111333304301133333011333330113330301133333011333330113333"
    "3011333330113330301133333011333330113330301133333011333330113330301133333011333033011304"
    "3301130133011304330113013301130433011301330113043301130133011301330113043301130133011301"
    "3301130433011301330113013301130430113301301133013011330130113301301133013011330130113304"
    "3011330130113301301133013011330130113301301133013011330130113301301133010133013301330133"
    "0133013301330133013301300133013301330133013301330133013301330130013301330133013301330133"
    "013301300133013301330130",
};

struct RsqrtTable {
    std::uint16_t code[2][4096];
};

constexpr int HexDigit(char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }

constexpr RsqrtTable BuildRsqrtTable() {
    RsqrtTable t{};
    for (int p = 0; p < 2; ++p) {
        int value = kRsqrtFirst[p];
        t.code[p][0] = static_cast<std::uint16_t>(value);
        for (int j = 1; j < 4096; ++j) {
            const int digit = HexDigit(kRsqrtSteps[p][(j - 1) / 2]);
            value -= (j - 1) % 2 == 0 ? digit % 3 : digit / 3;
            t.code[p][j] = static_cast<std::uint16_t>(value);
        }
    }
    return t;
}

constexpr RsqrtTable kRsqrt = BuildRsqrtTable();

}  // namespace

float RsqrtZen3(float x) {
    const std::uint32_t b = Bits(x);
    const std::uint32_t e = (b >> 23) & 0xFFu;
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) {
        return FromBits(b | 0x00400000u);
    }
    if (e == 0) {
        return FromBits((b & 0x80000000u) | 0x7F800000u);
    }
    if (b & 0x80000000u) {
        return FromBits(0xFFC00000u);
    }
    if (e == 0xFFu) {
        return 0.0f;
    }
    const std::uint32_t parity = e & 1u;
    const std::uint32_t code = kRsqrt.code[parity][(b >> 11) & 0xFFFu];
    const int shift = (static_cast<int>(126 + parity) - static_cast<int>(e)) / 2;
    const std::uint32_t exponent = static_cast<std::uint32_t>(126 + (code >> 12) + shift);
    return FromBits(exponent << 23 | (code & 0xFFFu) << 11);
}

namespace {
constexpr float kInv255 = 0.0039215688593685627f;

inline std::uint32_t Load32(const std::uint8_t *p) {
    return static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1]) << 8 |
           static_cast<std::uint32_t>(p[2]) << 16 | static_cast<std::uint32_t>(p[3]) << 24;
}

inline void Store32(std::uint8_t *p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}

inline float BlitCoord(int i, int size) {
    return ftz::Div(ftz::Add(static_cast<float>(i), 0.5f), static_cast<float>(size));
}

struct AosCoord {
    int i0;
    int i1;
    int weight;
};

inline AosCoord AosClamp(float s, int size) {
    const std::int32_t c = static_cast<std::int32_t>(
        static_cast<std::uint32_t>(CvtPs2Dq(ftz::Mul(s, static_cast<float>(size << 8)))) - 128u);
    const std::int32_t i = c >> 8;
    const bool ge0 = i >= 0;
    const bool lt = i < size - 1;
    const int i0 = lt ? (ge0 ? i : 0) : size - 1;
    return AosCoord{i0, i0 + ((ge0 && lt) ? 1 : 0), c & 255};
}

inline std::uint32_t Lerp8(std::uint32_t a, std::uint32_t b, int w) {
    const std::int32_t d =
        static_cast<std::int16_t>(static_cast<std::int32_t>(b) - static_cast<std::int32_t>(a));
    const std::int16_t y = static_cast<std::int16_t>(d * 128);
    const std::int32_t prod = ((w * static_cast<std::int32_t>(y)) >> 14) + 1;
    const std::uint32_t r = static_cast<std::uint32_t>(prod >> 1) & 0xFFu;
    return (a + r) & 0xFFu;
}

inline std::uint32_t StoreUnorm8(std::uint32_t byte) {
    const float v = ftz::Mul(static_cast<float>(byte), kInv255);
    const std::int32_t c = CvtPs2Dq(ftz::Mul(ftz::MinPs(1.0f, v), 255.0f));
    return static_cast<std::uint32_t>(c < 0 ? 0 : (c > 255 ? 255 : c));
}

void BlitAos(const TextureLevel &src, TextureLevel &dst, bool has_alpha) {
    for (int y = 0; y < dst.height; ++y) {
        const AosCoord cy = AosClamp(BlitCoord(y, dst.height), src.height);
        const std::uint8_t *row0 =
            src.texels.data() + static_cast<std::size_t>(cy.i0) * src.width * 4;
        const std::uint8_t *row1 =
            src.texels.data() + static_cast<std::size_t>(cy.i1) * src.width * 4;
        std::uint8_t *out = dst.texels.data() + static_cast<std::size_t>(y) * dst.width * 4;
        for (int x = 0; x < dst.width; ++x) {
            const AosCoord cx = AosClamp(BlitCoord(x, dst.width), src.width);
            const std::uint32_t t00 = Load32(row0 + 4 * cx.i0), t10 = Load32(row0 + 4 * cx.i1);
            const std::uint32_t t01 = Load32(row1 + 4 * cx.i0), t11 = Load32(row1 + 4 * cx.i1);
            std::uint32_t v = 0;
            for (int k = 0; k < 4; ++k) {
                const int sh = 8 * k;
                const std::uint32_t r0 = Lerp8((t00 >> sh) & 0xFFu, (t10 >> sh) & 0xFFu, cx.weight);
                const std::uint32_t r1 = Lerp8((t01 >> sh) & 0xFFu, (t11 >> sh) & 0xFFu, cx.weight);
                const std::uint32_t c =
                    k == 3 && !has_alpha ? 255u : StoreUnorm8(Lerp8(r0, r1, cy.weight));
                v |= c << sh;
            }
            Store32(out + 4 * x, v);
        }
    }
}

void BlitSrgb(Format format, const TextureLevel &src, TextureLevel &dst) {
    const bool has_alpha = format == Format::R8G8B8A8_SRGB;
    const int n = dst.width * dst.height;
    float s[4], t[4], rgba[16];
    for (int base = 0; base < n; base += 4) {
        const int lanes = n - base < 4 ? n - base : 4;
        for (int l = 0; l < 4; ++l) {
            const int p = base + (l < lanes ? l : 0);
            s[l] = BlitCoord(p % dst.width, dst.width);
            t[l] = BlitCoord(p / dst.width, dst.height);
        }
        SampleLevelQuads(SamplerKind::kLinearClampLod0, format, src, 1, s, t, rgba);
        for (int l = 0; l < lanes; ++l) {
            std::uint32_t v = 0;
            for (int k = 0; k < 3; ++k) {
                const float c = ftz::MinPs(ftz::MaxPs(rgba[4 * k + l], 0.0f), 1.0f);
                v |= static_cast<std::uint32_t>(SrgbEncodeByte(c)) << (8 * k);
            }
            if (has_alpha) {
                const float a = ftz::MinPs(ftz::MaxPs(rgba[12 + l], 0.0f), 1.0f);
                const float a2 = ftz::MinPs(ftz::MaxPs(a, 0.0f), 1.0f);
                v |= static_cast<std::uint32_t>(CvtPs2Dq(ftz::Mul(a2, 255.0f))) << 24;
            }
            Store32(dst.texels.data() + 4 * static_cast<std::size_t>(base + l), v);
        }
    }
}

void GenerateChain(Texture &texture) {
    const int count = MipLevelCount(texture.levels[0].width, texture.levels[0].height);
    texture.levels.resize(count);
    for (int l = 1; l < count; ++l) {
        const TextureLevel &src = texture.levels[l - 1];
        TextureLevel &dst = texture.levels[l];
        dst.width = src.width > 1 ? src.width / 2 : 1;
        dst.height = src.height > 1 ? src.height / 2 : 1;
        dst.texels.assign(static_cast<std::size_t>(dst.width) * dst.height * 4, 0);
        switch (texture.format) {
            case Format::R8G8B8X8_UNORM:
                BlitAos(src, dst, false);
                break;
            case Format::R8G8B8A8_UNORM:
                BlitAos(src, dst, true);
                break;
            default:
                BlitSrgb(texture.format, src, dst);
                break;
        }
    }
}

}  // namespace

int MipLevelCount(int width, int height) {
    int size = width > height ? width : height;
    int count = 1;
    while (size > 1) {
        size >>= 1;
        ++count;
    }
    return count;
}

bool CanGenerateMipmaps(Format format, int width, int height) {
    const bool eight_bit = format == Format::R8G8B8X8_UNORM || format == Format::R8G8B8A8_UNORM ||
                           format == Format::R8G8B8X8_SRGB || format == Format::R8G8B8A8_SRGB;
    const bool pot =
        width > 0 && height > 0 && (width & (width - 1)) == 0 && (height & (height - 1)) == 0;
    return eight_bit && pot;
}

void GenerateMipmaps(Texture &texture) {
    FpScope scope(FpMode::kFtzDaz);
    GenerateChain(texture);
}

}  // namespace cuju::render
