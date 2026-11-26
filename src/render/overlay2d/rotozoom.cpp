// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports zoomSurface and zoomSurfaceSize of SDL2_gfx 1.0.4 for 32-bit surfaces with smoothing,
// changed by the cuju Authors to work on cuju's Surface. The original notice follows.
//
//   SDL2_rotozoom.c: rotozoomer, zoomer and shrinker for 32bit or 8bit surfaces
//
//   Copyright (C) 2012-2014  Andreas Schiffler
//
//   This software is provided 'as-is', without any express or implied
//   warranty. In no event will the authors be held liable for any damages
//   arising from the use of this software.
//
//   Permission is granted to anyone to use this software for any purpose,
//   including commercial applications, and to alter it and redistribute it
//   freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
//
//   2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
//
//   3. This notice may not be removed or altered from any source
//   distribution.
//
//   Andreas Schiffler -- aschiffler at ferzkopp dot net

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "surface.h"

namespace cuju::render::overlay2d {
namespace {
constexpr double kValueLimit = 0.001;

inline int Shr16(int x) { return x >= 0 ? x >> 16 : -((-x + 65535) >> 16); }

inline int TruncToInt(double v) {
    if (!(v > -2147483649.0 && v < 2147483648.0)) return std::numeric_limits<int>::min();
    return static_cast<int>(v);
}

}  // namespace

void ZoomSurfaceSize(int width, int height, double zoomx, double zoomy, int &dstw, int &dsth) {
    if (zoomx < 0.0) zoomx = -zoomx;
    if (zoomy < 0.0) zoomy = -zoomy;
    if (zoomx < kValueLimit) zoomx = kValueLimit;
    if (zoomy < kValueLimit) zoomy = kValueLimit;
    dstw = static_cast<int>(std::floor(static_cast<double>(width) * zoomx + 0.5));
    dsth = static_cast<int>(std::floor(static_cast<double>(height) * zoomy + 0.5));
    if (dstw < 1) dstw = 1;
    if (dsth < 1) dsth = 1;
}

Surface ZoomSurface(const Surface &src, double zoomx, double zoomy) {
    int dw, dh;
    ZoomSurfaceSize(src.w, src.h, zoomx, zoomy, dw, dh);
    Surface dst(dw, dh, src.order);
    if (src.w == 0 || src.h == 0) return dst;

    const int spixelw = src.w - 1, spixelh = src.h - 1;
    const int sx = TruncToInt(65536.0 * static_cast<float>(spixelw) / static_cast<float>(dw - 1));
    const int sy = TruncToInt(65536.0 * static_cast<float>(spixelh) / static_cast<float>(dh - 1));
    const int ssx = (src.w << 16) - 1, ssy = (src.h << 16) - 1;
    std::vector<int> sax(dw + 1), say(dh + 1);
    int csx = 0;
    for (int x = 0; x <= dw; ++x) {
        sax[x] = csx;
        csx = static_cast<int>(static_cast<unsigned>(csx) + static_cast<unsigned>(sx));
        if (csx > ssx) csx = ssx;
    }
    int csy = 0;
    for (int y = 0; y <= dh; ++y) {
        say[y] = csy;
        csy = static_cast<int>(static_cast<unsigned>(csy) + static_cast<unsigned>(sy));
        if (csy > ssy) csy = ssy;
    }

    const std::uint8_t *base = src.px.data();
    const long gap = src.w;
    long sp = 0;
    std::uint8_t *dp = dst.px.data();
    for (int y = 0; y < dh; ++y) {
        const long csp = sp;
        for (int x = 0; x < dw; ++x) {
            const int ex = sax[x] & 0xffff, ey = say[y] & 0xffff;
            const int cx = Shr16(sax[x]), cy = Shr16(say[y]);
            const long c00 = sp;
            long c01 = sp, c10 = sp;
            if (cy < spixelh) c10 += gap;
            long c11 = c10;
            if (cx < spixelw) {
                ++c01;
                ++c11;
            }
            const std::uint8_t *p00 = base + 4 * c00, *p01 = base + 4 * c01;
            const std::uint8_t *p10 = base + 4 * c10, *p11 = base + 4 * c11;
            for (int k = 0; k < 4; ++k) {
                const int t1 = (Shr16((p01[k] - p00[k]) * ex) + p00[k]) & 0xff;
                const int t2 = (Shr16((p11[k] - p10[k]) * ex) + p10[k]) & 0xff;
                dp[k] = static_cast<std::uint8_t>(Shr16((t2 - t1) * ey) + t1);
            }
            sp += Shr16(sax[x + 1]) - Shr16(sax[x]);
            dp += 4;
        }
        sp = csp + static_cast<long>(Shr16(say[y + 1]) - Shr16(say[y])) * gap;
    }
    return dst;
}

}  // namespace cuju::render::overlay2d
