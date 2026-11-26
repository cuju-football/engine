// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The blit's blend is the one SDL 2.30.0 applies (zlib licence, notice below), measured on the
// reference CPU over every source alpha, source value and destination value. SetSurfaceAlpha
// ports GRF's base/sdl_surface.cpp (Apache-2.0).
//
//   Simple DirectMedia Layer
//   Copyright (C) 1997-2024 Sam Lantinga <slouken@libsdl.org>
//
//   This software is provided 'as-is', without any express or implied
//   warranty.  In no event will the authors be held liable for any damages
//   arising from the use of this software.
//
//   Permission is granted to anyone to use this software for any purpose,
//   including commercial applications, and to alter it and redistribute it
//   freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must not
//      claim that you wrote the original software. If you use this software
//      in a product, an acknowledgment in the product documentation would be
//      appreciated but is not required.
//   2. Altered source versions must be plainly marked as such, and must not be
//      misrepresented as being the original software.
//   3. This notice may not be removed or altered from any source distribution.

#include "surface.h"

#include <algorithm>

namespace cuju::render::overlay2d {
void FillRect(Surface &s, int x, int y, int w, int h, std::uint8_t r, std::uint8_t g,
              std::uint8_t b, std::uint8_t a) {
    const int x0 = std::max(x, 0), y0 = std::max(y, 0);
    const int x1 = std::min(x + w, s.w), y1 = std::min(y + h, s.h);
    const Channels c = ChannelsOf(s.order);
    for (int yy = y0; yy < y1; ++yy) {
        std::uint8_t *p = s.Row(yy) + 4 * x0;
        for (int xx = x0; xx < x1; ++xx, p += 4) {
            p[c.r] = r;
            p[c.g] = g;
            p[c.b] = b;
            p[c.a] = a;
        }
    }
}

namespace {
inline std::uint8_t Mix(int s, int d, int a) {
    const int t = (s - d) * a;
    const int q = t >= 0 ? t >> 8 : -((-t + 255) >> 8);
    return static_cast<std::uint8_t>(d + q);
}

}  // namespace

void BlitBlend(const Surface &src, Surface &dst, int x, int y) {
    const int sx0 = std::max(0, -x), sy0 = std::max(0, -y);
    const int sx1 = std::min(src.w, dst.w - x), sy1 = std::min(src.h, dst.h - y);
    const Channels cs = ChannelsOf(src.order), cd = ChannelsOf(dst.order);
    for (int sy = sy0; sy < sy1; ++sy) {
        const std::uint8_t *s = src.Row(sy) + 4 * sx0;
        std::uint8_t *d = dst.Row(sy + y) + 4 * (sx0 + x);
        for (int sx = sx0; sx < sx1; ++sx, s += 4, d += 4) {
            const int a = s[cs.a];
            if (a == 0) continue;
            if (a == 255) {
                d[cd.r] = s[cs.r];
                d[cd.g] = s[cs.g];
                d[cd.b] = s[cs.b];
                d[cd.a] = 255;
                continue;
            }
            d[cd.r] = Mix(s[cs.r], d[cd.r], a);
            d[cd.g] = Mix(s[cs.g], d[cd.g], a);
            d[cd.b] = Mix(s[cs.b], d[cd.b], a);
            d[cd.a] = static_cast<std::uint8_t>(a + ((d[cd.a] * (255 - a)) >> 8));
        }
    }
}

void SetSurfaceAlpha(Surface &s, int alpha) {
    const float fa = static_cast<float>(alpha) / 256.0f;
    for (std::size_t i = 3; i < s.px.size(); i += 4) {
        float v = (static_cast<float>(s.px[i]) / 256.0f) * fa * 256.0f;
        if (0.0f > v) v = 0.0f;
        if (255.0f < v) v = 255.0f;
        s.px[i] = static_cast<std::uint8_t>(v);
    }
}

}  // namespace cuju::render::overlay2d
