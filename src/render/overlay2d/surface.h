// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// 32-bit surfaces and the SDL 2.30 operations that GRF's overlay applies to them: fills, the
// per-pixel alpha blit, GRF's sdl_setsurfacealpha and SDL2_gfx's smoothed zoom. Rows are top
// first with no padding. The blit follows SDL (zlib licence, Copyright 1997-2024 Sam Lantinga),
// the zoom follows SDL2_gfx 1.0.4, see rotozoom.cpp.

#ifndef CUJU_RENDER_OVERLAY2D_SURFACE_H
#define CUJU_RENDER_OVERLAY2D_SURFACE_H

#include <cstdint>
#include <vector>

namespace cuju::render::overlay2d {
enum class Order : std::uint8_t { kRGBA, kBGRA };

struct Surface {
    int w = 0;
    int h = 0;
    Order order = Order::kRGBA;
    std::vector<std::uint8_t> px;

    Surface() = default;
    Surface(int width, int height, Order o)
        : w(width), h(height), order(o), px(static_cast<std::size_t>(4) * width * height, 0) {}
    std::uint8_t *Row(int y) { return px.data() + static_cast<std::size_t>(4) * w * y; }
    const std::uint8_t *Row(int y) const { return px.data() + static_cast<std::size_t>(4) * w * y; }
};

struct Channels {
    int r, g, b, a;
};
inline Channels ChannelsOf(Order o) {
    return o == Order::kRGBA ? Channels{0, 1, 2, 3} : Channels{2, 1, 0, 3};
}

void FillRect(Surface &s, int x, int y, int w, int h, std::uint8_t r, std::uint8_t g,
              std::uint8_t b, std::uint8_t a);

void BlitBlend(const Surface &src, Surface &dst, int x, int y);

// GRF's sdl_setsurfacealpha (base/sdl_surface.cpp).
void SetSurfaceAlpha(Surface &s, int alpha);

Surface ZoomSurface(const Surface &src, double zoomx, double zoomy);

void ZoomSurfaceSize(int width, int height, double zoomx, double zoomy, int &dstw, int &dsth);

}

#endif
