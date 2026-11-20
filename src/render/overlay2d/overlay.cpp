// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports GRF's utils/gui2 (windowmanager.cpp, widgets/caption.cpp, widgets/image.cpp),
// scene/resources/surface.cpp and the strings of menu/ingame/scoreboard.cpp and
// onthepitch/match.cpp (Apache-2.0). Float expressions keep GRF's order and types.

#include "overlay.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cuju::render::overlay2d {

namespace {

inline float Clamp(float v, float lo, float hi) {
    if (lo > v) return lo;
    if (hi < v) return hi;
    return v;
}

inline std::uint8_t ToUint8(float v) { return static_cast<std::uint8_t>(static_cast<int>(v)); }

}  // namespace

WindowGeometry::WindowGeometry(int context_w, int context_h, float aspect, float margin)
    : context_w_(context_w), context_h_(context_h), aspect_(aspect), margin_(margin) {
    const int w = static_cast<int>(static_cast<float>(context_w) - margin * 2);
    const int h = static_cast<int>(static_cast<float>(context_h) - margin * 2);
    const float context_aspect = static_cast<float>(w) / static_cast<float>(h);
    if (context_aspect > aspect) {
        effective_w_ = static_cast<float>(h) * aspect;
        effective_h_ = static_cast<float>(h);
    } else {
        effective_w_ = static_cast<float>(w);
        effective_h_ = static_cast<float>(w) / aspect;
    }
    effective_w_ = Clamp(effective_w_, 1.0f, static_cast<float>(w));
    effective_h_ = Clamp(effective_h_, 1.0f, static_cast<float>(h));
}

void WindowGeometry::Coordinates(float x_percent, float y_percent, float w_percent, float h_percent,
                                 int &x, int &y, int &w, int &h) const {
    const int cw = static_cast<int>(static_cast<float>(context_w_) - margin_ * 2);
    const int ch = static_cast<int>(static_cast<float>(context_h_) - margin_ * 2);
    const float context_aspect = static_cast<float>(cw) / static_cast<float>(ch);
    int start_x, start_y;
    if (context_aspect > aspect_) {
        start_x = static_cast<int>(margin_ + (static_cast<float>(cw) - effective_w_) * 0.5f);
        start_y = static_cast<int>(margin_);
    } else {
        start_x = static_cast<int>(margin_);
        start_y = static_cast<int>(margin_ + (static_cast<float>(ch) - effective_h_) * 0.5f);
    }
    w = static_cast<int>(std::round(effective_w_ * w_percent * 0.01f));
    h = static_cast<int>(std::round(effective_h_ * h_percent * 0.01f));
    x = start_x + static_cast<int>(std::round(effective_w_ * x_percent * 0.01f));
    y = start_y + static_cast<int>(std::round(effective_h_ * y_percent * 0.01f));
    if (w < 1) w = 1;
    if (h < 1) h = 1;
}

Caption::Caption(const GlyphAtlas &atlas, const WindowGeometry &geometry, float x_percent,
                 float y_percent, float w_percent, float h_percent, const std::string &caption,
                 bool render)
    : atlas_(atlas),
      geometry_(geometry),
      x_percent_(x_percent),
      y_percent_(y_percent),
      w_percent_(w_percent),
      h_percent_(h_percent),
      render_(render) {
    int x, y, w, h;
    geometry_.Coordinates(x_percent_, y_percent_, w_percent_, h_percent_, x, y, w, h);
    size_w_ = w;
    size_h_ = h;
    surface_ = Surface(w, h, Order::kRGBA);
    SetCaption(caption);
}

bool Caption::SetCaption(const std::string &caption) {
    if (!render_) return false;
    const std::string adapted = caption.empty() ? std::string(" ") : caption;
    // GRF compares the new text with the stored uppercase text, so text with lowercase letters
    // redraws on every call.
    if (caption_ == adapted) return false;
    caption_ = adapted;
    // ::toupper byte by byte, which in the UTF-8 locale Python sets changes ASCII letters only.
    for (char &c : caption_)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    Redraw();
    return true;
}

bool Caption::SetColor(const Color3 &color) {
    if (color == color_) return false;
    color_ = color;
    Redraw();
    return render_;
}

bool Caption::SetOutlineColor(const Color3 &color) {
    if (color == outline_color_) return false;
    outline_color_ = color;
    Redraw();
    return render_;
}

bool Caption::SetTransparency(float transparency) {
    if (transparency == transparency_) return false;
    transparency_ = transparency;
    Redraw();
    return render_;
}

void Caption::Redraw() {
    if (!render_) return;
    int x, y, w, h;
    geometry_.Coordinates(x_percent_, y_percent_, w_percent_, h_percent_, x, y, w, h);
    const int outline_width = atlas_.outline();
    const Surface fill = atlas_.RenderBlended(GlyphAtlas::kFill, caption_, ToUint8(color_[0]),
                                              ToUint8(color_[1]), ToUint8(color_[2]));
    const Surface outline =
        atlas_.RenderBlended(GlyphAtlas::kOutline, caption_, ToUint8(outline_color_[0]),
                             ToUint8(outline_color_[1]), ToUint8(outline_color_[2]));
    int res_w, res_h;
    atlas_.Size(GlyphAtlas::kOutline, caption_, res_w, res_h);
    const float zoomy = static_cast<float>(h) / static_cast<float>(outline.h);
    const Surface outline_zoomed = ZoomSurface(outline, zoomy, zoomy);
    const Surface fill_zoomed = ZoomSurface(fill, zoomy, zoomy);
    const float text_w = static_cast<float>(res_w) * zoomy;
    const float text_h = static_cast<float>(res_h) * zoomy;
    text_width_percent_ = geometry_.WidthPercent(static_cast<int>(text_w));

    const int old_w = surface_.w, old_h = surface_.h;
    FillRect(surface_, 0, 0, w, h, 0, 0, 0, 0);
    if (text_w > static_cast<float>(size_w_) || text_h > static_cast<float>(size_h_)) {
        // Image2D::Resize through Surface::Resize, which zooms the old surface; its content is
        // cleared below, so only the size counts.
        const int nx = static_cast<int>(text_w), ny = static_cast<int>(text_h);
        double xfac = nx / (surface_.w * 1.0), yfac = ny / (surface_.h * 1.0);
        if (yfac == 0) yfac = xfac;
        if (xfac == 0) xfac = yfac;
        if (!(xfac == 0 || yfac == 0)) {
            int dw, dh;
            ZoomSurfaceSize(surface_.w, surface_.h, xfac, yfac, dw, dh);
            surface_ = Surface(dw, dh, Order::kRGBA);
        }
        size_w_ = nx;
        size_h_ = ny;
        w_percent_ = geometry_.WidthPercent(static_cast<int>(text_w));
        h_percent_ = geometry_.HeightPercent(static_cast<int>(text_h));
    }
    std::fill(surface_.px.begin(), surface_.px.end(), std::uint8_t{0});
    BlitBlend(outline_zoomed, surface_, 0, 0);
    const int offset = static_cast<int>(std::round(static_cast<float>(outline_width) * zoomy));
    BlitBlend(fill_zoomed, surface_, offset, offset);
    if (transparency_ > 0.0f)
        SetSurfaceAlpha(surface_, static_cast<int>((1.0f - transparency_) * 255));
    resized_ = surface_.w != old_w || surface_.h != old_h;
}

namespace {

inline std::uint32_t Le32(const std::uint8_t *p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

inline int Shift(std::uint32_t mask) {
    int s = 0;
    while (mask && !(mask & 1)) {
        mask >>= 1;
        ++s;
    }
    return s;
}

}  // namespace

bool DecodeBmp(const std::uint8_t *data, std::size_t size, SourceImage &out) {
    if (size < 54 || data[0] != 'B' || data[1] != 'M') return false;
    const std::uint32_t offset = Le32(data + 10), header = Le32(data + 14);
    const auto width = static_cast<std::int32_t>(Le32(data + 18));
    const auto height = static_cast<std::int32_t>(Le32(data + 22));
    const int bpp = data[28] | (data[29] << 8);
    const std::uint32_t compression = Le32(data + 30);
    if (width <= 0 || height == 0) return false;
    const bool bottom_up = height > 0;
    const int h = bottom_up ? height : -height, w = width;
    std::uint32_t masks[4] = {0, 0, 0, 0};
    if (bpp == 32 && compression == 3) {
        if (header < 56 && size < 14 + header + 16) return false;
        for (int k = 0; k < 4; ++k) masks[k] = Le32(data + 54 + 4 * k);
        for (int k = 0; k < 4; ++k)
            if (masks[k] >> Shift(masks[k]) != 0xFF) return false;
    } else if (!(bpp == 24 && compression == 0)) {
        return false;
    }
    const std::size_t stride = (static_cast<std::size_t>(w) * (bpp / 8) + 3) & ~std::size_t{3};
    if (offset + stride * h > size) return false;
    out.w = w;
    out.h = h;
    out.channels = bpp == 32 ? 4 : 3;
    out.px.assign(static_cast<std::size_t>(out.channels) * w * h, 0);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t *row = data + offset + stride * (bottom_up ? h - 1 - y : y);
        std::uint8_t *o = out.px.data() + static_cast<std::size_t>(out.channels) * w * y;
        for (int x = 0; x < w; ++x) {
            if (bpp == 32) {
                const std::uint32_t v = Le32(row + 4 * x);
                for (int k = 0; k < 4; ++k)
                    o[4 * x + k] = static_cast<std::uint8_t>((v & masks[k]) >> Shift(masks[k]));
            } else {
                o[3 * x + 0] = row[3 * x + 2];
                o[3 * x + 1] = row[3 * x + 1];
                o[3 * x + 2] = row[3 * x + 0];
            }
        }
    }
    return true;
}

Surface ImageSurface(const SourceImage &source, int w, int h) {
    Surface src(source.w, source.h, Order::kRGBA);
    if (source.channels == 4) {
        src.px = source.px;
    } else {
        // zoomSurface blits a 24-bit source onto an opaque 32-bit surface first.
        for (std::size_t i = 0, n = static_cast<std::size_t>(source.w) * source.h; i < n; ++i) {
            src.px[4 * i + 0] = source.px[3 * i + 0];
            src.px[4 * i + 1] = source.px[3 * i + 1];
            src.px[4 * i + 2] = source.px[3 * i + 2];
            src.px[4 * i + 3] = 255;
        }
    }
    const double zoomx = static_cast<double>(w) / source.w;
    const double zoomy = static_cast<double>(h) / source.h;
    return ZoomSurface(src, zoomx, zoomy);
}

Surface ImageSurface(const SourceImage &source, const WindowGeometry &geometry, float w_percent,
                     float h_percent) {
    int x, y, w, h;
    geometry.Coordinates(0.0f, 0.0f, w_percent, h_percent, x, y, w, h);
    return ImageSurface(source, w, h);
}

Surface BlackoutSurface(const WindowGeometry &geometry) {
    Surface s(geometry.context_w(), geometry.context_h(), Order::kRGBA);
    FillRect(s, 0, 0, s.w, s.h, 0, 0, 0, 255);
    return s;
}

std::string ClockString(unsigned long match_time_ms) {
    const int seconds = static_cast<int>(match_time_ms / 1000.0) % 60;
    const int minutes = static_cast<int>(match_time_ms / 60000.0);
    std::string s;
    if (minutes < 10) s += "0";
    s += std::to_string(minutes);
    s += ":";
    if (seconds < 10) s += "0";
    s += std::to_string(seconds);
    return s;
}

std::string GoalString(int goals) {
    std::string s;
    if (goals < 10) s += " ";
    s += std::to_string(goals);
    return s;
}

Color3 PlayerColor(int controller_index) {
    switch (controller_index % 6) {
        case 0:
            return {100.0f, 140.0f, 255.0f};  // blue
        case 1:
            return {100.0f, 255.0f, 140.0f};  // green
        case 2:
            return {255.0f, 110.0f, 110.0f};  // red
        case 3:
            return {255.0f, 255.0f, 60.0f};  // yellow
        case 4:
            return {200.0f, 80.0f, 200.0f};  // purple
        default:
            return kDefaultPlayerColor;
    }
}

Color3 OutlineOf(const Color3 &c) { return {c[0] * 0.4f, c[1] * 0.4f, c[2] * 0.4f}; }

}  // namespace cuju::render::overlay2d
