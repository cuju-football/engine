// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The surfaces of GRF's 2D overlay, ported from GRF's gui2 widgets (Apache-2.0): the window
// manager's geometry, Gui2Caption and Gui2Image as they redraw their SDL surfaces, and the inputs
// and widgets of a match: scoreboard, radar, player name tags, message caption, loading page.
// Each widget keeps GRF's state, so a sequence of the same calls gives the same surfaces, and
// reports every redraw, which GRF follows with a texture upload of the surface.

#ifndef CUJU_RENDER_OVERLAY2D_OVERLAY_H
#define CUJU_RENDER_OVERLAY2D_OVERLAY_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "atlas.h"
#include "surface.h"

namespace cuju::render::overlay2d {

// Gui2WindowManager's geometry: the 5/4 band of the context, 900 x 720 at 1280 x 720.
class WindowGeometry {
public:
    WindowGeometry(int context_w, int context_h, float aspect = 5.0f / 4.0f, float margin = 0.0f);
    // GetCoordinates: pixel position and size of a view from its percents.
    void Coordinates(float x_percent, float y_percent, float w_percent, float h_percent, int &x,
                     int &y, int &w, int &h) const;
    float WidthPercent(int pixels) const { return pixels / (effective_w_ * 0.01f); }
    float HeightPercent(int pixels) const { return pixels / (effective_h_ * 0.01f); }
    float aspect() const { return aspect_; }
    int context_w() const { return context_w_; }
    int context_h() const { return context_h_; }

private:
    int context_w_, context_h_;
    float aspect_, margin_, effective_w_ = 0.0f, effective_h_ = 0.0f;
};

using Color3 = std::array<float, 3>;  // GRF's Vector3 colours, components 0 to 255

// GRF's Style colours that captions start with.
inline constexpr Color3 kBright1 = {150.0f, 180.0f, 220.0f};
inline constexpr Color3 kDark1 = {20.0f, 35.0f, 55.0f};

// Gui2Caption. The surface is the Image2D's, SDL_PIXELFORMAT_ABGR8888, uploaded as GL_RGBA.
class Caption {
public:
    // The constructor draws once, as GRF's does. With `render` false, as GRF's game config has
    // it while an episode runs without rendering, the caption ignores new text and keeps colours
    // and transparency without drawing, so the first text after SetRender(true) draws with the
    // size the caption was created with.
    Caption(const GlyphAtlas &atlas, const WindowGeometry &geometry, float x_percent,
            float y_percent, float w_percent, float h_percent, const std::string &caption,
            bool render = true);

    void SetRender(bool render) { render_ = render; }

    // Each returns true when GRF redraws, which uploads the surface.
    bool SetCaption(const std::string &caption);
    bool SetColor(const Color3 &color);
    bool SetOutlineColor(const Color3 &color);
    bool SetTransparency(float transparency);

    const Surface &surface() const { return surface_; }
    // Whether the last redraw changed the surface's size, so GRF reallocates the texture.
    bool resized() const { return resized_; }
    float text_width_percent() const { return text_width_percent_; }
    float width_percent() const { return w_percent_; }
    float height_percent() const { return h_percent_; }
    const std::string &caption() const { return caption_; }

private:
    void Redraw();

    const GlyphAtlas &atlas_;
    const WindowGeometry &geometry_;
    float x_percent_, y_percent_, w_percent_, h_percent_;
    std::string caption_;
    Color3 color_ = kBright1, outline_color_ = kDark1;
    float transparency_ = 0.0f;
    bool render_;
    float text_width_percent_ = 0.0f;
    int size_w_, size_h_;  // Image2D's size
    Surface surface_;
    bool resized_ = false;
};

// A decoded BMP as GRF's IMG_LoadBmp leaves it: 4 channels in R G B A order, or 3 in R G B.
struct SourceImage {
    int w = 0, h = 0, channels = 4;
    std::vector<std::uint8_t> px;
};

// SDL_LoadBMP_RW followed by IMG_LoadBmp's conversion, for uncompressed 24-bit BMPs and 32-bit
// BMPs with bit fields, the kinds GRF's overlay images are. Returns false for other files.
bool DecodeBmp(const std::uint8_t *data, std::size_t size, SourceImage &out);

// Gui2Image::Redraw: the source zoomed to the view's pixel size with smoothing.
Surface ImageSurface(const SourceImage &source, int w, int h);
Surface ImageSurface(const SourceImage &source, const WindowGeometry &geometry, float w_percent,
                     float h_percent);

// The blackout image of Gui2WindowManager, opaque black over the context, never drawn.
Surface BlackoutSurface(const WindowGeometry &geometry);

// Match strings as GRF builds them.
std::string ClockString(unsigned long match_time_ms);  // Match::Put's "MM:SS"
std::string GoalString(int goals);                     // Gui2ScoreBoard::SetGoalCount
// The name tag colour of a player controlled by controller `index` with GRF's keyboard AI
// controllers, e_PlayerColor(index % 6), and of other players.
Color3 PlayerColor(int controller_index);
inline constexpr Color3 kDefaultPlayerColor = {200.0f, 200.0f, 200.0f};
Color3 OutlineOf(const Color3 &color);  // buf_playerColor * 0.4f

// Inputs of the overlay of one frame, as a RenderState carries them.
struct OverlayInputs {
    int context_w = 1280, context_h = 720;
    std::array<std::string, 2> team_name;        // TeamData::GetName
    std::array<std::string, 2> team_short_name;  // TeamData::GetShortName
    std::array<const SourceImage *, 2> team_logo = {nullptr, nullptr};
    unsigned long match_time_ms = 0;
    std::array<int, 2> goals = {0, 0};
    std::string message;  // the last SpamMessage text, "" before any
    struct Tag {
        std::string last_name;
        Color3 color = kDefaultPlayerColor;
    };
    std::vector<Tag> tags;  // name tags shown this frame
};

}  // namespace cuju::render::overlay2d

#endif  // CUJU_RENDER_OVERLAY2D_OVERLAY_H
