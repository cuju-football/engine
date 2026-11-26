// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GRF's gui2 window manager and the views of a match (utils/gui2, menu/ingame,
// menu/startmatch, Apache-2.0) over the scene's Image2D objects: views with their visibility and
// derived positions, captions and images that upload their surfaces when they redraw, the radar,
// the scoreboard, the message caption and the loading page.

#ifndef CUJU_RENDER_SCENE_GUI_H
#define CUJU_RENDER_SCENE_GUI_H

#include <array>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../overlay2d/atlas.h"
#include "../overlay2d/overlay.h"
#include "scene.h"

namespace cuju::render::scene {
class Gui;

struct GrfCrash : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class GuiView {
public:
    GuiView(Gui *gui, std::string name, float x, float y, float w, float h)
        : gui(gui), name(std::move(name)), x_percent(x), y_percent(y), w_percent(w), h_percent(h) {}
    virtual ~GuiView() = default;
    virtual void GetImages(std::vector<Image2DPtr> &) {}
    bool IsVisible() const { return visible && (!parent || parent->IsVisible()); }
    void DerivedPosition(float *x, float *y) const;

    Gui *gui;
    std::string name;
    float x_percent, y_percent, w_percent, h_percent;
    GuiView *parent = nullptr;
    std::vector<GuiView *> children;
    bool visible = false;
    bool exited = false;
};

class CaptionView : public GuiView {
public:
    CaptionView(Gui *gui, const std::string &name, float x, float y, float w, float h,
                const std::string &caption);
    void GetImages(std::vector<Image2DPtr> &out) override { out.push_back(image); }
    void SetCaption(const std::string &caption);
    void SetColor(const overlay2d::Color3 &color);
    void SetOutlineColor(const overlay2d::Color3 &color);
    void SetTransparency(float transparency);
    float TextWidthPercent() const { return caption->text_width_percent(); }
    void GetSize(float *w, float *h) const;

    Image2DPtr image;
    std::unique_ptr<overlay2d::Caption> caption;

private:
    void Upload();
    // GRF: a redraw of a caption that never took text, which only a caption made without rendering
    // holds, renders no text and crashes on the missing surface.
    void CheckRedraw(bool redraws) const;
    overlay2d::Color3 color_ = overlay2d::kBright1, outline_color_ = overlay2d::kDark1;
    float transparency_ = 0.0f;
};

class ImageView : public GuiView {
public:
    ImageView(Gui *gui, const std::string &name, float x, float y, float w, float h);
    void GetImages(std::vector<Image2DPtr> &out) override { out.push_back(image); }
    void LoadImage(const std::string &file);

    Image2DPtr image;
    SurfacePtr source;
};

class Gui {
public:
    using ImageLoader = std::function<bool(const std::string &path, overlay2d::SourceImage *)>;
    Gui(Scene &scene, const overlay2d::GlyphAtlas &atlas, ImageLoader load_image, bool render);
    ~Gui();

    Scene &scene() { return scene_; }
    const overlay2d::GlyphAtlas &atlas() const { return atlas_; }
    const overlay2d::WindowGeometry &geometry() const { return geometry_; }
    bool render() const { return render_; }
    // GRF's `game_config.render`, which every caption reads when it is asked to draw.
    void SetRender(bool render) { render_ = render; }
    GuiView *root() { return root_.get(); }

    void AddView(GuiView *parent, GuiView *view);
    void Show(GuiView *view);
    void Hide(GuiView *view);
    void SetPosition(GuiView *view, float x, float y);
    void SetZPriority(GuiView *view, int priority);
    void SetRecursiveZPriority(GuiView *view, int priority);
    void Exit(GuiView *view);
    bool LoadSourceImage(const std::string &path, overlay2d::SourceImage *out) {
        return load_image_(path, out);
    }

    void CreateLoadingPage(const std::array<std::string, 2> &team_names,
                           const std::array<std::string, 2> &team_logos);
    void CreateGamePage();
    void ClearPagePath();

private:
    void UpdateImagePosition(GuiView *view);
    void UpdateImageVisibility(GuiView *view);
    void RemoveView(GuiView *parent, GuiView *view);

    Scene &scene_;
    const overlay2d::GlyphAtlas &atlas_;
    ImageLoader load_image_;
    bool render_;
    overlay2d::WindowGeometry geometry_;
    std::unique_ptr<GuiView> root_;
    Image2DPtr blackout_;
    GuiView *page_ = nullptr;
};

class Radar : public GuiView {
public:
    Radar(Gui *gui, float x, float y, float w, float h);
    void ReloadAvatars(int team, int count);
    void Put(const Vector3 &ball, const std::vector<Vector3> &team0,
             const std::vector<Vector3> &team1);
    ImageView *bg = nullptr;
    ImageView *ball = nullptr;
    std::vector<ImageView *> avatars[2];
};

class Scoreboard : public GuiView {
public:
    Scoreboard(Gui *gui, const std::array<std::string, 2> &short_names,
               const std::array<std::string, 2> &logos);
    void SetTimeStr(const std::string &s) { time->SetCaption(s); }
    void SetGoalCount(int team, int count);
    CaptionView *time = nullptr;
    CaptionView *names[2] = {nullptr, nullptr};
    CaptionView *goals[2] = {nullptr, nullptr};
};

void AssignSurface(assets::Surface *dst, const overlay2d::Surface &src);

}  // namespace cuju::render::scene

#endif
