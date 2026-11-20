// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "gui.h"

#include <algorithm>

namespace cuju::render::scene {

void AssignSurface(assets::Surface *dst, const overlay2d::Surface &src) {
    dst->w = src.w;
    dst->h = src.h;
    dst->pitch = src.w * 4;
    dst->alpha = true;
    dst->pixels = src.px;
}

void GuiView::DerivedPosition(float *x, float *y) const {
    float tx = x_percent, ty = y_percent;
    if (parent) {
        float px = 0.0f, py = 0.0f;
        parent->DerivedPosition(&px, &py);
        tx += px;
        ty += py;
    }
    *x = tx;
    *y = ty;
}

CaptionView::CaptionView(Gui *g, const std::string &n, float x, float y, float w, float h,
                         const std::string &text)
    : GuiView(g, n, x, y, w, h) {
    int px, py, pw, ph;
    g->geometry().Coordinates(x, y, w, h, px, py, pw, ph);
    image = g->scene().CreateImage2D(n, pw, ph, true);
    caption = std::make_unique<overlay2d::Caption>(g->atlas(), g->geometry(), x, y, w, h, text,
                                                   g->render());
    if (g->render()) Upload();
}

void CaptionView::Upload() {
    AssignSurface(image->image->res.get(), caption->surface());
    image->size[0] = image->image->res->w;
    image->size[1] = image->image->res->h;
    w_percent = caption->width_percent();
    h_percent = caption->height_percent();
    gui->scene().ImageChanged(*image);
}

void CaptionView::CheckRedraw(bool redraws) const {
    if (redraws && gui->render() && caption->caption().empty())
        throw GrfCrash(
            "GRF's process crashes here: the render redraws the caption " + name +
            ", which holds no text, since its match was built while game_config.render was off; "
            "a reset with the rendering on builds a match that renders");
}

void CaptionView::SetCaption(const std::string &text) {
    caption->SetRender(gui->render());
    if (caption->SetCaption(text)) Upload();
}

void CaptionView::SetColor(const overlay2d::Color3 &color) {
    CheckRedraw(color != color_);
    color_ = color;
    caption->SetRender(gui->render());
    if (caption->SetColor(color)) Upload();
}

void CaptionView::SetOutlineColor(const overlay2d::Color3 &color) {
    CheckRedraw(color != outline_color_);
    outline_color_ = color;
    caption->SetRender(gui->render());
    if (caption->SetOutlineColor(color)) Upload();
}

void CaptionView::SetTransparency(float transparency) {
    CheckRedraw(transparency != transparency_);
    transparency_ = transparency;
    caption->SetRender(gui->render());
    if (caption->SetTransparency(transparency)) Upload();
}

void CaptionView::GetSize(float *w, float *h) const {
    *w = caption->width_percent();
    *h = caption->height_percent();
}

ImageView::ImageView(Gui *g, const std::string &n, float x, float y, float w, float h)
    : GuiView(g, n, x, y, w, h) {
    int px, py, pw, ph;
    g->geometry().Coordinates(x, y, w, h, px, py, pw, ph);
    image = g->scene().CreateImage2D(n, pw, ph, true);
}

void ImageView::LoadImage(const std::string &file) {
    overlay2d::SourceImage src;
    gui->LoadSourceImage(file, &src);
    // CreateImage2D(name + "source", w, h, false): a surface with no texture.
    source = gui->scene().surface_manager().Fetch(name + "source", false, true);
    int px, py, pw, ph;
    gui->geometry().Coordinates(x_percent, y_percent, w_percent, h_percent, px, py, pw, ph);
    AssignSurface(image->image->res.get(), overlay2d::ImageSurface(src, pw, ph));
    gui->scene().ImageChanged(*image);
}

Gui::Gui(Scene &scene, const overlay2d::GlyphAtlas &atlas, ImageLoader load_image, bool render)
    : scene_(scene),
      atlas_(atlas),
      load_image_(std::move(load_image)),
      render_(render),
      geometry_(scene.renderer().context_width(), scene.renderer().context_height()) {
    root_ = std::make_unique<GuiView>(this, "root", 0, 0, 100, 100);
    int w = scene.renderer().context_width(), h = scene.renderer().context_height();
    SurfacePtr surface = scene.surface_manager().Fetch("gui2_blackoutbackground", false, true);
    assets::Surface &s = *surface->res;
    s.w = w;
    s.h = h;
    s.pitch = w * 4;
    s.alpha = true;
    s.pixels.assign(static_cast<std::size_t>(w) * h * 4, 0);
    blackout_ = std::make_shared<Image2D>();
    blackout_->name = "gui2_blackoutbackground";
    blackout_->has_gfx = true;
    scene.SetImage(*blackout_, surface);
    AssignSurface(&s, overlay2d::BlackoutSurface(geometry_));
    scene.ImageChanged(*blackout_);
    scene.MoveImage(*blackout_, 0, 0);
    blackout_->enabled = false;
    scene.images().push_back(blackout_);
    // MenuTask shows the root.
    root_->visible = true;
}

Gui::~Gui() = default;

void Gui::UpdateImagePosition(GuiView *view) {
    std::vector<Image2DPtr> images;
    view->GetImages(images);
    for (const Image2DPtr &im : images) {
        if (!im) continue;
        float xp, yp;
        int x, y, w, h;
        view->DerivedPosition(&xp, &yp);
        geometry_.Coordinates(xp, yp, 0, 0, x, y, w, h);
        scene_.MoveImage(*im, x, y);
    }
    for (GuiView *c : view->children) UpdateImagePosition(c);
}

void Gui::UpdateImageVisibility(GuiView *view) {
    bool on = view->IsVisible();
    std::vector<Image2DPtr> images;
    view->GetImages(images);
    for (const Image2DPtr &im : images)
        if (im) im->enabled = on;
    for (GuiView *c : view->children) UpdateImageVisibility(c);
}

void Gui::AddView(GuiView *parent, GuiView *view) {
    parent->children.push_back(view);
    view->parent = parent;
    UpdateImagePosition(view);
    UpdateImageVisibility(view);
}

void Gui::Show(GuiView *view) {
    if (!view->visible) {
        view->visible = true;
        UpdateImageVisibility(view);
    }
}

void Gui::Hide(GuiView *view) {
    if (view->visible) {
        view->visible = false;
        UpdateImageVisibility(view);
    }
}

void Gui::SetPosition(GuiView *view, float x, float y) {
    view->x_percent = x;
    view->y_percent = y;
    UpdateImagePosition(view);
}

void Gui::SetZPriority(GuiView *view, int priority) {
    std::vector<Image2DPtr> images;
    view->GetImages(images);
    for (const Image2DPtr &im : images) im->poke_priority = priority;
}

void Gui::SetRecursiveZPriority(GuiView *view, int priority) {
    SetZPriority(view, priority);
    for (GuiView *c : view->children) SetRecursiveZPriority(c, priority);
}

void Gui::RemoveView(GuiView *parent, GuiView *view) {
    auto it = std::find(parent->children.begin(), parent->children.end(), view);
    if (it != parent->children.end()) {
        Hide(view);
        view->parent = nullptr;
        parent->children.erase(it);
    }
}

void Gui::Exit(GuiView *view) {
    view->exited = true;
    Hide(view);
    std::vector<GuiView *> copy = view->children;
    for (int i = static_cast<int>(copy.size()) - 1; i >= 0; i--) Exit(copy[i]);
    view->children.clear();
    if (view->parent) RemoveView(view->parent, view);
    std::vector<Image2DPtr> images;
    view->GetImages(images);
    for (const Image2DPtr &im : images)
        if (im) scene_.DeleteImage(im);
    delete view;
}

void Gui::ClearPagePath() {
    if (page_) {
        Exit(page_);
        page_ = nullptr;
    }
}

void Gui::CreateLoadingPage(const std::array<std::string, 2> &team_names,
                            const std::array<std::string, 2> &team_logos) {
    ClearPagePath();
    GuiView *page = new GuiView(this, "page_1", 0, 0, 100, 100);
    auto *loading = new ImageView(this, "image_main_loading", 0, 0, 100, 100);
    loading->LoadImage("media/menu/main/loading01.png");
    AddView(page, loading);
    Show(loading);
    auto *caption1 =
        new CaptionView(this, "main_loading_team1caption", 20, 35, 40, 5, team_names[0]);
    float w = caption1->TextWidthPercent();
    SetPosition(caption1, static_cast<float>(30 - w * 0.5), 35);
    AddView(page, caption1);
    auto *logo1 = new ImageView(this, "main_loading_team1logo", 25, 48, 10, 12.5f);
    AddView(page, logo1);
    logo1->LoadImage(team_logos[0]);
    auto *caption2 =
        new CaptionView(this, "main_loading_team2caption", 60, 35, 40, 5, team_names[1]);
    w = caption2->TextWidthPercent();
    SetPosition(caption2, static_cast<float>(70 - w * 0.5), 35);
    AddView(page, caption2);
    auto *logo2 = new ImageView(this, "main_loading_team2logo", 65, 48, 10, 12.5f);
    AddView(page, logo2);
    logo2->LoadImage(team_logos[1]);
    Show(caption1);
    Show(caption2);
    Show(logo1);
    Show(logo2);
    Show(page);
    page_ = page;
    AddView(root_.get(), page);
}

void Gui::CreateGamePage() {
    ClearPagePath();
    GuiView *page = new GuiView(this, "page_2", 0, 0, 100, 100);
    Show(page);
    page_ = page;
    AddView(root_.get(), page);
}

Radar::Radar(Gui *g, float x, float y, float w, float h) : GuiView(g, "game_radar", x, y, w, h) {
    bg = new ImageView(g, "bg_radar", 0, 0, w, h);
    g->AddView(this, bg);
    bg->LoadImage("media/menu/radar/radar.png");
    g->Show(bg);
    ball = new ImageView(g, "radar_ball", 0, 0, 1, 1.2f);
    g->AddView(this, ball);
    ball->LoadImage("media/menu/radar/ball.png");
    g->Show(ball);
    g->Show(this);
}

void Radar::ReloadAvatars(int team, int count) {
    for (ImageView *a : avatars[team]) gui->Exit(a);
    avatars[team].clear();
    for (int i = 0; i < count; i++) {
        auto *a =
            new ImageView(gui, "radar_avatar_" + std::to_string(team) + "_" + std::to_string(i), 0,
                          0, 1.2f, 1.6f);
        gui->AddView(this, a);
        a->LoadImage(team == 0 ? "media/menu/radar/p1.png" : "media/menu/radar/p2.png");
        gui->Show(a);
        avatars[team].push_back(a);
    }
}

void Radar::Put(const Vector3 &ball_position, const std::vector<Vector3> &team0,
                const std::vector<Vector3> &team1) {
    const float half_w = 55, half_h = 36;
    const Vector3 scale(1 / (half_w * 2), -(1 / (half_h * 2)), 0);
    Vector3 pos2d = ball_position.Get2D() * scale;
    pos2d = pos2d + Vector3(0.5f, 0.5f, 0);
    pos2d = pos2d * Vector3(0.96f, 0.96f, 0) + Vector3(0.02f, 0.02f, 0);
    gui->SetPosition(ball, pos2d.coordinates[0] * w_percent - 0.5f,
                     pos2d.coordinates[1] * h_percent - 0.6f);
    if (team0.size() != avatars[0].size()) ReloadAvatars(0, static_cast<int>(team0.size()));
    if (team1.size() != avatars[1].size()) ReloadAvatars(1, static_cast<int>(team1.size()));
    gui->SetZPriority(ball, 1);
    for (std::size_t i = 0; i < team0.size(); i++) {
        Vector3 p = team0[i] * scale;
        p = p + Vector3(0.5f, 0.5f, 0);
        p = p * Vector3(0.96f, 0.96f, 0) + Vector3(0.02f, 0.02f, 0);
        gui->SetPosition(avatars[0][i], p.coordinates[0] * w_percent - 0.6f,
                         p.coordinates[1] * h_percent - 0.8f);
    }
    for (std::size_t i = 0; i < team1.size(); i++) {
        Vector3 position = team1[i];
        position *= Vector3(-1, -1, 0);
        Vector3 p = position * scale;
        p = p + Vector3(0.5f, 0.5f, 0);
        p = p * Vector3(0.96f, 0.96f, 0) + Vector3(0.02f, 0.02f, 0);
        gui->SetPosition(avatars[1][i], p.coordinates[0] * w_percent - 0.6f,
                         p.coordinates[1] * h_percent - 0.8f);
    }
}

Scoreboard::Scoreboard(Gui *g, const std::array<std::string, 2> &short_names,
                       const std::array<std::string, 2> &logos)
    : GuiView(g, "scoreboard", 2, 2, 96, 4) {
    const overlay2d::Color3 white = {255, 255, 255}, black = {0, 0, 0};
    float x_offset[9] = {1, 5, 15, 19, 25, 29, 33, 39, 43};
    float content = 0.2f;
    float aspect = g->geometry().aspect();
    auto *bg = new ImageView(g, "image_scoreboard_bg", 0, 0, x_offset[8] + 1, h_percent);
    bg->LoadImage("media/menu/scoreboard_bg.png");
    g->AddView(this, bg);
    g->Show(bg);
    auto *league = new ImageView(g, "game_scoreboard_leaguelogo", x_offset[0], 0,
                                 h_percent / aspect, h_percent);
    g->AddView(this, league);
    league->LoadImage("media/menu/league.png");
    g->Show(league);
    float ch = h_percent * 0.9f;
    time =
        new CaptionView(g, "game_scoreboard_timecaption", x_offset[1] + content, 0, 5, ch, "0:00");
    names[0] = new CaptionView(g, "game_scoreboard_team1name", x_offset[3] + content, 0, 5, ch,
                               short_names[0]);
    names[1] = new CaptionView(g, "game_scoreboard_team2name", x_offset[6] + content, 0, 5, ch,
                               short_names[1]);
    goals[0] =
        new CaptionView(g, "game_scoreboard_team1goals", x_offset[4] + content, 0, 5, ch, "0");
    goals[1] =
        new CaptionView(g, "game_scoreboard_team2goals", x_offset[7] + content, 0, 5, ch, "0");
    for (CaptionView *c : {time, names[0], names[1], goals[0], goals[1]}) {
        c->SetColor(white);
        c->SetOutlineColor(black);
    }
    for (CaptionView *c : {time, names[0], names[1], goals[0], goals[1]}) {
        g->AddView(this, c);
        g->Show(c);
    }
    SetGoalCount(0, 0);
    SetGoalCount(1, 0);
    auto *logo0 = new ImageView(g, "game_scoreboard_team1logo", x_offset[2], 0, h_percent / aspect,
                                h_percent);
    g->AddView(this, logo0);
    logo0->LoadImage(logos[0]);
    g->Show(logo0);
    auto *logo1 = new ImageView(g, "game_scoreboard_team2logo", x_offset[5], 0, h_percent / aspect,
                                h_percent);
    g->AddView(this, logo1);
    logo1->LoadImage(logos[1]);
    g->Show(logo1);
    g->Show(this);
}

void Scoreboard::SetGoalCount(int team, int count) {
    goals[team]->SetCaption(overlay2d::GoalString(count));
}

}  // namespace cuju::render::scene
