// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GameTask::PrepareRender and Match::Put over a RenderState, ported from GRF's gametask.cpp,
// match.cpp, team.cpp, officials.cpp, ball.cpp and humanoidbase.cpp, GRF is Apache-2.0.

#include "frame.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "assets/text.h"

#include "fpenv.h"
#include "scene/vertex_colors.h"

namespace cuju::render {
namespace {
using scene::Node;
using scene::Object;
using scene::ObjectPtr;
using scene::ObjectType;
using scene::Quaternion;
using scene::Vector3;

// GRF's MatchData of GameEnv::reset, teams 0 and 1, and the last names of its database.
const char *const TEAM_NAMES[2] = {"Frequentists United", "Real Bayesians"};
const char *const TEAM_SHORT_NAMES[2] = {"FRQ", "RBA"};
const char *const TEAM_URLS[2] = {"databases/default/images_teams/primeradivision/fcbarcelona",
                                  "databases/default/images_teams/primeradivision/realmadrid"};
const char *const LAST_NAMES[2][RENDER_TEAM_PLAYERS] = {
    {"Lovelace", "Turing", "Johnson", "da Vinci", "Newton", "Blackwell", "Borg", "Euler",
     "Pythagoras", "Curie", "Nixon Sutton"},
    {"Meitner", "Einstein", "Vaughaun", "Archimedinho", "Banach", "Banneker", "Goodall",
     "Copernicus", "Feynman", "Franklin", "Roy Mouton"}};
// GRF: Team::InitPlayers activates the first 11 players of the team's data.
constexpr int ACTIVE_PLAYERS = 11;
const char *const HAIR_STYLES[] = {"bald",     "short01", "short02", "medium01",
                                   "medium02", "long01",  "long02"};
const char *const HAIR_COLORS[] = {"black", "blonde", "darkblonde"};
const Vector3 HIDDEN(1000, 1000, -1000);

std::string HairStyle(int style) {
    return style >= 0 && style < 7 ? HAIR_STYLES[style] : "short01";
}

std::string HairColor(int color) {
    return color >= 0 && color < 3 ? HAIR_COLORS[color] : "darkblonde";
}

std::string Text(const char *text, std::size_t size) {
    std::size_t n = 0;
    while (n < size && text[n]) n++;
    return std::string(text, n);
}

void ComputePoses(Node &node, const Vector3 &parent_position, const Quaternion &parent_rotation,
                  bool has_parent) {
    Vector3 position = node.position;
    Quaternion rotation = node.rotation;
    if (!node.absolute && has_parent) {
        position = parent_rotation * node.position;
        position += parent_position;
        rotation = (parent_rotation * node.rotation).GetNormalized();
    }
    for (ObjectPtr &o : node.objects) {
        Vector3 p = o->position;
        Quaternion r = o->rotation;
        if (!o->absolute) {
            p = rotation * o->position;
            p += position;
            r = (rotation * o->rotation).GetNormalized();
        }
        o->spatial.derived_position = p;
        o->spatial.derived_rotation = r;
        o->spatial.aabb.reset();
        // GRF: OnMove and OnRotate hand the derived values to the graphics objects.
        o->gfx_position = p;
        o->gfx_rotation = r;
        if (o->type == ObjectType::kLight) o->light_gfx_position = p;
    }
    for (scene::NodePtr &n : node.nodes) ComputePoses(*n, position, rotation, true);
    node.aabb.reset();
}

template <typename T>
void SplitInto(const T *v, std::size_t size, scene::GeometryData &data) {
    std::size_t total = 0;
    for (scene::Mesh &m : data.meshes) total += m.vertices.size();
    if (size != total) return;
    std::size_t elem = total / scene::MESH_ELEMENTS, start = 0;
    for (scene::Mesh &m : data.meshes) {
        std::size_t me = m.vertices.size() / scene::MESH_ELEMENTS;
        for (int e = 0; e < scene::MESH_ELEMENTS; e++)
            if (me) std::memcpy(&m.vertices[e * me], v + e * elem + start, me * sizeof(float));
        start += me;
    }
}

std::vector<scene::PlanarMesh> PlanarMeshes(const scene::GeometryData &data) {
    std::vector<scene::PlanarMesh> out;
    for (const scene::Mesh &m : data.meshes) out.push_back({m.vertices, m.indices});
    return out;
}

}

SharedMedia::SharedMedia(std::string media_path, int threads)
    : media_path_(std::move(media_path)),
      threads_(threads),
      mip_cache_(std::make_shared<gl::MipCache>()) {
    std::string error;
    if (!atlas_.LoadFile(media_path_ + "/fonts/caption_glyphs.bin", &error)) SetError(error);
}

bool SharedMedia::ok() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_.empty();
}

std::string SharedMedia::error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_;
}

void SharedMedia::SetError(const std::string &error) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (error_.empty()) error_ = error;
}

const pitch::PitchTextures &SharedMedia::Pitch() {
    std::call_once(pitch_once_, [this] {
        pitch::PitchMedia media;
        std::string error;
        if (!pitch::LoadPitchMedia(data_dir() + "/media", &media, &error)) {
            SetError(error);
            return;
        }
        pitch::PitchOptions options;
        options.threads =
            std::max<int>(threads_, std::min(12u, std::thread::hardware_concurrency()));
        if (!pitch::GeneratePitch(media, options, &pitch_, &error)) SetError(error);
    });
    return pitch_;
}

const scene::VertexColorMap &SharedMedia::VertexColors() {
    std::call_once(colors_once_, [this] {
        std::string error;
        if (!scene::LoadVertexColors(data_dir(), &colors_, &error)) SetError(error);
    });
    return colors_;
}

bool SharedMedia::LoadSurface(const std::string &path, assets::Surface *surface) {
    {
        std::lock_guard<std::mutex> lock(files_mutex_);
        auto it = surfaces_.find(path);
        if (it != surfaces_.end()) {
            *surface = *it->second;
            return true;
        }
    }
    std::string error;
    if (!assets::LoadSurface(assets::text::DataPath(data_dir(), path), surface, &error))
        return false;
    std::lock_guard<std::mutex> lock(files_mutex_);
    surfaces_.emplace(path, std::make_shared<const assets::Surface>(*surface));
    return true;
}

bool SharedMedia::LoadAse(const std::string &path, assets::GeometryData *data) {
    {
        std::lock_guard<std::mutex> lock(files_mutex_);
        auto it = meshes_.find(path);
        if (it != meshes_.end()) {
            *data = *it->second;
            return true;
        }
    }
    std::string error;
    if (!assets::LoadAse(data_dir(), path, data, &error)) return false;
    std::lock_guard<std::mutex> lock(files_mutex_);
    meshes_.emplace(path, std::make_shared<const assets::GeometryData>(*data));
    return true;
}

scene::MatchLook MatchLookOf(const RenderState &state, std::uint32_t seed) {
    scene::MatchLook look;
    const RenderEpisode &episode = state.episode;
    for (int t = 0; t < 2; t++) {
        scene::TeamLook &team = look.teams[t];
        team.name = TEAM_NAMES[t];
        team.short_name = TEAM_SHORT_NAMES[t];
        team.kit_url = TEAM_URLS[t];
        team.logo_url = std::string(TEAM_URLS[t]) + "_logo.png";
        team.active = ACTIVE_PLAYERS;
        int size = episode.teamSize[t];
        if (size < 0) size = 0;
        if (size > RENDER_TEAM_PLAYERS) size = RENDER_TEAM_PLAYERS;
        for (int slot = 0; slot < size; slot++) {
            const RenderHumanoidStatic &fixed = episode.players[t][slot];
            scene::PlayerLook p;
            int index = fixed.dataIndex >= 0 && fixed.dataIndex < RENDER_TEAM_PLAYERS
                            ? fixed.dataIndex
                            : slot;
            p.last_name = LAST_NAMES[t][index];
            p.skin = fixed.skin;
            p.hair_style = HairStyle(fixed.hairStyle);
            p.hair_color = HairColor(fixed.hairColor);
            p.z_multiplier = fixed.zMultiplier;
            p.goalkeeper = fixed.kit == e_RenderKit_Goalkeeper;
            team.players.push_back(p);
        }
    }
    look.first_team = state.firstTeam;
    for (int k = 0; k < RENDER_OFFICIALS; k++) {
        const RenderHumanoidStatic &fixed = episode.officials[k];
        scene::PlayerLook &p = look.officials[k];
        p.skin = fixed.skin;
        p.hair_style = HairStyle(fixed.hairStyle);
        p.hair_color = HairColor(fixed.hairColor);
        p.z_multiplier = fixed.zMultiplier;
    }
    look.sun_position = episode.sunPosition;
    look.sun_color = episode.sunColor;
    look.seed = seed;
    return look;
}

Renderer::Renderer(const RendererOptions &options, std::shared_ptr<SharedMedia> media)
    : options_(options), media_(std::move(media)) {
    FpScope fp(FpMode::kIeee);
    if (!media_) media_ = std::make_shared<SharedMedia>(options.media_path, options.threads);
    gl::ContextOptions co;
    co.width = options.width;
    co.height = options.height;
    co.threads = options.threads;
    co.mip_cache = media_->mip_cache();
    gl_ = std::make_unique<gl::Context>(co);
    stream_ = std::make_unique<scene::GlStream>(gl_.get(), options.recorder);
    scene::Context::Shared shared;
    shared.data_dir = media_->data_dir();
    shared.atlas = &media_->atlas();
    SharedMedia *m = media_.get();
    shared.pitch = [m]() -> const pitch::PitchTextures & { return m->Pitch(); };
    shared.load_surface = [m](const std::string &path, assets::Surface *s) {
        return m->LoadSurface(path, s);
    };
    shared.load_ase = [m](const std::string &path, assets::GeometryData *g) {
        return m->LoadAse(path, g);
    };
    context_ = std::make_unique<scene::Context>(*stream_, options.width, options.height, shared);
    context_->CreateContext();
}

Renderer::~Renderer() = default;

void Renderer::Resolve(bool discard) {
    if (!pending_) return;
    pending_ = false;
    if (discard) gl_->SetDiscard(true);
    RenderMatch(*pending_state_, pending_swap_);
    if (discard) gl_->SetDiscard(false);
}

void Renderer::Flush() {
    FpScope fp(FpMode::kIeee);
    Resolve(false);
}

void Renderer::ResetBegin() {
    FpScope fp(FpMode::kIeee);
    Resolve(false);
    context_->ResetBegin();
}

void Renderer::StopMatch() {
    FpScope fp(FpMode::kIeee);
    Resolve(false);
    context_->StopMatch();
}

void Renderer::LoadingPage() {
    FpScope fp(FpMode::kIeee);
    Resolve(false);
    context_->CreateLoadingPage(
        {TEAM_NAMES[0], TEAM_NAMES[1]},
        {std::string(TEAM_URLS[0]) + "_logo.png", std::string(TEAM_URLS[1]) + "_logo.png"});
}

const scene::SkinnedModel &Renderer::Model(int body) {
    while (static_cast<int>(models_.size()) <= body) {
        int b = static_cast<int>(models_.size());
        models_.emplace_back();
        const scene::GeometryData &tmpl =
            *context_->fullbody_template->GetObject("fullbody")->data->res;
        scene::PrepareFullbodyModel(PlanarMeshes(tmpl), media_->VertexColors(), body_z_[b],
                                    scene::FullbodyBindPose(), models_.back(), baked_);
    }
    return models_[body];
}

void Renderer::Skin(int body, scene::GeometryData &data, bool rest, const RenderJoint *joints) {
    if (body < 0 || body >= static_cast<int>(body_z_.size())) return;
    const scene::SkinnedModel &model = Model(body);
    skin_buffer_.resize(model.BufferFloats());
    if (rest)
        scene::FullbodyRestBuffer(model, skin_buffer_.data());
    else
        scene::SkinFullbody(model, joints, skin_buffer_.data());
    SplitInto(skin_buffer_.data(), skin_buffer_.size(), data);
}

void Renderer::StartMatch(const RenderState &state, std::uint32_t seed) {
    FpScope fp(FpMode::kIeee);
    Resolve(false);
    scene::MatchLook look = MatchLookOf(state, seed);
    body_z_.clear();
    body_slot_.clear();
    models_.clear();
    baked_ = render_;
    for (int k = 0; k < 2; k++) {
        int t = k == 0 ? look.first_team : 1 - look.first_team;
        int size = static_cast<int>(look.teams[t].players.size());
        for (int slot = 0; slot < size && slot < look.teams[t].active; slot++) {
            body_z_.push_back(state.episode.players[t][slot].zMultiplier);
            body_slot_.push_back({t, slot});
        }
    }
    for (int k = 0; k < RENDER_OFFICIALS; k++) {
        body_z_.push_back(state.episode.officials[k].zMultiplier);
        body_slot_.push_back({-1, k});
    }
    context_->StartMatch(look, [this](int body, Object &, scene::GeometryData &data) {
        Skin(body, data, true, nullptr);
    });
    if (!netting_prepared_) {
        netting_.Prepare(PlanarMeshes(*context_->goals->GetObject("goals")->data->res));
        netting_prepared_ = true;
    } else {
        netting_.NewMatch();
    }
    shown_goals_[0] = shown_goals_[1] = 0;
    shown_message_.clear();
    message_visible_ = false;
}

void Renderer::SetRender(bool render) {
    FpScope fp(FpMode::kIeee);
    Resolve(false);
    render_ = render;
    context_->scene().SetRender(render);
    context_->gui().SetRender(render);
}

void Renderer::StateProcessed() {
    FpScope fp(FpMode::kIeee);
    Resolve(false);
    if (context_->match()) netting_.StateProcessed();
}

const std::vector<std::uint8_t> &Renderer::RenderScene(bool swap) {
    FpScope fp(FpMode::kIeee);
    Resolve(false);
    context_->scene().Render(swap, &screen_);
    return screen_;
}

void Renderer::ApplyState(const RenderState &state) {
    scene::Match &match = *context_->match();
    std::vector<scene::Match::Body> &bodies = match.bodies();
    for (std::size_t b = 0; b < bodies.size() && b < body_slot_.size(); b++) {
        const std::array<int, 2> &at = body_slot_[b];
        const RenderHumanoid &h = at[0] >= 0 ? state.players[at[0]][at[1]] : state.officials[at[1]];
        if (h.present) {
            bodies[b].fullbody_node->position = h.fullbodyOffset;
            bodies[b].hair->position = h.hairPosition;
            bodies[b].hair->rotation = h.hairRotation;
        } else if (h.hidden) {
            bodies[b].fullbody_node->position = HIDDEN;
            bodies[b].hair->position = HIDDEN;
            bodies[b].hair->rotation = h.hairRotation;
        }
    }
    match.ball()->position = state.ball.position;
    match.ball()->rotation = state.ball.orientation;
    match.yellow_card()->position = state.yellowCard.position;
    match.yellow_card()->rotation = state.yellowCard.rotation;
    match.red_card()->position = state.redCard.position;
    match.red_card()->rotation = state.redCard.rotation;
    Object &camera = *match.camera();
    camera.position = Vector3(0, 0, 0);
    camera.rotation = state.camera.orientation;
    if (camera.parent) {
        camera.parent->position = state.camera.nodePosition;
        camera.parent->rotation = state.camera.nodeOrientation;
    }
    ComputePoses(context_->scene().root(), Vector3(), Quaternion(), false);
    camera.fov = state.camera.fieldOfView;
    camera.near_cap = state.camera.nearCap;
    camera.far_cap = state.camera.farCap;
    const Vector3 position = camera.spatial.derived_position;
    const Quaternion rotation = camera.spatial.derived_rotation;
    camera.exec_position = camera.cam_position = camera.shadow_camera_position = position;
    camera.exec_rotation = camera.cam_rotation = rotation;
    camera.exec_fov = camera.cam_fov = state.camera.fieldOfView;
    camera.exec_near = camera.cam_near = state.camera.nearCap;
    camera.exec_far = camera.cam_far = state.camera.farCap;
}

const std::vector<std::uint8_t> &Renderer::Render(const RenderState &state, bool swap, bool read) {
    FpScope fp(FpMode::kIeee);
    Resolve(context_->match() != nullptr);
    if (!context_->match()) return RenderScene(swap);
    if (!read && options_.skip_unread) {
        if (!pending_state_) pending_state_ = std::make_unique<RenderState>();
        *pending_state_ = state;
        pending_swap_ = swap;
        pending_ = true;
        return screen_;
    }
    return RenderMatch(state, swap);
}

const std::vector<std::uint8_t> &Renderer::RenderMatch(const RenderState &state, bool swap) {
    scene::Match *match = context_->match();
    for (int t = 0; t < 2; t++)
        if (state.overlay.goals[t] != shown_goals_[t]) {
            match->SetGoalCount(t, state.overlay.goals[t]);
            shown_goals_[t] = state.overlay.goals[t];
        }
    std::string message = Text(state.overlay.message, sizeof state.overlay.message);
    bool visible = state.overlay.messageVisible;
    if (visible && (!message_visible_ || message != shown_message_)) {
        if (message.empty())
            match->ShowMessage();
        else
            match->SpamMessage(message);
    }
    shown_message_ = message;
    message_visible_ = visible;
    ApplyState(state);
    auto humanoid = [&](int body) -> const RenderHumanoid & {
        const std::array<int, 2> &at = body_slot_.at(body);
        return at[0] >= 0 ? state.players[at[0]][at[1]] : state.officials[at[1]];
    };
    std::vector<scene::TagInput> tags;
    for (int k = 0; k < 2; k++) {
        int t = k == 0 ? match->look().first_team : 1 - match->look().first_team;
        for (int b : match->team(t).bodies) {
            const RenderHumanoid &h = humanoid(b);
            scene::TagInput tag;
            tag.shown = h.present && h.nameCaptionShow;
            tag.keep = !h.present && !h.hidden;
            tag.geom_position = h.geomPosition;
            tag.color = {h.nameCaptionColor.coordinates[0], h.nameCaptionColor.coordinates[1],
                         h.nameCaptionColor.coordinates[2]};
            tags.push_back(tag);
        }
    }
    scene::ProjectionCamera projection;
    projection.position = match->camera()->spatial.derived_position;
    projection.rotation = match->camera()->spatial.derived_rotation;
    projection.fov = state.camera.fieldOfView;
    match->PutTags(tags, projection);
    match->PutClock(Text(state.overlay.clock, sizeof state.overlay.clock));
    if (!visible) match->HideMessage();
    std::vector<Vector3> radar[2];
    for (int t = 0; t < 2; t++)
        for (int b : match->team(t).bodies)
            if (humanoid(b).present) radar[t].push_back(humanoid(b).radarPosition);
    match->PutRadar(state.ball.radarPosition, radar[0], radar[1]);
    bool netting = netting_.Update(state.ball.position, state.ball.touchesNet);
    const int body_count = static_cast<int>(std::min(match->bodies().size(), body_slot_.size()));
    std::vector<int> skinned;
    for (int b = 0; b < body_count; b++) {
        if (match->bodies()[b].team < 0 || humanoid(b).present) {
            Model(b);
            skinned.push_back(b);
        }
    }
    if (skin_buffers_.size() < static_cast<std::size_t>(body_count))
        skin_buffers_.resize(body_count);
    gl_->Parallel(static_cast<int>(skinned.size()), [&](int k) {
        FpScope task_fp(FpMode::kIeee);
        const int b = skinned[k];
        skin_buffers_[b].resize(models_[b].BufferFloats());
        scene::SkinFullbody(models_[b], humanoid(b).joints, skin_buffers_[b].data());
    });
    match->UploadBodies(
        [&](int body, Object &, scene::GeometryData &data) {
            if (body < 0 || body >= body_count) return;
            SplitInto(skin_buffers_[body].data(), skin_buffers_[body].size(), data);
        },
        [&](int body) { return humanoid(body).present; });
    if (netting)
        match->UploadNetting([&](scene::GeometryData &goals) {
            SplitInto(netting_.Buffer().data(), netting_.Buffer().size(), goals);
        });
    context_->scene().Render(swap, &screen_);
    return screen_;
}

}
