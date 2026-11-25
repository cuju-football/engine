// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// The graphics side of GRF's match lifecycle (onthepitch/match.cpp, team.cpp, officials.cpp,
// ball.cpp, player/humanoid/humanoidbase.cpp, gametask.cpp, game_env.cpp, Apache-2.0): what the
// Match constructor builds in the scene and the GUI in GRF's order, what Match::Put and
// GameTask::PrepareRender upload each frame, what Match::Exit removes, and the process-wide
// state GRF keeps in its GameContext across matches.

#ifndef CUJU_RENDER_SCENE_MATCH_H
#define CUJU_RENDER_SCENE_MATCH_H

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../pitch/pitch.h"
#include "gui.h"
#include "scene.h"

namespace cuju::render::scene {
struct PlayerLook {
    std::string last_name;
    int skin = 1;
    std::string hair_style = "short01";
    std::string hair_color = "darkblonde";
    float height = 1.80f;
    float z_multiplier = 0.0f;
    bool goalkeeper = false;
    std::string kit;
};

struct TeamLook {
    std::string name, short_name;
    std::string kit_url;
    std::string logo_url;
    std::vector<PlayerLook> players;
    int active = 0;
};

struct MatchLook {
    std::array<TeamLook, 2> teams;
    int first_team = 0;
    std::array<PlayerLook, 3> officials;
    Vector3 sun_position;
    Vector3 sun_color;
    std::uint32_t seed = 42;
};

struct ProjectionCamera {
    Vector3 position;
    Quaternion rotation;
    float fov = 0;
};

struct TagInput {
    bool shown = false;
    // GRF's Put2D and Hide2D pass over inactive players, so the tag of a player that a load made
    // inactive keeps its last state, shown with the name where `Activate` left it at first.
    bool keep = false;
    Vector3 geom_position;
    overlay2d::Color3 color = overlay2d::kDefaultPlayerColor;
};

Vector3 ProjectedCoord(const Vector3 &pos, const ProjectionCamera &camera, int context_w,
                       int context_h, float aspect2d);

using SkinFunction = std::function<void(int body, Object &fullbody, GeometryData &data)>;

class Context;

class Match {
public:
    struct Body {
        NodePtr fullbody_node;
        ObjectPtr fullbody;
        ObjectPtr hair;
        float z_multiplier = 1;
        int team = -1;
        CaptionView *tag = nullptr;
    };
    struct TeamNodes {
        NodePtr node;
        std::vector<CaptionView *> tags;
        std::vector<int> bodies;
    };

    Match(Context &context, const MatchLook &look, const SkinFunction &skin);
    ~Match();
    void Exit();

    std::vector<Body> &bodies() { return bodies_; }
    TeamNodes &team(int t) { return teams_[t]; }
    ObjectPtr ball() { return ball_; }
    ObjectPtr camera() { return camera_; }
    ObjectPtr sun() { return sun_; }
    ObjectPtr yellow_card() { return yellow_; }
    ObjectPtr red_card() { return red_; }
    Radar *radar() { return radar_; }
    Scoreboard *scoreboard() { return scoreboard_; }
    CaptionView *message() { return message_; }
    const MatchLook &look() const { return look_; }

    void UploadBodies(const SkinFunction &skin, const std::function<bool(int body)> &active = {});
    void PutTags(const std::vector<TagInput> &tags, const ProjectionCamera &camera);
    void PutClock(const std::string &time);
    void HideMessage();
    void SetGoalCount(int team, int goals);
    void SpamMessage(const std::string &message);
    void ShowMessage();
    void PutRadar(const Vector3 &ball, const std::vector<Vector3> &team0,
                  const std::vector<Vector3> &team1) {
        radar_->Put(ball, team0, team1);
    }
    void UploadNetting(const std::function<void(GeometryData &goals)> &netting);

private:
    Body CreateBody(int team, const PlayerLook &player, const SurfacePtr &kit, Node &target,
                    const SkinFunction &skin);

    Context &context_;
    MatchLook look_;
    NodePtr dynamic_node_, stadium_node_, camera_node_, sun_node_, ball_node_;
    ObjectPtr ball_, camera_, sun_, yellow_, red_;
    std::array<TeamNodes, 2> teams_;
    std::vector<Body> bodies_;
    Radar *radar_ = nullptr;
    Scoreboard *scoreboard_ = nullptr;
    CaptionView *message_ = nullptr;
    bool exited_ = false;
};

// GRF's GameContext as far as graphics go, one per renderer: the scene, the GUI, and what GRF keeps
// across matches: the fullbody template, the stadium, the goals, the animation skeleton, the
// procedural pitch and the counters that name copies.
class Context {
public:
    struct Shared {
        std::string data_dir;
        const overlay2d::GlyphAtlas *atlas = nullptr;
        std::function<const pitch::PitchTextures &()> pitch;
        std::function<bool(const std::string &path, assets::Surface *)> load_surface;
        std::function<bool(const std::string &path, assets::GeometryData *)> load_ase;
    };
    Context(GlStream &gl, int width, int height, const Shared &shared);
    ~Context();

    Scene &scene() { return *scene_; }
    Gui &gui() { return *gui_; }
    Match *match() { return match_.get(); }

    void CreateContext();
    void ResetBegin();
    void StopMatch();
    void CreateLoadingPage(const std::array<std::string, 2> &names,
                           const std::array<std::string, 2> &logos);
    void StartMatch(const MatchLook &look, const SkinFunction &skin);

    NodePtr LoadObject(const std::string &path);
    const Shared &shared() const { return shared_; }

    NodePtr fullbody_template, stadium, stadium_no_render, goals, anim_skeleton;
    int player_count = 0;
    std::uint64_t light_count = 0;
    bool pitch_loaded = false;

private:
    friend class Match;
    Shared shared_;
    std::unique_ptr<Scene> scene_;
    std::unique_ptr<Gui> gui_;
    std::unique_ptr<Match> match_;
};

}

#endif
