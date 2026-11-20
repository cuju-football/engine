// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// What GRF's graphics side reads at each render, as plain data. Ported from GRF's Match::Put,
// Team::Put, HumanoidBase::UpdateFullbodyNodes, Officials::Put, Ball::Put, Player::Put2D,
// Gui2Radar::Put and the scoreboard, GRF is Apache-2.0.
// Frames: the camera, the ball, the joints, the hair and the cards are in GRF's drawn frame, the
// one of team 0. The name tag and radar positions are in each team's own frame, as GRF reads them.
#pragma once

#include <cstdint>

#include "../quaternion.h"
#include "../state/environment_state.h"
#include "../vector3.h"

namespace cuju {
namespace render {

constexpr uint32_t RENDER_STATE_VERSION = 1;
// GRF's `joints`, the nodes below the humanoid's root in `GetNodes` pre-order. Joint j is
// skeleton node j + 1.
constexpr int RENDER_JOINTS = state::HUMANOID_NODES - 1;
constexpr int RENDER_TEAM_PLAYERS = state::MAXIMUM_TEAM_PLAYERS;
constexpr int RENDER_OFFICIALS = 3;
constexpr int RENDER_MESSAGE_BYTES = 128;
constexpr int RENDER_CLOCK_BYTES = 16;

enum RenderKit : int8_t {
    e_RenderKit_Team = 0,
    e_RenderKit_Goalkeeper = 1,
    e_RenderKit_Referee = 2
};
// Hair styles and colours in GRF's file names.
enum RenderHairStyle : int8_t {
    e_RenderHair_Bald,
    e_RenderHair_Short01,
    e_RenderHair_Short02,
    e_RenderHair_Medium01,
    e_RenderHair_Medium02,
    e_RenderHair_Long01,
    e_RenderHair_Long02
};
enum RenderHairColor : int8_t {
    e_RenderHairColor_Black,
    e_RenderHairColor_Blonde,
    e_RenderHairColor_DarkBlonde
};

// One entry of GRF's `joints` after `UpdateFullbodyNodes`: the derived rotation, and the derived
// position minus the full body offset, both unscaled.
struct RenderJoint {
    Quaternion orientation;
    Vector3 position;
};

// What does not change within a match, set when the match is created.
struct RenderHumanoidStatic {
    // `(1.0f / 1.92f) * height`, which scales the mesh and the joints, never the hair.
    float zMultiplier;
    int8_t skin;  // 1 to 4, `skin0N.png`
    int8_t hairStyle;
    int8_t hairColor;
    int8_t kit;
    // GRF's player database slot of the team, 0 to 10, for the names, or -1 for an official.
    int8_t dataIndex;
    uint8_t padding[3];
};

struct RenderHumanoid {
    // An active player or an official. GRF skins and uploads only these.
    bool present;
    // The `mirror` argument of `Put`: the humanoid was mirrored into the drawn frame.
    bool mirrored;
    bool nameCaptionShow;  // `buf_nameCaptionShowCondition`, players only
    // An inactive player whose mesh and hair `Hide` moved to (1000, 1000, -1000) when it was sent
    // off. Any other inactive player, such as one a load made inactive, keeps the mesh and the
    // place of its last render, or of the match's creation.
    bool hidden;
    // The full body node's position, the root's position with z set to 0.
    Vector3 fullbodyOffset;
    RenderJoint joints[RENDER_JOINTS];
    // The hair geometry's absolute transform, `joints[2]` placed as GRF does.
    Vector3 hairPosition;
    Quaternion hairRotation;
    // `buf_playerColor`, 0 to 255 per channel. The tag's outline is this times 0.4f.
    Vector3 nameCaptionColor;
    // The root node's position after `Put`, in the team's own frame, which `Put2D` reads.
    Vector3 geomPosition;
    // `GetPosition`, the simulation position in the team's own frame, which the radar reads.
    Vector3 radarPosition;
};

// As `Match::Put` passes the camera: the camera's own rotation, its node's rotation and position,
// the field of view in degrees and the clip distances.
struct RenderCamera {
    Quaternion orientation;
    Quaternion nodeOrientation;
    Vector3 nodePosition;
    float fieldOfView;
    float nearCap;
    float farCap;
};

struct RenderBall {
    Vector3 position;        // `positionBuffer`, which the net also reads
    Quaternion orientation;  // `orientationBuffer`
    bool touchesNet;
    uint8_t padding[3];
    // `Predict(0)` with z kept, for the radar, which drops z.
    Vector3 radarPosition;
};

// A card's absolute transform. GRF moves the cards only while the referee shows one, and parks
// them at (0, 0, -10) when that animation ends, so a card keeps its transform between.
struct RenderCard {
    Vector3 position;
    Quaternion rotation;
};

struct RenderOverlay {
    uint64_t matchTimeMilliseconds;
    uint64_t actualTimeMilliseconds;
    // `MM:SS` as `Match::Put` formats it, zero terminated.
    char clock[RENDER_CLOCK_BYTES];
    // What the scoreboard shows per team, which changes only at a goal.
    int32_t goals[2];
    bool messageVisible;
    uint8_t padding[3];
    // The message caption's text before uppercasing, zero terminated.
    char message[RENDER_MESSAGE_BYTES];
};

// Fixed for a match: the sun of `SetRandomSunParams` and the humanoids' looks.
struct RenderEpisode {
    // The light's position, `sunPos * 10000.0f`, and colour.
    Vector3 sunPosition;
    Vector3 sunColor;
    int32_t teamSize[2];
    // `reverse_team_processing`: the second team is processed first.
    bool reverseTeamProcessing;
    uint8_t padding[3];
    RenderHumanoidStatic players[2][RENDER_TEAM_PLAYERS];
    RenderHumanoidStatic officials[RENDER_OFFICIALS];
};

// What one render reads. Players are indexed by GRF team, 0 the left team, and by their index in
// the team's lineup. Officials are the referee, then the two linesmen.
struct RenderState {
    uint32_t version;
    // GRF's `first_team`, the team whose players `PrepareRender` uploads first.
    int32_t firstTeam;
    RenderEpisode episode;
    RenderCamera camera;
    RenderBall ball;
    RenderHumanoid players[2][RENDER_TEAM_PLAYERS];
    RenderHumanoid officials[RENDER_OFFICIALS];
    RenderCard yellowCard;
    RenderCard redCard;
    RenderOverlay overlay;
    // The match's goal counts at this render, in the overlay's order. The scoreboard reads the
    // engine's own record of the goals instead, which a load leaves as it was.
    int32_t lastWorldGoals[2];
};

}  // namespace render
}  // namespace cuju
