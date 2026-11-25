// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../quaternion.h"
#include "../state/environment_state.h"
#include "../vector3.h"

namespace cuju {
namespace render {
constexpr uint32_t RENDER_STATE_VERSION = 1;
// GRF's `joints`, the nodes below the humanoid's root in `GetNodes` pre-order.
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

struct RenderJoint {
    Quaternion orientation;
    Vector3 position;
};

struct RenderHumanoidStatic {
    float zMultiplier;
    int8_t skin;
    int8_t hairStyle;
    int8_t hairColor;
    int8_t kit;
    // GRF's player database slot of the team, 0 to 10, for the names, or -1 for an official.
    int8_t dataIndex;
    uint8_t padding[3];
};

struct RenderHumanoid {
    bool present;
    bool mirrored;
    bool nameCaptionShow;
    bool hidden;
    Vector3 fullbodyOffset;
    RenderJoint joints[RENDER_JOINTS];
    Vector3 hairPosition;
    Quaternion hairRotation;
    Vector3 nameCaptionColor;
    Vector3 geomPosition;
    Vector3 radarPosition;
};

struct RenderCamera {
    Quaternion orientation;
    Quaternion nodeOrientation;
    Vector3 nodePosition;
    float fieldOfView;
    float nearCap;
    float farCap;
};

struct RenderBall {
    Vector3 position;
    Quaternion orientation;
    bool touchesNet;
    uint8_t padding[3];
    Vector3 radarPosition;
};

struct RenderCard {
    Vector3 position;
    Quaternion rotation;
};

struct RenderOverlay {
    uint64_t matchTimeMilliseconds;
    uint64_t actualTimeMilliseconds;
    char clock[RENDER_CLOCK_BYTES];
    int32_t goals[2];
    bool messageVisible;
    uint8_t padding[3];
    char message[RENDER_MESSAGE_BYTES];
};

struct RenderEpisode {
    Vector3 sunPosition;
    Vector3 sunColor;
    int32_t teamSize[2];
    bool reverseTeamProcessing;
    uint8_t padding[3];
    RenderHumanoidStatic players[2][RENDER_TEAM_PLAYERS];
    RenderHumanoidStatic officials[RENDER_OFFICIALS];
};

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
    int32_t lastWorldGoals[2];
};

}
}
