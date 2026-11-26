// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The scene of a match, what GRF's state says and a renderer needs, in 4-byte fields only so the
// layout is the same on every compiler. Python reads it as `cuju.scene.Frames`.
#pragma once

#include <cstdint>

#include "../state/environment_state.h"

namespace cuju {
namespace scene {
constexpr uint32_t SCENE_STATE_VERSION = 1;
// GRF's `joints`, the nodes below the humanoid's root in pre-order.
constexpr int SCENE_JOINTS = state::HUMANOID_NODES - 1;
constexpr int SCENE_TEAM_PLAYERS = state::MAXIMUM_TEAM_PLAYERS;
constexpr int SCENE_OFFICIALS = 3;
// The entries of the players: the left team, the right team and the officials, each team's room in
// full whatever its size.
constexpr int SCENE_ENTRIES = 2 * SCENE_TEAM_PLAYERS + SCENE_OFFICIALS;

struct ScenePlayer {
    // 0 left, 1 right, 2 an official, -1 for room that no player fills.
    int32_t team;
    // The index in the team, or of the official, -1 for room that no player fills.
    int32_t slot;
    // GRF's `isActive`, false for a player sent off.
    int32_t present;
    // GRF's `e_PlayerRole` of the formation, -1 for an official.
    int32_t role;
    // The yellow cards shown, and whether the player is sent off.
    int32_t yellow, red;
    int32_t controlled, designated;
    // GRF's `e_FunctionType` of the animation that the body runs.
    int32_t animation;
    // 1 plus GRF's `e_TouchType` when the player touched the ball within the last step, else 0.
    int32_t touch;
    float height;
    float rootPosition[3];
    float rootRotation[4];
    float joints[SCENE_JOINTS][3];
    float jointRotations[SCENE_JOINTS][4];
};

struct SceneState {
    uint32_t version;
    // The count of resets of the environment, so a change tells a new episode.
    int32_t episode;
    // GRF's step counter, which is -1 right after a reset and counts the steps in play.
    int32_t step;
    int32_t stepsPerSecond;
    // The steps of the match, GRF's `game_duration`.
    int32_t duration;
    int32_t matchMilliseconds;
    // 0 before the match, 1 the first half, 2 the second.
    int32_t half;
    int32_t inPlay;
    int32_t score[2];
    // GRF's `game_mode` of the observation, the set piece being played, 0 for normal play.
    int32_t gameMode;
    // The set piece that the referee has stopped play for, 0 when none.
    int32_t setPiece;
    float restartPosition[3];
    // The team and the entry of the player that owns the ball, -1 when none.
    int32_t possessionTeam, possessionPlayer;
    int32_t touchTeam, touchPlayer, touchKind;
    int32_t offsidePlayer, offsideCalled;
    // 0 none, 1 a foul, 2 a yellow card, 3 a red card.
    int32_t foulKind, foulOffender, foulVictim;
    int32_t goalTeam, goalScorer;
    float ballPosition[3];
    float ballRotation[4];
    int32_t ballTouchesNet;
    // The card in the referee's hand, 0 none, 1 yellow, 2 red.
    int32_t cardKind;
    float cardPosition[3];
    float cardRotation[4];
    ScenePlayer players[SCENE_ENTRIES];
};

static_assert(sizeof(ScenePlayer) % 4 == 0 && sizeof(SceneState) % 4 == 0,
              "the scene holds 4-byte fields only");
}  // namespace scene
}  // namespace cuju
