// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../possession.h"
#include "world.h"

namespace cuju {
namespace simulation {
CUJU_FUNCTION inline int TeamSlot(const World &world, int teamID) {
    return teamID == world.FirstTeamID() ? 0 : 1;
}
CUJU_FUNCTION inline state::TeamState &TeamByID(World &world, int teamID) {
    return world.serialized.match.teams[TeamSlot(world, teamID)];
}
CUJU_FUNCTION inline const state::TeamState &TeamByID(const World &world, int teamID) {
    return world.serialized.match.teams[TeamSlot(world, teamID)];
}
CUJU_FUNCTION inline const state::TeamState &TeamOfPlayer(const World &world, int player) {
    return TeamByID(world, world.playerTeamID[player]);
}
CUJU_FUNCTION inline const state::TeamState &OpponentTeamOfPlayer(const World &world, int player) {
    return TeamByID(world, 1 - world.playerTeamID[player]);
}
CUJU_FUNCTION inline HumanoidState &Humanoid(World &world, int player) {
    return world.serialized.match.players[player].base.humanoid;
}
CUJU_FUNCTION inline const HumanoidState &Humanoid(const World &world, int player) {
    return world.serialized.match.players[player].base.humanoid;
}
CUJU_FUNCTION inline Vector3 BallPredict(const state::BallSerialized &ball,
                                         int predictTimeMilliseconds) {
    Vector3 prediction = cuju::BallPredict(ball.predictions, predictTimeMilliseconds);
    if (ball.lazyFlip) prediction.Mirror();
    return prediction;
}

CUJU_FUNCTION inline int HumanGamerCount(const state::TeamState &team) {
    int count = 0;
    for (int gamer = 0; gamer < team.humanGamers.count; gamer++)
        if (!team.humanGamers[gamer].controller.keyboard.disabled) count++;
    return count;
}

CUJU_FUNCTION inline void InitializePlayerTeams(World &world) {
    int firstTeamID = world.FirstTeamID();
    int firstTeamSize = firstTeamID == 0 ? world.serialized.scenario.leftTeamSize
                                         : world.serialized.scenario.rightTeamSize;
    for (int player = 0; player < world.serialized.match.playerCount; player++)
        world.playerTeamID[player] = player < firstTeamSize ? firstTeamID : 1 - firstTeamID;
    for (int player = 0; player < state::MAXIMUM_PLAYERS; player++)
        world.formationOverride[player] = 0;
}

}  // namespace simulation
}  // namespace cuju
