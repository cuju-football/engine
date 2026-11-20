// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0

// GRF's `SharedInfo`, which `GameEnv::get_info` returns, with the raw vectors that GRF's `Position`
// holds. `EnvironmentGetState` gives the same values in the coordinates of the environment.
#pragma once

#include "../simulation/environment.h"

namespace cuju {
namespace engine {

struct PlayerInfo {
    Vector3 position, direction;
    float tiredFactor;
    bool hasCard, isActive, designatedPlayer;
    int role;
};
struct TeamInfo {
    int count = 0;
    PlayerInfo players[state::MAXIMUM_TEAM_PLAYERS];
};
// `team[0]` is the left team. A controller holds the index of its player within the team, or -1.
struct EngineInfo {
    Vector3 ballPosition, ballDirection, ballRotation;
    TeamInfo team[2];
    int controlledPlayer[2][simulation::MAXIMUM_TEAM_CONTROLLERS];
    int gameMode;
    int score[2];
    int ballOwnedTeam, ballOwnedPlayer;
    int step;
    bool isInPlay;
};

// GRF: `Match::GetState` and `GameEnv::get_info`, line for line with `EnvironmentGetState`.
inline void EnvironmentGetInfo(const simulation::World &world,
                               const simulation::SimulationContext &context, EngineInfo &info) {
    using namespace simulation;
    const state::MatchState &match = world.serialized.match;
    info = EngineInfo();
    info.ballPosition = BallAveragePosition(world, 5);
    float rotationX, rotationY, rotationZ;
    match.ball.rotationPerMillisecond.GetAngles(rotationX, rotationY, rotationZ);
    info.ballRotation = Vector3(rotationX, rotationY, rotationZ) / (float)context.ticksPerStep;
    info.ballDirection = match.ball.momentum / (float)context.ticksPerStep;
    info.ballOwnedPlayer = -1;
    info.ballOwnedTeam = -1;
    info.score[0] = MatchGoalCount(world, 0);
    info.score[1] = MatchGoalCount(world, 1);
    // GRF: the step before the start at 2000 ms, which ends at 1900 ms, reports in play.
    info.isInPlay = match.inPlay || match.actualTimeMilliseconds == 1900;
    info.gameMode = match.inSetPiece ? match.referee.buffer.desiredSetPiece : e_GameMode_Normal;
    for (auto &teamControllers : info.controlledPlayer)
        for (int &controlled : teamControllers) controlled = -1;
    const int firstTeamID = world.FirstTeamID();
    for (int teamID : {firstTeamID, 1 - firstTeamID}) {
        TeamInfo &teamInfo = info.team[teamID];
        const state::TeamState &team = TeamByID(world, teamID);
        for (int player : TeamAllPlayers(world, teamID)) {
            const state::PlayerState &playerState = match.players[player];
            if (playerState.base.externalController >= 0) {
                int controller =
                    GamerByGlobal(const_cast<World &>(world), playerState.base.externalController)
                        .keyboardIndex;
                info.controlledPlayer[teamID][controller % MAXIMUM_TEAM_CONTROLLERS] =
                    teamInfo.count;
            }
            Vector3 position = playerState.base.humanoid.spatialState.position;
            Vector3 movement = playerState.base.humanoid.spatialState.movement;
            if (teamID == 1) {
                position.Mirror();
                movement.Mirror();
            }
            PlayerInfo &playerInfo = teamInfo.players[teamInfo.count];
            playerInfo.position = position;
            playerInfo.direction = movement / (float)context.ticksPerStep;
            playerInfo.tiredFactor = 1 - playerState.base.fatigueFactorInverse;
            playerInfo.hasCard = playerState.cards > 0;
            playerInfo.isActive = playerState.base.isActive;
            playerInfo.role = Formation(world, context, player).role;
            if (playerState.hasPossession && match.lastTouchTeamID != -1 &&
                TeamByID(world, MatchLastTouchTeam(world)).lastTouchPlayer == player) {
                info.ballOwnedPlayer = teamInfo.count;
                info.ballOwnedTeam = match.lastTouchTeamID;
            }
            playerInfo.designatedPlayer = player == team.mainSelectedPlayer;
            teamInfo.count++;
        }
    }
    info.step = world.serialized.step;
}

}  // namespace engine
}  // namespace cuju
