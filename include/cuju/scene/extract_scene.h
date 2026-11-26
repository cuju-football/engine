// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// Reads the scene of a match from its world, changing nothing in it.
#pragma once

#include <cstring>

#include "../render/extract_render_state.h"
#include "../simulation/environment.h"
#include "scene_state.h"

namespace cuju {
namespace scene {
namespace detail {
CUJU_FUNCTION inline void PutJoints(const state::HumanoidState &humanoid,
                                    const animation::Skeleton &skeleton, bool mirror, bool mirrored,
                                    ScenePlayer &out,
                                    animation::Derived derived[state::HUMANOID_NODES]) {
    state::NodeState nodes[state::HUMANOID_NODES];
    for (int node = 0; node < state::HUMANOID_NODES; node++) nodes[node] = humanoid.nodes[node];
    if (mirror) render::detail::MirrorRoot(nodes[0], mirrored);
    animation::DeriveNodes(skeleton, nodes, derived);
    for (int axis = 0; axis < 3; axis++)
        out.rootPosition[axis] = derived[0].position.coordinates[axis];
    for (int axis = 0; axis < 4; axis++)
        out.rootRotation[axis] = derived[0].rotation.elements[axis];
    for (int joint = 0; joint < SCENE_JOINTS; joint++) {
        for (int axis = 0; axis < 3; axis++)
            out.joints[joint][axis] = derived[joint + 1].position.coordinates[axis];
        for (int axis = 0; axis < 4; axis++)
            out.jointRotations[joint][axis] = derived[joint + 1].rotation.elements[axis];
    }
    out.animation = humanoid.currentAnimation.functionType;
}

CUJU_FUNCTION inline int EntryOf(const int *entries, int count, int player) {
    return player >= 0 && player < count ? entries[player] : -1;
}
}  // namespace detail

CUJU_FUNCTION inline void ExtractScene(const simulation::World &world,
                                       const render::RenderContext &context, SceneState &out) {
    const state::MatchState &match = world.serialized.match;
    const simulation::SimulationContext &simulation = *context.simulation;
    const state::ScenarioState &scenario = world.serialized.scenario;
    const bool reverse = scenario.reverseTeamProcessing;
    memset(&out, 0, sizeof(SceneState));
    out.version = SCENE_STATE_VERSION;
    for (ScenePlayer &entry : out.players) entry.team = entry.slot = entry.role = -1;

    int entries[state::MAXIMUM_PLAYERS];
    int slots[2] = {0, 0};
    for (int player = 0; player < match.playerCount; player++) {
        const int team = world.playerTeamID[player];
        entries[player] = team * SCENE_TEAM_PLAYERS + slots[team]++;
    }

    out.step = world.serialized.step;
    out.stepsPerSecond = 100 / simulation.ticksPerStep;
    out.duration = scenario.gameDuration;
    out.matchMilliseconds = (int32_t)match.matchTimeMilliseconds;
    out.half = match.matchPhase;
    out.inPlay = match.inPlay;
    for (int team = 0; team < 2; team++) out.score[team] = simulation::MatchGoalCount(world, team);
    out.gameMode = match.inSetPiece ? match.referee.buffer.desiredSetPiece : 0;
    if (match.referee.buffer.active) {
        out.setPiece = match.referee.buffer.desiredSetPiece;
        for (int axis = 0; axis < 3; axis++)
            out.restartPosition[axis] = match.referee.buffer.restartPosition.coordinates[axis];
    }

    out.touchTeam = match.lastTouchTeamID;
    out.possessionTeam = out.possessionPlayer = out.touchPlayer = out.touchKind = -1;
    if (match.lastTouchTeamID != -1) {
        const int touched = simulation::TeamByID(world, match.lastTouchTeamID).lastTouchPlayer;
        out.touchPlayer = detail::EntryOf(entries, match.playerCount, touched);
        if (out.touchPlayer >= 0) out.touchKind = match.players[touched].base.lastTouchType;
    }
    const int ownerTeam = simulation::MatchLastTouchTeam(world);
    const int ownerCandidate = simulation::TeamByID(world, ownerTeam).lastTouchPlayer;
    if (match.lastTouchTeamID != -1 && ownerCandidate >= 0 && ownerCandidate < match.playerCount &&
        match.players[ownerCandidate].hasPossession) {
        out.possessionTeam = match.lastTouchTeamID;
        out.possessionPlayer = entries[ownerCandidate];
    }

    const state::FoulState &foul = match.referee.foul;
    out.foulKind = foul.foulType;
    out.foulOffender = out.foulVictim = -1;
    if (foul.foulType != 0) {
        out.foulOffender = detail::EntryOf(entries, match.playerCount, foul.foulPlayer);
        out.foulVictim = detail::EntryOf(entries, match.playerCount, foul.foulVictim);
    }
    out.offsideCalled = match.referee.buffer.active &&
                        match.referee.buffer.desiredSetPiece == simulation::e_GameMode_FreeKick &&
                        foul.foulType == 0 && match.lastTouchTeamID != -1;
    out.offsidePlayer =
        out.offsideCalled ? out.touchPlayer
        : match.referee.offsidePlayers.count > 0
            ? detail::EntryOf(entries, match.playerCount, match.referee.offsidePlayers[0])
            : -1;
    out.goalTeam = match.lastGoalTeam;
    out.goalScorer = detail::EntryOf(entries, match.playerCount, match.lastGoalScorer);

    for (int axis = 0; axis < 3; axis++)
        out.ballPosition[axis] = match.ball.positionBuffer.coordinates[axis];
    for (int axis = 0; axis < 4; axis++)
        out.ballRotation[axis] = match.ball.orientationBuffer.elements[axis];
    out.ballTouchesNet = match.ball.ballTouchesNet;

    const int64_t touchWindowStart =
        (int64_t)match.actualTimeMilliseconds - 10 * simulation.ticksPerStep;
    animation::Derived derived[state::HUMANOID_NODES];
    for (int player = 0; player < match.playerCount; player++) {
        const int team = world.playerTeamID[player];
        const state::PlayerState &state = match.players[player];
        ScenePlayer &entry = out.players[entries[player]];
        entry.team = team;
        entry.slot = entries[player] - team * SCENE_TEAM_PLAYERS;
        entry.present = state.base.isActive;
        entry.role = simulation::Formation(world, simulation, player).role;
        entry.yellow = state.cards % 3;
        entry.red = state.cards >= 2;
        entry.controlled = state.base.externalController >= 0;
        entry.designated = simulation::TeamByID(world, team).mainSelectedPlayer == player;
        entry.height = simulation.staticData->data[player].height;
        if ((int64_t)state.base.lastTouchTimeMilliseconds > touchWindowStart)
            entry.touch = state.base.lastTouchType + 1;
        detail::PutJoints(state.base.humanoid, simulation.skeletons[player], team == 1,
                          world.flags.humanoidMirrored[player], entry, derived);
    }
    for (int official = 0; official < SCENE_OFFICIALS; official++) {
        const int humanoid = match.playerCount + official;
        ScenePlayer &entry = out.players[2 * SCENE_TEAM_PLAYERS + official];
        entry.team = 2;
        entry.slot = official;
        entry.present = true;
        entry.height = simulation.staticData->data[humanoid].height;
        detail::PutJoints(match.officials[official].humanoid, simulation.skeletons[humanoid],
                          reverse, false, entry, derived);
        if (official != 0) continue;
        const state::HumanoidState &referee = match.officials[0].humanoid;
        if (referee.currentAnimation.functionType != simulation::e_FunctionType_Special ||
            (foul.foulType != 2 && foul.foulType != 3))
            continue;
        const int animationIndex = referee.currentAnimation.animation;
        const bool mirrorName = context.animationMirrored && animationIndex >= 0 &&
                                context.animationMirrored[animationIndex];
        const int part =
            mirrorName ? simulation::e_BodyPart_LeftElbow : simulation::e_BodyPart_RightElbow;
        const int node = simulation.skeletons[match.playerCount].nodeMap[part];
        if (node < 0) continue;
        const animation::Derived &elbow = derived[node];
        const Vector3 position = elbow.position + elbow.rotation * Vector3(0.04f, 0, -0.25f);
        out.cardKind = foul.foulType - 1;
        for (int axis = 0; axis < 3; axis++) out.cardPosition[axis] = position.coordinates[axis];
        for (int axis = 0; axis < 4; axis++) out.cardRotation[axis] = elbow.rotation.elements[axis];
    }
}

}  // namespace scene
}  // namespace cuju
