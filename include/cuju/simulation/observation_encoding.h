// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../libm/pow.h"
#include "environment.h"

namespace cuju {
namespace simulation {
struct AgentView {
    Observation observation;
    int active;
    int designated;
    int stickyActionCount;
    uint8_t stickyActions[MAXIMUM_STICKY_ACTIONS];
};

CUJU_FUNCTION inline void StickyActions(World &world, int actionSet, bool leftTeam, int agent,
                                        uint8_t *out) {
    const int count = StickyActionCount(actionSet);
    for (int index = 0; index < count; index++)
        out[index] = StickyActionState(world, StickyAction(index), leftTeam, agent);
}

CUJU_FUNCTION inline void SwapTeams(const Observation &source, Observation &swapped) {
    swapped = source;
    swapped.ballOwnedTeam = source.ballOwnedTeam > -1 ? 1 - source.ballOwnedTeam : -1;
    swapped.score[0] = source.score[1];
    swapped.score[1] = source.score[0];
    for (int team = 0; team < 2; team++) {
        swapped.team[team] = source.team[1 - team];
        for (int controller = 0; controller < MAXIMUM_TEAM_CONTROLLERS; controller++)
            swapped.controlledPlayer[team][controller] =
                source.controlledPlayer[1 - team][controller];
    }
}

CUJU_FUNCTION inline void NegateGeometry(Observation &observation, int negate) {
    for (int coordinate = 0; coordinate < 2; coordinate++) {
        if (!(negate & (1 << coordinate))) continue;
        observation.ball[coordinate] = -observation.ball[coordinate];
        observation.ballDirection[coordinate] = -observation.ballDirection[coordinate];
    }
    for (int team = 0; team < 2; team++)
        for (int player = 0; player < observation.team[team].count; player++)
            for (int coordinate = 0; coordinate < 2; coordinate++) {
                if (!(negate & (1 << coordinate))) continue;
                observation.team[team].players[player].position[coordinate] =
                    -observation.team[team].players[player].position[coordinate];
                observation.team[team].players[player].direction[coordinate] =
                    -observation.team[team].players[player].direction[coordinate];
            }
}

CUJU_FUNCTION inline void MakeAgentView(const Observation &raw, bool right, int negate, int agent,
                                        int actionSet, const uint8_t *stickyActions,
                                        AgentView &view) {
    if (right)
        SwapTeams(raw, view.observation);
    else
        view.observation = raw;
    if (negate) NegateGeometry(view.observation, negate);
    view.designated = view.observation.team[0].designatedPlayer;
    view.active = view.observation.controlledPlayer[0][agent];
    view.stickyActionCount = StickyActionCount(actionSet);
    for (int index = 0; index < view.stickyActionCount; index++) {
        int source = index;
        if (negate) {
            const int flippedAction = NegateActionAxes(
                StickyAction(index), (negate & NEGATE_X) != 0, (negate & NEGATE_Y) != 0);
            for (int other = 0; other < view.stickyActionCount; other++)
                if (StickyAction(other) == flippedAction) source = other;
        }
        view.stickyActions[index] = stickyActions[source];
    }
}

CUJU_FUNCTION inline void EncodeSimple115(const AgentView &view, bool fixedPositions, float *out) {
    int count = 0;
    auto teamValues = [&](int team, bool directions, float *destination) {
        int written = 0;
        for (int player = 0; player < view.observation.team[team].count; player++)
            for (int coordinate = 0; coordinate < 2; coordinate++)
                destination[written++] =
                    directions ? view.observation.team[team].players[player].direction[coordinate]
                               : view.observation.team[team].players[player].position[coordinate];
        return written;
    };
    for (int part = 0; part < 4; part++) {
        count += teamValues(part / 2, part % 2 == 1, out + count);
        if (fixedPositions)
            while (count < (part + 1) * 22) out[count++] = -1;
    }
    while (count < 88) out[count++] = -1;
    for (int index = 0; index < 3; index++) out[count++] = view.observation.ball[index];
    for (int index = 0; index < 3; index++) out[count++] = view.observation.ballDirection[index];
    for (int index = 0; index < 3; index++)
        out[count++] = (view.observation.ballOwnedTeam + 1 == index) ? 1.0f : 0.0f;
    for (int index = 0; index < 11; index++)
        out[count++] = (view.active != -1 && view.active == index) ? 1.0f : 0.0f;
    for (int index = 0; index < 7; index++)
        out[count++] = (view.observation.gameMode == index) ? 1.0f : 0.0f;
}

// GRF's default SMM `channel_dimensions`.
constexpr int SMM_WIDTH = 96, SMM_HEIGHT = 72, SMM_LAYERS = 4;
constexpr int SMM_MARKS = 2 * state::MAXIMUM_TEAM_PLAYERS + 2;
CUJU_FUNCTION inline int SmmIndex(int layer, double positionX, double positionY, int width,
                                  int height) {
    const double minimumX = -1.0, maximumX = 1.0, minimumY = -1.0 / 2.25, maximumY = 1.0 / 2.25;
    int column = (int)((positionX - minimumX) / (maximumX - minimumX) * width);
    int row = (int)((positionY - minimumY) / (maximumY - minimumY) * height);
    column = column < 0 ? 0 : (column > width - 1 ? width - 1 : column);
    row = row < 0 ? 0 : (row > height - 1 ? height - 1 : row);
    return (row * width + column) * SMM_LAYERS + layer;
}
CUJU_FUNCTION inline int SmmMarks(const AgentView &view, int *marks, int width, int height) {
    int count = 0;
    for (int team = 0; team < 2; team++)
        for (int player = 0; player < view.observation.team[team].count; player++)
            marks[count++] =
                SmmIndex(team, view.observation.team[team].players[player].position[0],
                         view.observation.team[team].players[player].position[1], width, height);
    marks[count++] = SmmIndex(2, view.observation.ball[0], view.observation.ball[1], width, height);
    if (view.active != -1)
        marks[count++] =
            SmmIndex(3, view.observation.team[0].players[view.active].position[0],
                     view.observation.team[0].players[view.active].position[1], width, height);
    return count;
}

CUJU_FUNCTION inline float AddFloat(float reward, double increment, bool numpyLegacy) {
    return numpyLegacy ? (float)((double)reward + increment) : reward + (float)increment;
}
// GRF's `checkpoints` reward on top of the scoring `reward`.
CUJU_FUNCTION inline float CheckpointReward(const AgentView &view, float reward, int &collected,
                                            bool numpyLegacy) {
    const int checkpointCount = 10;
    const double checkpointReward = 0.1;
    if (reward == 1) {
        reward = AddFloat(reward, checkpointReward * (checkpointCount - collected), numpyLegacy);
        collected = checkpointCount;
        return reward;
    }
    if (view.observation.ballOwnedTeam != 0 || view.observation.ballOwnedPlayer != view.active)
        return reward;
    const float ballX = view.observation.ball[0];
    const float ballY = view.observation.ball[1];
    const double distance =
        libm::pow(libm::pow((double)ballX - 1, 2.0) + libm::pow((double)ballY, 2.0), 0.5);
    while (collected < checkpointCount) {
        const double threshold = 0.99 - 0.8 / (checkpointCount - 1) * collected;
        if (distance > threshold) break;
        reward = AddFloat(reward, checkpointReward, numpyLegacy);
        collected++;
    }
    return reward;
}

}  // namespace simulation
}  // namespace cuju
