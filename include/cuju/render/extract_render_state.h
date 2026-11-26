// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../animation/skeleton.h"
#include "../libm/powf.h"
#include "../math_utilities.h"
#include "../simulation/access.h"
#include "../simulation/humanoid.h"
#include "render_state.h"

namespace cuju {
namespace render {
struct RenderMessage {
    bool spammed = false;
    uint64_t removeTimeMilliseconds = 0;
    char text[RENDER_MESSAGE_BYTES] = {};
};

struct RenderContext {
    const simulation::SimulationContext *simulation = nullptr;
    const uint8_t *animationMirrored = nullptr;
    const RenderMessage *message = nullptr;
};

namespace detail {
constexpr float DEFAULT_PLAYER_HEIGHT = 1.92f;

// GRF's player database, slots 0 to 10 of either team: hair style and colour.
CUJU_FUNCTION inline void HairOfSlot(int slot, int8_t &style, int8_t &color) {
    const int8_t styles[11] = {e_RenderHair_Long02,   e_RenderHair_Short02, e_RenderHair_Long02,
                               e_RenderHair_Medium01, e_RenderHair_Short01, e_RenderHair_Long02,
                               e_RenderHair_Long02,   e_RenderHair_Long01,  e_RenderHair_Medium01,
                               e_RenderHair_Long02,   e_RenderHair_Short02};
    const int8_t colors[11] = {
        e_RenderHairColor_Blonde, e_RenderHairColor_Black,  e_RenderHairColor_Black,
        e_RenderHairColor_Black,  e_RenderHairColor_Black,  e_RenderHairColor_Black,
        e_RenderHairColor_Black,  e_RenderHairColor_Blonde, e_RenderHairColor_Black,
        e_RenderHairColor_Black,  e_RenderHairColor_Black};
    if (slot >= 0 && slot < 11) {
        style = styles[slot];
        color = colors[slot];
    } else {
        style = e_RenderHair_Short01;
        color = e_RenderHairColor_DarkBlonde;
    }
}

CUJU_FUNCTION inline void MirrorRoot(state::NodeState &root, bool mirrored) {
    root.position = root.position * Vector3(-1, -1, 1);
    const Quaternion rotation = root.rotation;
    if (!mirrored)
        root.rotation = Quaternion(rotation.elements[1], rotation.elements[0], rotation.elements[3],
                                   -rotation.elements[2]);
    else
        root.rotation = Quaternion(rotation.elements[1], rotation.elements[0],
                                   -rotation.elements[3], rotation.elements[2]);
}

CUJU_FUNCTION inline void PutHumanoid(const simulation::World &world, int humanoidIndex,
                                      const state::HumanoidState &humanoid,
                                      const animation::Skeleton &skeleton, bool mirror,
                                      bool mirrored, float zMultiplier, RenderHumanoid &out,
                                      animation::Derived derived[state::HUMANOID_NODES]) {
    state::NodeState nodes[state::HUMANOID_NODES];
    for (int node = 0; node < state::HUMANOID_NODES; node++) nodes[node] = humanoid.nodes[node];
    if (mirror) MirrorRoot(nodes[0], mirrored);
    const state::NodeState root =
        mirror ? nodes[0] : simulation::RenderRoot(world, humanoidIndex, nodes[0]);
    animation::DeriveNodesWithRoot(skeleton, nodes, root, derived);
    const Vector3 &rootPosition = nodes[0].position;
    out.fullbodyOffset = Vector3(rootPosition.coordinates[0], rootPosition.coordinates[1], 0);
    for (int joint = 0; joint < RENDER_JOINTS; joint++) {
        out.joints[joint].orientation = derived[joint + 1].rotation;
        out.joints[joint].position = derived[joint + 1].position - out.fullbodyOffset;
    }
    out.hairRotation = out.joints[2].orientation;
    out.hairPosition = out.joints[2].position * zMultiplier + out.fullbodyOffset;
    out.mirrored = mirror;
    out.geomPosition = humanoid.nodes[0].position;
    out.radarPosition = humanoid.spatialState.position;
}

CUJU_FUNCTION inline void HideHumanoid(RenderHumanoid &out, bool hidden) {
    out.present = false;
    out.hidden = hidden;
    out.mirrored = false;
    out.nameCaptionShow = false;
    if (!hidden) return;
    out.fullbodyOffset = Vector3(1000, 1000, -1000);
    out.hairPosition = Vector3(1000, 1000, -1000);
}

CUJU_FUNCTION inline int WriteNumber(char *text, int at, int value) {
    char digits[12];
    int count = 0;
    do {
        digits[count++] = (char)('0' + value % 10);
        value /= 10;
    } while (value > 0 && count < 11);
    while (count > 0) text[at++] = digits[--count];
    return at;
}

}  // namespace detail

// Match::SetRandomSunParams from its six draws in the order of the RNG: two in -1.7 to 1.7, one in
// 0 to 1, three in -0.1 to 0.1.
CUJU_FUNCTION inline void SunFromDraws(const float draws[6], Vector3 &position, Vector3 &color) {
    Vector3 sun(clamp(draws[1], -1.0f, 1.0f), clamp(draws[0], -1.0f, 1.0f), 1.3f);
    sun.Normalize();
    if (draws[2] > 0.5f && sun.coordinates[1] > 0.25f) sun.coordinates[1] = -sun.coordinates[1];
    position = sun * 10000.0f;
    Vector3 noon(0.9f, 0.8f, 1.0f);
    noon *= 1.4f;
    Vector3 dusk(1.4f, 0.9f, 0.7f);
    dusk *= 1.2f;
    const float noonBias = libm::powf(NormalizedClamp(sun.coordinates[2], 0.5f, 1.0f), 1.2f);
    color = noon * noonBias + dusk * (1.0f - noonBias);
    Vector3 addition(draws[5], draws[4], draws[3]);
    addition *= 1.2f;
    color += addition;
}

struct RenderMatchDraws {
    float officialSkin = 1.0f;
    float light[6] = {};
};

CUJU_FUNCTION inline void ResetRenderState(const simulation::World &world,
                                           const RenderContext &context,
                                           const RenderMatchDraws &draws, RenderState &out) {
    const state::MatchState &match = world.serialized.match;
    const simulation::StaticData &staticData = *context.simulation->staticData;
    out.version = RENDER_STATE_VERSION;
    out.firstTeam = world.FirstTeamID();
    RenderEpisode &episode = out.episode;
    SunFromDraws(draws.light, episode.sunPosition, episode.sunColor);
    episode.reverseTeamProcessing = world.serialized.scenario.reverseTeamProcessing;
    episode.teamSize[0] = episode.teamSize[1] = 0;
    for (int player = 0; player < match.playerCount; player++) {
        const int team = world.playerTeamID[player];
        const int slot = episode.teamSize[team]++;
        RenderHumanoidStatic &entry = episode.players[team][slot];
        entry.zMultiplier = (1.0f / detail::DEFAULT_PLAYER_HEIGHT) * staticData.data[player].height;
        entry.skin = (int8_t)staticData.data[player].skinColor;
        detail::HairOfSlot(slot, entry.hairStyle, entry.hairColor);
        entry.kit = staticData.formation[player].role == simulation::e_PlayerRole_Goalkeeper
                        ? e_RenderKit_Goalkeeper
                        : e_RenderKit_Team;
        entry.dataIndex = (int8_t)slot;
    }
    for (int official = 0; official < RENDER_OFFICIALS; official++) {
        RenderHumanoidStatic &entry = episode.officials[official];
        const simulation::PlayerData &data = staticData.data[match.playerCount + official];
        entry.zMultiplier = (1.0f / detail::DEFAULT_PLAYER_HEIGHT) * data.height;
        // GRF: one PlayerData for the three, `int(std::round(boostrandom(1, 4)))`.
        entry.skin = (int8_t)(int)roundf(draws.officialSkin);
        detail::HairOfSlot(-1, entry.hairStyle, entry.hairColor);
        entry.kit = e_RenderKit_Referee;
        entry.dataIndex = -1;
    }
    out.yellowCard.position = Vector3(0, 0, -10);
    // GRF: the cards' nodes start with the rotation (0, 0, -0, 1).
    out.yellowCard.rotation = Quaternion(0, 0, -0.0f, 1);
    out.redCard = out.yellowCard;
    for (int team = 0; team < 2; team++) {
        out.overlay.goals[team] = 0;
        out.lastWorldGoals[team] = match.goalCount[team ^ out.firstTeam];
    }
}

CUJU_FUNCTION inline void ExtractRenderState(const simulation::World &world,
                                             const RenderContext &context, RenderState &out) {
    const state::MatchState &match = world.serialized.match;
    const simulation::SimulationContext &simulation = *context.simulation;
    const bool reverse = world.serialized.scenario.reverseTeamProcessing;
    out.version = RENDER_STATE_VERSION;
    out.firstTeam = world.FirstTeamID();

    out.ball.position = match.ball.positionBuffer;
    out.ball.orientation = match.ball.orientationBuffer;
    out.ball.touchesNet = match.ball.ballTouchesNet;
    out.ball.radarPosition = simulation::BallPredict(match.ball, 0);

    animation::Derived derived[state::HUMANOID_NODES];
    int slots[2] = {0, 0};
    for (int player = 0; player < match.playerCount; player++) {
        const int team = world.playerTeamID[player];
        RenderHumanoid &humanoid = out.players[team][slots[team]];
        const RenderHumanoidStatic &fixed = out.episode.players[team][slots[team]];
        slots[team]++;
        const state::PlayerState &state = match.players[player];
        humanoid.nameCaptionShow = state.bufferNameCaptionShowCondition;
        humanoid.nameCaptionColor = state.bufferPlayerColor;
        if (!state.base.isActive) {
            detail::HideHumanoid(humanoid, world.humanoidHidden[player]);
            continue;
        }
        humanoid.present = true;
        humanoid.hidden = false;
        detail::PutHumanoid(world, player, state.base.humanoid, simulation.skeletons[player],
                            team == 1, world.flags.humanoidMirrored[player], fixed.zMultiplier,
                            humanoid, derived);
    }
    for (int team = 0; team < 2; team++)
        for (int slot = slots[team]; slot < RENDER_TEAM_PLAYERS; slot++)
            out.players[team][slot].present = false;

    for (int official = 0; official < RENDER_OFFICIALS; official++) {
        RenderHumanoid &humanoid = out.officials[official];
        humanoid.present = true;
        humanoid.hidden = false;
        humanoid.nameCaptionShow = false;
        humanoid.nameCaptionColor = Vector3(0);
        detail::PutHumanoid(world, match.playerCount + official, match.officials[official].humanoid,
                            simulation.skeletons[match.playerCount + official], reverse, false,
                            out.episode.officials[official].zMultiplier, humanoid, derived);
        if (official == 0) {
            const state::HumanoidState &referee = match.officials[0].humanoid;
            const int foulType = match.referee.foul.foulType;
            if (referee.currentAnimation.functionType == simulation::e_FunctionType_Special &&
                (foulType == 2 || foulType == 3)) {
                const int animationIndex = referee.currentAnimation.animation;
                const bool mirrorName = context.animationMirrored && animationIndex >= 0 &&
                                        context.animationMirrored[animationIndex];
                const int part = mirrorName ? simulation::e_BodyPart_LeftElbow
                                            : simulation::e_BodyPart_RightElbow;
                const int node = simulation.skeletons[match.playerCount].nodeMap[part];
                if (node >= 0) {
                    const animation::Derived &elbow = derived[node];
                    const Vector3 position =
                        elbow.position + elbow.rotation * Vector3(0.04f, 0, -0.25f);
                    RenderCard &card = foulType == 2 ? out.yellowCard : out.redCard;
                    card.position = position;
                    card.rotation = elbow.rotation;
                }
            } else if (referee.previousAnimationFunctionType ==
                       simulation::e_FunctionType_Special) {
                out.yellowCard.position = Vector3(0, 0, -10);
                out.redCard.position = Vector3(0, 0, -10);
            }
        }
    }

    out.camera.orientation = match.cameraOrientation;
    out.camera.nodeOrientation = match.cameraNodeOrientation;
    out.camera.nodePosition = match.cameraNodePosition;
    out.camera.fieldOfView = match.cameraFieldOfView;
    out.camera.nearCap = match.cameraNearCap;
    out.camera.farCap = match.cameraFarCap;

    RenderOverlay &overlay = out.overlay;
    overlay.matchTimeMilliseconds = match.matchTimeMilliseconds;
    overlay.actualTimeMilliseconds = match.actualTimeMilliseconds;
    const int seconds = (int)(match.matchTimeMilliseconds / 1000.0) % 60;
    const int minutes = (int)(match.matchTimeMilliseconds / 60000.0);
    int at = 0;
    if (minutes < 10) overlay.clock[at++] = '0';
    at = detail::WriteNumber(overlay.clock, at, minutes);
    overlay.clock[at++] = ':';
    if (seconds < 10) overlay.clock[at++] = '0';
    at = detail::WriteNumber(overlay.clock, at, seconds);
    overlay.clock[at] = 0;
    for (int team = 0; team < 2; team++) {
        overlay.goals[team] = world.scoreboardGoals[team ^ out.firstTeam];
        out.lastWorldGoals[team] = match.goalCount[team ^ out.firstTeam];
    }
    const RenderMessage *message = context.message;
    overlay.messageVisible = message && message->spammed &&
                             !(message->removeTimeMilliseconds <= match.actualTimeMilliseconds);
    int length = 0;
    if (message && message->spammed)
        for (; length < RENDER_MESSAGE_BYTES - 1 && message->text[length]; length++)
            overlay.message[length] = message->text[length];
    overlay.message[length] = 0;
}

}  // namespace render
}  // namespace cuju
