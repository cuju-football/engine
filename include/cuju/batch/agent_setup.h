// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "../simulation/environment.h"
#include "../simulation/state_io.h"
#include "../simulation/static_data.h"

namespace cuju {
namespace batch {
using state::MATCH_KEYBOARDS;

// GRF: a reset disables and releases every keyboard, then sets its mirror, -1 for a right agent.
inline state::KeyboardState ResetKeyboard(float mirror) {
    state::KeyboardState keyboard{};
    keyboard.mirror = mirror;
    keyboard.direction = Vector3(0, 0, 0) * mirror;
    keyboard.disabled = true;
    for (bool &button : keyboard.buttonsPressed) button = false;
    return keyboard;
}

// GRF: a newly built gamer without a player.
inline state::HumanGamerState NewHumanGamer(int slot, int keyboardIndex, float mirror) {
    state::HumanGamerState gamer{};
    gamer.selectedPlayer = -1;
    gamer.team = slot;
    gamer.keyboardIndex = keyboardIndex;
    state::PlayerControllerState &base = gamer.controller.base;
    base.player = -1;
    base.team = -1;
    base.opponentTeam = -1;
    base.opponentPlayer = -1;
    base.inputDirection = Vector3(0, -1, 0);
    base.inputVelocityFloat = 0;
    base.timeNeededMilliseconds = 0;
    base.mentalImageTime = 0;
    base.hasPossession = base.hasUniquePossession = base.teamHasPossession =
        base.teamHasUniquePossession = base.opponentTeamHasPossession =
            base.opponentTeamHasUniquePossession = base.hasBestPossession =
                base.teamHasBestPossession = base.hasBestChanceOfPossession = false;
    base.possessionAmount = 0.9f;
    base.teamPossessionAmount = 1.0f;
    base.opponentTimeNeededToGetToBall = 100;
    // GRF: `HumanController::Reset` overrides these values of `PlayerController::Reset`.
    base.lastSwitchTimeMilliseconds = (uint64_t)(int64_t)-10000;
    base.lastSwitchTimeDurationMilliseconds = 300;
    base.fadingTeamPossessionAmount = 1.0f;
    state::HumanControllerState &controller = gamer.controller;
    controller.keyboard = ResetKeyboard(mirror);
    controller.actionMode = 0;
    controller.actionButton = simulation::e_ButtonFunction_ShortPass;
    controller.actionBufferTimeMilliseconds = 0;
    controller.gaugeMilliseconds = 0;
    controller.previousDirection = Vector3(0, -1, 0);
    controller.steadyDirection = Vector3(0, -1, 0);
    controller.lastSteadyDirectionSnapshotTimeMilliseconds = 0;
    return gamer;
}

inline void StripAgents(simulation::World &world) {
    state::MatchState &match = world.serialized.match;
    world.serialized.scenario.leftAgents = 0;
    world.serialized.scenario.rightAgents = 0;
    for (state::TeamState &team : match.teams) {
        team.humanGamers.count = 0;
        team.switchPriority.count = 0;
        team.mainSelectedPlayer = -1;
    }
    for (int player = 0; player < match.playerCount; player++)
        match.players[player].base.externalController = -1;
    match.freeControllerCount = MATCH_KEYBOARDS;
    for (int keyboard = 0; keyboard < MATCH_KEYBOARDS; keyboard++)
        match.freeControllers[keyboard] = ResetKeyboard(1.0f);
}

// GRF: the observation reads each agent's controller from a list as long as its team, so a side
// holds 0 to its team size in agents.
inline void CheckAgents(const state::ScenarioState &scenario, int left, int right,
                        const std::string &bundle) {
    auto check = [&](const char *side, int agents, int players) {
        if (agents < 0 || agents > players)
            throw std::invalid_argument("expected from 0 to " + std::to_string(players) + " " +
                                        side + " agents, the players of the " + side + " team of " +
                                        bundle + ", found " + std::to_string(agents));
    };
    check("left", left, scenario.leftTeamSize);
    check("right", right, scenario.rightTeamSize);
    if (left + right > MATCH_KEYBOARDS)
        throw std::invalid_argument("expected at most " + std::to_string(MATCH_KEYBOARDS) +
                                    " agents in " + bundle + ", found " +
                                    std::to_string(left + right));
}

inline void SetupAgents(simulation::World &world, const simulation::StaticData &staticData,
                        int left, int right, const std::string &bundle) {
    state::ScenarioState &scenario = world.serialized.scenario;
    state::MatchState &match = world.serialized.match;
    CheckAgents(scenario, left, right, bundle);
    StripAgents(world);
    scenario.leftAgents = left;
    scenario.rightAgents = right;
    const int agents[2] = {left, right};
    for (int teamID = 0; teamID < 2; teamID++) {
        state::TeamState &team = simulation::TeamByID(world, teamID);
        team.humanGamers.count = agents[teamID];
        team.switchPriority.count = agents[teamID];
        for (int gamer = 0; gamer < agents[teamID]; gamer++) team.switchPriority[gamer] = gamer;
    }
    for (int teamID = 0; teamID < 2; teamID++) {
        // GRF: both teams still face side -1 at the controller setup, so the right keyboards
        // mirror.
        const float mirror =
            teamID == 1 && simulation::TeamDynamicSide(world, 1) == -1 ? -1.0f : 1.0f;
        for (int gamer = 0; gamer < agents[teamID]; gamer++) {
            const int gamerIndex = simulation::GlobalGamerIndex(world, teamID, gamer);
            state::HumanGamerState &human = simulation::GamerByGlobal(world, gamerIndex);
            human =
                NewHumanGamer(simulation::TeamSlot(world, teamID),
                              gamer + (teamID ? simulation::MAXIMUM_TEAM_CONTROLLERS : 0), mirror);
            match.secondKeyboardCopies[gamerIndex] = human.controller.keyboard;
        }
    }
    match.freeControllerCount = MATCH_KEYBOARDS - left - right;
    // GRF: `Team::AddHumanGamers` takes the active controllable players closest to the ball, which
    // is at the origin with the players at their formation spots, ties in state order.
    simulation::SimulationContext context{};
    context.staticData = &staticData;
    auto stableID = [&](int player) { return staticData.stableID[player]; };
    for (int teamID = 0; teamID < 2; teamID++) {
        if (agents[teamID] == 0) continue;
        const Vector3 ball = simulation::BallPredict(match.ball, 0).Get2D();
        simulation::PlayerList players = simulation::GetClosestPlayers(
            world, context, teamID, ball, true, (unsigned int)agents[teamID], true);
        if (players.empty()) continue;
        simulation::TeamByID(world, teamID).mainSelectedPlayer = players[0];
        std::sort(players.begin(), players.end(),
                  [&](int first, int second) { return stableID(first) < stableID(second); });
        for (int gamer = 0; gamer < (int)players.size(); gamer++)
            simulation::HumanGamerSetSelectedPlayer(
                world, simulation::GlobalGamerIndex(world, teamID, gamer), players[gamer]);
    }
}

}
}
