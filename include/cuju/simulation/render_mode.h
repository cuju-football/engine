// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// GRF's render mode on the match. Ported from GRF's spatial.cpp, geometry.cpp, node.cpp,
// humanoidbase.cpp, match.cpp, referee.cpp and player.cpp, GRF is Apache-2.0.
// GRF: with the graphics system, every geometry of a humanoid observes its node, and an
// invalidation of a node computes the derived transforms of the node and its ancestors at once.
// The root keeps its cache until the next invalidation, which `ResetPosition` skips, so a read
// after it derives the nodes from the root where it was. A state's load or save marks every
// cache to compute afresh. Off, every read derives from the current root, as without graphics.
#pragma once

#include "../animation/skeleton.h"
#include "world.h"

namespace cuju {
namespace simulation {

// GRF: `Node::ProcessState`, which `get_state` and `set_state` run.
CUJU_FUNCTION inline void InvalidateRootCaches(World &world) {
    for (int humanoid = 0; humanoid < MAXIMUM_HUMANOIDS; humanoid++)
        world.rootCaches[humanoid].valid = false;
}

// GRF: `GameEnv::ProcessState` of a save or a load.
CUJU_FUNCTION inline void ProcessStateMarks(World &world) {
    InvalidateRootCaches(world);
    world.stateProcessed++;
}

// GRF: a read of the root computes a cache that is not valid. A valid one stays.
CUJU_FUNCTION inline void ReadRootCache(World &world, int humanoidIndex,
                                        const state::NodeState &root) {
    if (world.renderMode == e_Render_On && !world.rootCaches[humanoidIndex].valid)
        RefreshRootCache(world.rootCaches[humanoidIndex], root);
}

// The nodes as a read sees them. `humanoidIndex` counts the players, then the officials.
// Exact: the root's derived transform is its own, so only the root's values change.
CUJU_FUNCTION inline void DeriveHumanoidNodes(const World &world, int humanoidIndex,
                                              const animation::Skeleton &skeleton,
                                              const state::NodeState *nodes,
                                              animation::Derived derived[state::HUMANOID_NODES]) {
    const RootCache &cache = world.rootCaches[humanoidIndex];
    if (!world.renderMode || !cache.valid) {
        animation::DeriveNodes(skeleton, nodes, derived);
        return;
    }
    state::NodeState root = nodes[0];
    root.position = cache.position;
    root.rotation = cache.rotation;
    animation::DeriveNodesWithRoot(skeleton, nodes, root, derived);
}

// GRF: `GameTask::PrepareRender`'s `Match::Put` at each render. Team 1 is put mirrored, and
// `Mirror` invalidates the root twice. Team 0 is only read. The officials are mirrored when the
// teams are processed in reverse.
CUJU_FUNCTION inline void MatchPut(World &world) {
    if (world.renderMode != e_Render_On) return;
    state::MatchState &match = world.serialized.match;
    const int playerCount = match.playerCount;
    const bool reverse = world.serialized.scenario.reverseTeamProcessing;
    WarpFor(playerCount + 3, [&](int humanoidIndex) {
        const bool official = humanoidIndex >= playerCount;
        if (!official && !match.players[humanoidIndex].base.isActive) return;
        if (!official) world.humanoidHidden[humanoidIndex] = false;
        const state::NodeState &root =
            official ? match.officials[humanoidIndex - playerCount].humanoid.nodes[0]
                     : match.players[humanoidIndex].base.humanoid.nodes[0];
        const bool mirror = official ? reverse : world.playerTeamID[humanoidIndex] == 1;
        if (mirror)
            RefreshRootCache(world.rootCaches[humanoidIndex], root);
        else
            ReadRootCache(world, humanoidIndex, root);
    });
}

// GRF: `Match::SpamMessage`, which shows the caption until `actualTime_ms + time_ms`.
CUJU_FUNCTION inline void SpamMessage(World &world, MessageKind kind, int team, int player,
                                      int milliseconds = 3000) {
    MatchMessage &message = world.message;
    message.spammed = true;
    message.kind = kind;
    message.team = (int8_t)team;
    message.player = (int8_t)player;
    message.removeTimeMilliseconds =
        world.serialized.match.actualTimeMilliseconds + (uint64_t)milliseconds;
    // GRF: `Gui2Caption::SetCaption` returns at once without rendering, and `Show` and the
    // timeout apply all the same.
    if (world.renderMode == e_Render_On) {
        message.drawn = true;
        message.drawnKind = message.kind;
        message.drawnTeam = message.team;
        message.drawnPlayer = message.player;
    }
}

// GRF: `TeamData`'s names, the team of database ID 3 on the left and of 8 on the right, and
// `PlayerData`'s last names of the eleven database players in lineup order, left then right.
CUJU_FUNCTION inline const char *TeamName(int team) {
    return team == 0 ? "Frequentists United" : "Real Bayesians";
}
CUJU_FUNCTION inline const char *PlayerLastName(int team, int slot) {
    const char *const names[2][11] = {
        {"Lovelace", "Turing", "Johnson", "da Vinci", "Newton", "Blackwell", "Borg", "Euler",
         "Pythagoras", "Curie", "Nixon Sutton"},
        {"Meitner", "Einstein", "Vaughaun", "Archimedinho", "Banach", "Banneker", "Goodall",
         "Copernicus", "Feynman", "Franklin", "Roy Mouton"}};
    return slot >= 0 && slot < 11 ? names[team ? 1 : 0][slot] : "";
}

// The text of `message` as GRF's `SpamMessage` sites compose it, zero terminated, at most
// `MESSAGE_BYTES - 1` bytes. Returns the length.
CUJU_FUNCTION inline int MessageText(const World &world, const MatchMessage &message,
                                     char text[MESSAGE_BYTES]) {
    int length = 0;
    auto append = [&](const char *part) {
        for (; *part && length < MESSAGE_BYTES - 1; part++) text[length++] = *part;
    };
    const char *name = "";
    if (message.player >= 0) {
        const int team = world.playerTeamID[message.player];
        int slot = 0;
        for (int player = 0; player < message.player; player++)
            if (world.playerTeamID[player] == team) slot++;
        name = PlayerLastName(team, slot);
    }
    switch (message.kind) {
        case e_Message_Goal:
            append("GOAL for ");
            append(TeamName(message.team));
            append("! ");
            append(name);
            append(" scores!");
            break;
        case e_Message_GoalNoScorer:
            append("GOAL!!!");
            break;
        case e_Message_OwnGoal:
            append("OWN GOAL! ");
            append(name);
            append(" is so unlucky!");
            break;
        case e_Message_OwnGoalNoScorer:
            append("It's an OWN GOAL! oh noes!");
            break;
        case e_Message_Offside:
            append("offside!");
            break;
        case e_Message_Foul:
            append("foul!");
            break;
        case e_Message_FoulYellow:
            append("foul! yellow card");
            break;
        case e_Message_FoulRed:
            append("foul! red card!!!");
            break;
        case e_Message_EarlyShower:
            append("an early shower for ");
            append(name);
            append("!");
            break;
        case e_Message_SentOff:
            append(name);
            append(" is sent off!");
            break;
        case e_Message_AllOver:
            append("it's all over for ");
            append(name);
            append("!");
            break;
        default:
            break;
    }
    text[length] = 0;
    return length;
}

// The root a render derives an unmirrored humanoid's joints from: the cache when it holds one,
// as `UpdateFullbodyNodes` reads `GetDerivedRotation` and `GetDerivedPosition`. A mirrored one
// derives from its mirrored root, since `Mirror` refreshes the cache.
CUJU_FUNCTION inline state::NodeState RenderRoot(const World &world, int humanoidIndex,
                                                 const state::NodeState &root) {
    state::NodeState out = root;
    const RootCache &cache = world.rootCaches[humanoidIndex];
    if (world.renderMode && cache.valid) {
        out.position = cache.position;
        out.rotation = cache.rotation;
    }
    return out;
}

}  // namespace simulation
}  // namespace cuju
