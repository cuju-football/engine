// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The world of one environment and the mirroring between its coordinate frames.
// Exact: a mirror turns the pitch by negating x and y only, so a second mirror restores the bits.
#pragma once

#include "../warp.h"

#include "../profiling.h"

#include "../state/environment_state.h"

namespace cuju {
namespace simulation {

using state::EnvironmentStateFull;
using state::HumanoidState;
using state::MatchState;
using state::MentalImageState;
using state::PlayerImage;
using state::Radian;

// In the rest frame between ticks, each team is in its own frame, side -1, no mirror, the ball in
// team 0's. `MirrorMatch` moves to the call frame of the team being processed, and a state stores
// the canonical frame of `CanonicalToRest`. These flags are not in the state.
struct FrameFlags {
    bool teamMirrored[2] = {false, false};
    int teamSide[2] = {-1, -1};
    bool ballMirrored = false;
    bool humanoidMirrored[state::MAXIMUM_PLAYERS] = {};
    bool mentalBallMirrored[4] = {};
};

// GRF: render mode. Off plays without graphics. On plays with graphics that render, whose
// observers recompute a root's cache at every invalidation, and `ResetPosition` moves the root
// without one, so later reads see the root where it was. Idle plays with graphics after
// `game_config.render` turned off, where a cache computed before stays until its invalidation.
enum RenderMode : int8_t { e_Render_Off, e_Render_On, e_Render_Idle };

// GRF: the cached world transform of a humanoid's root node, which is absolute, so the cache is
// its own position and rotation when last computed. Not `valid` is a cache to compute afresh.
struct RootCache {
    Vector3 position;
    Quaternion rotation;
    bool valid = false;
};
// The players, then the three officials, as `AnyHumanoid` counts them.
constexpr int MAXIMUM_HUMANOIDS = state::MAXIMUM_PLAYERS + 3;

// GRF: `SpamMessage`, the message caption's last message, which no state holds. The text is made
// from `kind`, `team`, a GRF team ID, and `player`, at render time.
enum MessageKind : int8_t {
    e_Message_Goal,
    e_Message_GoalNoScorer,
    e_Message_OwnGoal,
    e_Message_OwnGoalNoScorer,
    e_Message_Offside,
    e_Message_Foul,
    e_Message_FoulYellow,
    e_Message_FoulRed,
    e_Message_EarlyShower,
    e_Message_SentOff,
    e_Message_AllOver
};
// The bytes of a message's text with its terminating zero, at most.
constexpr int MESSAGE_BYTES = 128;
struct MatchMessage {
    bool spammed = false;
    int8_t kind = 0;
    int8_t team = -1;
    int8_t player = -1;
    // GRF: the caption takes a message's text only while rendering, so it shows the last message
    // spammed then, or none.
    bool drawn = false;
    int8_t drawnKind = 0;
    int8_t drawnTeam = -1;
    int8_t drawnPlayer = -1;
    uint64_t removeTimeMilliseconds = 0;
};

// The draws of the match's creation that only the pictures read, the officials' shared skin and
// the six of the light, in the order of the RNG.
struct MatchDraws {
    float officialSkin = 1.0f;
    float light[6] = {};
};

// A state together with the data that a state lacks.
struct World {
    EnvironmentStateFull serialized;
    FrameFlags flags;
    int playerTeamID[state::MAXIMUM_PLAYERS];
    // 0 selects the player's own formation entry, any other value 1 plus the player whose entry it
    // uses. GRF: a goalkeeper sent off hands its entry to the first active player of its team.
    int8_t formationOverride[state::MAXIMUM_PLAYERS];
    // GRF: the `animations` flag of `GameEnv::reset`, which the referee holds for the match.
    // Without it the clock skips the wait of a stoppage at once.
    bool refereeAnimations = false;
    // GRF: `ScenarioConfig` caches `DynamicPlayerSelection` until a state is saved or loaded.
    // -1 computes it at every read, as the batch does. 0 and 1 are a cached value, and 2 and 3 a
    // value that the first read caches.
    int8_t dynamicPlayerSelection = -1;
    // A `RenderMode`. Off, every read of a root is fresh.
    int8_t renderMode = e_Render_Off;
    RootCache rootCaches[MAXIMUM_HUMANOIDS];
    // For the pictures only, cleared at the creation of each match.
    MatchMessage message;
    MatchDraws draws;
    // GRF: the scoreboard, in `goalCount`'s order, which takes a team's count at its goals only,
    // and only while rendering, so a load leaves it as it was.
    int32_t scoreboardGoals[2] = {0, 0};
    // GRF: `Deactivate` moves a sent-off player's body and hair out of sight, and only a `Put` of
    // the player active again brings them back, so a player loaded inactive stays where it stood.
    bool humanoidHidden[state::MAXIMUM_PLAYERS] = {};
    // GRF: the count of `ProcessState` runs, each of which makes the next render upload the
    // nets again.
    uint32_t stateProcessed = 0;
    CUJU_FUNCTION int FirstTeamID() const {
        return serialized.match.firstTeam ^ (serialized.scenario.reverseTeamProcessing ? 1 : 0);
    }
};

// The root's cache computed afresh.
CUJU_FUNCTION inline void RefreshRootCache(RootCache &cache, const state::NodeState &root) {
    cache.position = root.position;
    cache.rotation = root.rotation;
    cache.valid = true;
}
// GRF: an invalidation of a root, which the observers of rendering graphics follow at once with a
// computation.
CUJU_FUNCTION inline void InvalidateRoot(World &world, int humanoidIndex,
                                         const state::NodeState &root) {
    if (world.renderMode == e_Render_On)
        RefreshRootCache(world.rootCaches[humanoidIndex], root);
    else if (world.renderMode == e_Render_Idle)
        world.rootCaches[humanoidIndex].valid = false;
}

CUJU_FUNCTION inline void MirrorVector(Vector3 &vector) { vector.Mirror(); }
CUJU_FUNCTION inline void MirrorRadian(Radian &radian) { radian.rotated = !radian.rotated; }

CUJU_FUNCTION_NO_INLINE inline void MirrorHumanoid(HumanoidState &humanoid, bool &mirrored) {
    CUJU_PROFILE_SCOPE(MIRROR_HUMANOID);
    state::NodeState &root = humanoid.nodes[0];
    root.position = root.position * Vector3(-1, -1, 1);
    Quaternion rotation = root.rotation;
    if (!mirrored) {
        root.rotation = Quaternion(rotation.elements[1], rotation.elements[0], rotation.elements[3],
                                   -rotation.elements[2]);
    } else {
        root.rotation = Quaternion(rotation.elements[1], rotation.elements[0],
                                   -rotation.elements[3], rotation.elements[2]);
    }
    mirrored = !mirrored;
    MirrorVector(humanoid.startPosition);
    MirrorRadian(humanoid.startAngle);
    MirrorVector(humanoid.nextStartPosition);
    MirrorRadian(humanoid.nextStartAngle);
    // A spatial state mirrors its positions and movements and keeps its angles.
    state::SpatialState &spatialState = humanoid.spatialState;
    MirrorVector(spatialState.position);
    MirrorVector(spatialState.actualMovement);
    MirrorVector(spatialState.physicsMovement);
    MirrorVector(spatialState.animationMovement);
    MirrorVector(spatialState.movement);
    MirrorVector(spatialState.actionSmuggleMovement);
    MirrorVector(spatialState.movementSmuggleMovement);
    MirrorVector(spatialState.positionOffsetMovement);
    MirrorVector(humanoid.previousPosition2D);
    MirrorVector(humanoid.tripDirection);
    MirrorVector(humanoid.decayingPositionOffset);
    // The scratch values of the animation selection are not state and stay as they are.
}

CUJU_FUNCTION inline void MirrorTeam(World &world, int teamID) {
    world.flags.teamSide[teamID] *= -1;
    world.flags.teamMirrored[teamID] = !world.flags.teamMirrored[teamID];
    WarpFor(world.serialized.match.playerCount, [&](int player) {
        if (world.playerTeamID[player] != teamID) return;
        HumanoidState &humanoid = world.serialized.match.players[player].base.humanoid;
        MirrorHumanoid(humanoid, world.flags.humanoidMirrored[player]);
        // GRF: `Mirror` ends with `SetRotation`, which invalidates the root.
        if (world.renderMode) InvalidateRoot(world, player, humanoid.nodes[0]);
    });
}

// The prediction and history arrays mirror lazily. `lazyFlip` says they hold negated x and y, a
// reader applies it per value, and a user of the whole arrays calls `BallMaterialize` first.
CUJU_FUNCTION inline void MirrorBall(state::BallSerialized &ball) {
    MirrorVector(ball.momentum);
    ball.lazyFlip = !ball.lazyFlip;
    MirrorVector(ball.positionBuffer);
}
CUJU_FUNCTION inline void BallMaterialize(state::BallSerialized &ball) {
    if (!ball.lazyFlip) return;
    constexpr int predictionCount = sizeof(ball.predictions) / sizeof(ball.predictions[0]);
    WarpFor(predictionCount + ball.ballPositionHistory.count, [&](int index) {
        MirrorVector(index < predictionCount ? ball.predictions[index]
                                             : ball.ballPositionHistory[index - predictionCount]);
    });
    ball.lazyFlip = false;
}
CUJU_FUNCTION inline void MentalMaterialize(MentalImageState &image) {
    if (!image.lazyFlip) return;
    WarpFor(image.ballPredictions.count,
            [&](int index) { MirrorVector(image.ballPredictions[index]); });
    image.lazyFlip = false;
}

CUJU_FUNCTION inline void MirrorPlayerImage(PlayerImage &image) {
    MirrorVector(image.position);
    MirrorVector(image.directionVector);
    MirrorVector(image.movement);
}

CUJU_FUNCTION inline void MirrorMentalImage(World &world, int imageIndex, bool team0, bool team1,
                                            bool ball) {
    MentalImageState &image = world.serialized.match.mentalImages[imageIndex];
    WarpFor(image.players.count, [&](int index) {
        int teamID = world.playerTeamID[image.players[index].player];
        if (teamID == 0 ? team0 : team1) MirrorPlayerImage(image.players[index]);
    });
    if (ball) {
        image.lazyFlip = !image.lazyFlip;
        world.flags.mentalBallMirrored[imageIndex] = !world.flags.mentalBallMirrored[imageIndex];
    }
}

CUJU_FUNCTION inline void MirrorMatch(World &world, bool team0, bool team1, bool ball) {
    CUJU_PROFILE_SCOPE(MIRROR_MATCH);
    if (team0) MirrorTeam(world, 0);
    if (team1) MirrorTeam(world, 1);
    if (ball) {
        world.flags.ballMirrored = !world.flags.ballMirrored;
        MirrorBall(world.serialized.match.ball);
    }
    for (int image = 0; image < world.serialized.match.mentalImages.count; image++)
        MirrorMentalImage(world, image, team0, team1, ball);
}

// A state stores the ball, the last-touch team IDs and the referee's team and positions relative
// to processing order, and the goal counts and possession times in state order. The flags stay.
// The conversion is an involution, so `RestToCanonical` is the same function.
CUJU_FUNCTION_NO_INLINE inline void CanonicalToRest(World &world) {
    MatchState &match = world.serialized.match;
    if (world.FirstTeamID() == 1) {
        for (int32_t &teamID : match.lastTouchTeamIDs)
            if (teamID != -1) teamID = 1 - teamID;
        if (match.lastTouchTeamID != -1) match.lastTouchTeamID = 1 - match.lastTouchTeamID;
    }
    if (!world.serialized.scenario.reverseTeamProcessing) return;
    MirrorBall(match.ball);
    for (int image = 0; image < match.mentalImages.count; image++)
        match.mentalImages[image].lazyFlip = !match.mentalImages[image].lazyFlip;
    match.referee.buffer.teamID = 1 - match.referee.buffer.teamID;
    MirrorVector(match.referee.buffer.restartPosition);
    MirrorVector(match.referee.foul.foulPosition);
}
CUJU_FUNCTION inline void RestToCanonical(World &world) { CanonicalToRest(world); }

}  // namespace simulation
}  // namespace cuju
