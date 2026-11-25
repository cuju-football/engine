// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../inline_vector.h"
#include "../state/environment_state.h"

namespace cuju {
namespace simulation {
constexpr int MAXIMUM_ANIMATIONS = 1536;
constexpr int MAXIMUM_COMMAND_QUEUE = 16;
constexpr int MAXIMUM_FORCE_SPOTS = 64;
constexpr int MAXIMUM_POSITIONS = state::MAXIMUM_ANIMATION_POSITIONS;

using PlayerList = InlineVector<int, state::MAXIMUM_PLAYERS>;
using AnimationList = InlineVector<int16_t, MAXIMUM_ANIMATIONS>;
using CommandQueue = InlineVector<state::PlayerCommand, MAXIMUM_COMMAND_QUEUE>;
using PositionList = InlineVector<Vector3, MAXIMUM_POSITIONS>;

}
}
