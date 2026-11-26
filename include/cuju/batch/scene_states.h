// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../scene/extract_scene.h"
#include "batch_core.h"

namespace cuju {
namespace batch {
struct SceneBuffers {
    const uint8_t *animationMirrored;
    const int32_t *indices;
    int count;
    scene::SceneState *out;
};

CUJU_FUNCTION inline void WriteSceneState(const EnvironmentBuffers &buffers,
                                          const Variants &variants, const SceneBuffers &scene,
                                          int output) {
    const int environment = scene.indices[output];
    const EnvironmentSlot &slot = buffers.slots[environment];
    render::RenderContext context;
    context.simulation = &variants.contexts[slot.variant];
    context.animationMirrored = scene.animationMirrored;
    scene::SceneState &state = scene.out[output];
    scene::ExtractScene(buffers.worlds[environment], context, state);
    state.episode = slot.episode;
}

}  // namespace batch
}  // namespace cuju
