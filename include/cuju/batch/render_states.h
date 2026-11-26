// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "../render/extract_render_state.h"
#include "batch_core.h"

namespace cuju {
namespace batch {
struct RenderBuffers {
    render::RenderState *states;
    int32_t *episodes;
    const uint8_t *animationMirrored;
    const int32_t *indices;
    int count;
    render::RenderState *out;
    uint32_t *processed;
};

CUJU_FUNCTION inline void WriteRenderState(const EnvironmentBuffers &buffers,
                                           const Variants &variants, const RenderBuffers &render,
                                           int output) {
    const int environment = render.indices[output];
    const simulation::World &world = buffers.worlds[environment];
    const EnvironmentSlot &slot = buffers.slots[environment];
    render::RenderMessage message;
    MakeRenderMessage(world, message);
    render::RenderContext context;
    context.simulation = &variants.contexts[slot.variant];
    context.animationMirrored = render.animationMirrored;
    context.message = &message;
    render::RenderState &state = render.states[environment];
    if (render.episodes[environment] != slot.episode) {
        render::ResetRenderState(world, context, MakeRenderDraws(world), state);
        render.episodes[environment] = slot.episode;
    }
    render::ExtractRenderState(world, context, state);
    render.out[output] = state;
    render.processed[output] = world.stateProcessed;
}

inline std::vector<uint8_t> AnimationMirroredNames(const std::vector<std::string> &names) {
    std::vector<uint8_t> mirrored;
    mirrored.reserve(names.size());
    for (const std::string &name : names)
        mirrored.push_back(name.find("mirror") != std::string::npos ? 1 : 0);
    return mirrored;
}

}  // namespace batch
}  // namespace cuju
