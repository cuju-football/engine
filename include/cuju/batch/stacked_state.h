// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_core.h"

namespace cuju {
namespace batch {
template <class Batch>
void SetStackedFrames(Batch &batch, int environment, const std::string &name, const void *frames,
                      size_t size) {
    const EnvironmentBuffers &buffers = batch.Buffers();
    if (environment < 0 || environment >= buffers.environmentCount)
        throw std::invalid_argument("expected an environment from 0 to " +
                                    std::to_string(buffers.environmentCount - 1) + ", found " +
                                    std::to_string(environment));
    if (buffers.stack < 2)
        throw std::invalid_argument("expected a batch that stacks its observations");
    const size_t agents = (size_t)(batch.LeftAgents(environment) + batch.RightAgents(environment));
    const size_t first = (size_t)environment * buffers.agents;
    const int stack = buffers.stack;
    auto checkSize = [&](size_t expected) {
        if (size != expected)
            throw std::invalid_argument("expected a stack of " + std::to_string(expected) +
                                        " bytes for the agents of environment " +
                                        std::to_string(environment) + ", found " +
                                        std::to_string(size));
    };
    if (name == "simple115" && buffers.simple115) {
        const size_t width = (size_t)SIMPLE115 * stack;
        checkSize(agents * width * sizeof(float));
        batch.Device().ToDevice(buffers.simple115 + first * width, frames, size);
        return;
    }
    if (name != "smm" || !buffers.smm)
        throw std::invalid_argument("expected the stack simple115 or smm of the batch, found " +
                                    name);
    constexpr int LAYERS = simulation::SMM_LAYERS, MARK_ROW = 1 + simulation::SMM_MARKS;
    const size_t frameMarks = (size_t)buffers.smmWidth * buffers.smmHeight * LAYERS;
    const size_t agentBytes = frameMarks * stack;
    checkSize(agents * agentBytes);
    const uint8_t *bytes = static_cast<const uint8_t *>(frames);
    std::vector<int32_t> marks(agents * stack * MARK_ROW, 0);
    for (size_t agent = 0; agent < agents; agent++)
        for (int frame = 0; frame < stack; frame++) {
            int32_t *row = marks.data() + (agent * stack + frame) * MARK_ROW;
            for (size_t mark = 0; mark < frameMarks; mark++) {
                const uint8_t value = bytes[agent * agentBytes + mark / LAYERS * LAYERS * stack +
                                            LAYERS * frame + mark % LAYERS];
                if (value == 0) continue;
                // GRF: the minimap marks a cell with 255 and leaves the others 0.
                if (value != 255 || row[0] == simulation::SMM_MARKS)
                    throw std::invalid_argument(
                        "expected a minimap of GRF, of at most " +
                        std::to_string(simulation::SMM_MARKS) +
                        " cells of 255 a frame and zeros, found a cell of " +
                        std::to_string(value));
                row[1 + row[0]++] = (int32_t)mark;
            }
        }
    batch.Device().ToDevice(buffers.smm + first * agentBytes, frames, size);
    batch.Device().ToDevice(buffers.smmMarks + first * MARK_ROW * stack, marks.data(),
                            marks.size() * sizeof(int32_t));
}

}
}
