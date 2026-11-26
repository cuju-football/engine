// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The goal nets pulled toward the ball, as GRF's Match::PrepareGoalNetting and UpdateGoalNetting
// do.
#pragma once

#include <cstdint>
#include <vector>

#include "cuju/vector3.h"
#include "scene/skinning.h"

namespace cuju {
namespace render {
namespace scene {
class GoalNetting {
public:
    void Prepare(const std::vector<PlanarMesh> &meshes);
    void NewMatch();
    // GRF's Match::ProcessState, for a saved or a loaded state, makes the next render restore and
    // upload the nets.
    void StateProcessed() { resetNetting_ = true; }
    bool Update(const Vector3 &ballPosition, bool ballTouchesNet);
    const std::vector<float> &Buffer() const { return buffer_; }
    int NettingVertices(int side) const { return (int)side_[side].size(); }

private:
    struct Vertex {
        uint32_t offset;
        Vector3 source;
    };
    std::vector<float> buffer_;
    std::vector<size_t> meshFloats_;
    std::vector<Vertex> side_[2];
    bool resetNetting_ = false;
};

}  // namespace scene
}  // namespace render
}  // namespace cuju
