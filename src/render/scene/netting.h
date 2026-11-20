// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The goal nets pulled toward the ball, as GRF's Match::PrepareGoalNetting and
// UpdateGoalNetting do. Ported from GRF, Apache-2.0.
// Exact: float operations in GRF's order, glibc's powf and sinf through cuju's libm, distances
// with GRF's squares in double.
#pragma once

#include <cstdint>
#include <vector>

#include "cuju/vector3.h"
#include "scene/skinning.h"

namespace cuju {
namespace render {
namespace scene {

// The vertex buffer of the goals geometry and the state the nets keep between renders. GRF keeps
// the deformed vertices in the mesh, so a vertex that the ball stops pulling keeps its place until
// the ball leaves the net, and only the ball's side moves.
class GoalNetting {
public:
    // The meshes of goals.ase in GRF's order. Every position whose x lies beyond the goal lines by
    // 0.06 is netting, the rest is woodwork.
    void Prepare(const std::vector<PlanarMesh> &meshes);
    // A new match's PrepareGoalNetting. GRF keeps the goals mesh for the process, so a match
    // takes the vertices as the last one left them, deformed or not, as its rest positions.
    void NewMatch();
    // GRF's Match::ProcessState, for a saved or a loaded state, makes the next render restore
    // and upload the nets.
    void StateProcessed() { resetNetting_ = true; }
    // One render's UpdateGoalNetting. Returns GRF's `nettingHasChanged`, when GRF uploads the
    // whole buffer.
    bool Update(const Vector3 &ballPosition, bool ballTouchesNet);
    // The goals geometry's vertex buffer as GRF uploads it, every element of every mesh.
    const std::vector<float> &Buffer() const { return buffer_; }
    int NettingVertices(int side) const { return (int)side_[side].size(); }

private:
    struct Vertex {
        uint32_t offset;  // of x in `buffer_`
        Vector3 source;
    };
    std::vector<float> buffer_;
    // The floats of each mesh's elements, in order.
    std::vector<size_t> meshFloats_;
    std::vector<Vertex> side_[2];
    bool resetNetting_ = false;
};

}  // namespace scene
}  // namespace render
}  // namespace cuju
