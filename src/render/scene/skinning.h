// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The full body mesh of a player or an official, skinned on the CPU as GRF's
// HumanoidBase::PrepareFullbodyModel and UpdateFullbodyModel do. Ported from GRF, Apache-2.0.
// Exact: float operations in GRF's order, no contraction, FastNormalize's integer guess and one
// Newton step, the weights in double as GRF divides them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "cuju/quaternion.h"
#include "cuju/render/render_state.h"
#include "cuju/vector3.h"

namespace cuju {
namespace render {
namespace scene {

// GRF's mesh elements per vertex: position, normal, texture coordinate, tangent, bitangent.
constexpr int MESH_ELEMENTS = 5;

// One submesh in GRF's planar layout: element e of vertex v at `e * 3 * vertexCount + 3 * v`.
struct PlanarMesh {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    int VertexCount() const { return (int)(vertices.size() / (3 * MESH_ELEMENTS)); }
};

struct SkinBone {
    int32_t joint;
    float weight;
};
struct SkinVertex {
    int32_t boneCount;
    SkinBone bones[3];
};
struct SkinSubmesh {
    // The source of every render, the mesh after the bake, positions scaled by `zMultiplier`.
    PlanarMesh source;
    std::vector<SkinVertex> weights;  // one per vertex
};

// One humanoid's model. `origin` is the straight pose's joint positions, unscaled.
struct SkinnedModel {
    float zMultiplier = 1.0f;
    Vector3 origin[RENDER_JOINTS];
    std::vector<SkinSubmesh> submeshes;
    // The floats of the whole vertex buffer, every element of every submesh.
    size_t BufferFloats() const;
};

// The joints of the bake: the inverse rotations of base.anim.util and the joint positions of
// base.anim.util and straight.anim.util on players/player.object, all unscaled.
struct BindPose {
    Quaternion inverseRotation[RENDER_JOINTS];
    Vector3 basePosition[RENDER_JOINTS];
    Vector3 straightPosition[RENDER_JOINTS];
};
const BindPose &FullbodyBindPose();

// The key of `GetVertexColors`: a vertex position as GRF's `Vector3::operator<` orders it.
struct PositionLess {
    bool operator()(const Vector3 &a, const Vector3 &b) const;
};
using VertexColorMap = std::map<Vector3, Vector3, PositionLess>;

// PrepareFullbodyModel: de-duplicates each submesh by position and texture coordinate, scales the
// positions, derives the bone weights from the vertex colours and bakes the bind pose, which GRF
// skips when it does not render, `bake` false.
// Returns false if a vertex has no colour or a first weight of zero, which GRF asserts against.
bool PrepareFullbodyModel(const std::vector<PlanarMesh> &meshes, const VertexColorMap &colors,
                          float zMultiplier, const BindPose &pose, SkinnedModel &model,
                          bool bake = true);

// UpdateFullbodyModel for one render, into `buffer` with `model.BufferFloats()` floats laid out
// as GRF's vertex buffer: element e of every submesh in turn, then element e + 1.
void SkinFullbody(const SkinnedModel &model, const RenderJoint joints[RENDER_JOINTS],
                  float *buffer);

// The vertex buffer of a model before any render, the bake's result.
void FullbodyRestBuffer(const SkinnedModel &model, float *buffer);

}  // namespace scene
}  // namespace render
}  // namespace cuju
