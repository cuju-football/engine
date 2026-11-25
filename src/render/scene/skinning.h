// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

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
    PlanarMesh source;
    std::vector<SkinVertex> weights;
};

struct SkinnedModel {
    float zMultiplier = 1.0f;
    Vector3 origin[RENDER_JOINTS];
    std::vector<SkinSubmesh> submeshes;
    size_t BufferFloats() const;
};

struct BindPose {
    Quaternion inverseRotation[RENDER_JOINTS];
    Vector3 basePosition[RENDER_JOINTS];
    Vector3 straightPosition[RENDER_JOINTS];
};
const BindPose &FullbodyBindPose();

struct PositionLess {
    bool operator()(const Vector3 &a, const Vector3 &b) const;
};
using VertexColorMap = std::map<Vector3, Vector3, PositionLess>;

bool PrepareFullbodyModel(const std::vector<PlanarMesh> &meshes, const VertexColorMap &colors,
                          float zMultiplier, const BindPose &pose, SkinnedModel &model,
                          bool bake = true);

void SkinFullbody(const SkinnedModel &model, const RenderJoint joints[RENDER_JOINTS],
                  float *buffer);

void FullbodyRestBuffer(const SkinnedModel &model, float *buffer);

}
}
}
