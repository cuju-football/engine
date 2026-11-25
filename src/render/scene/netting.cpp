// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "scene/netting.h"

#include <cmath>

#include "cuju/libm/powf.h"
#include "cuju/libm/sincosf.h"

namespace cuju {
namespace render {
namespace scene {
namespace {
constexpr float kPitchHalfWidth = 55.0f;
constexpr float kPi = 3.1415926535897932384626433832795028841972f;

inline float Clamp(float value, float minimum, float maximum) {
    if (minimum > value) return minimum;
    if (maximum < value) return maximum;
    return value;
}

// GRF's Vector3::GetDistance: float differences, squares and their sum in double.
inline float Distance(const Vector3 &a, const Vector3 &b) {
    const float dx = a.coordinates[0] - b.coordinates[0];
    const float dy = a.coordinates[1] - b.coordinates[1];
    const float dz = a.coordinates[2] - b.coordinates[2];
    if (dx == 0.0f && dy == 0.0f && dz == 0.0f) return 0.0f;
    float length = (float)std::sqrt((double)dx * dx + (double)dy * dy + (double)dz * dz);
    if (length < 0.000001) length = 0;
    return length;
}

}

void GoalNetting::Prepare(const std::vector<PlanarMesh> &meshes) {
    size_t total = 0;
    meshFloats_.clear();
    for (const PlanarMesh &mesh : meshes) {
        total += mesh.vertices.size();
        meshFloats_.push_back(mesh.vertices.size() / MESH_ELEMENTS);
    }
    buffer_.assign(total, 0.0f);
    const size_t stride = total / MESH_ELEMENTS;
    size_t start = 0;
    for (const PlanarMesh &mesh : meshes) {
        const size_t elementFloats = mesh.vertices.size() / MESH_ELEMENTS;
        for (int e = 0; e < MESH_ELEMENTS; e++)
            for (size_t k = 0; k < elementFloats; k++)
                buffer_[e * stride + start + k] = mesh.vertices[e * elementFloats + k];
        start += elementFloats;
    }
    NewMatch();
}

void GoalNetting::NewMatch() {
    side_[0].clear();
    side_[1].clear();
    resetNetting_ = false;
    size_t start = 0;
    for (size_t elementFloats : meshFloats_) {
        for (size_t i = 0; i < elementFloats; i += 3) {
            const float *p = &buffer_[start + i];
            int goal = -1;
            if (p[0] < -kPitchHalfWidth - 0.06f) goal = 0;
            if (p[0] > kPitchHalfWidth + 0.06f) goal = 1;
            if (goal >= 0)
                side_[goal].push_back(Vertex{(uint32_t)(start + i), Vector3(p[0], p[1], p[2])});
        }
        start += elementFloats;
    }
}

bool GoalNetting::Update(const Vector3 &ball, bool ballTouchesNet) {
    const int side = ball.coordinates[0] < 0 ? 0 : 1;
    if (ballTouchesNet) {
        float shortest = 100000.0f;
        for (const Vertex &vertex : side_[side]) {
            const float distance = Distance(vertex.source, ball);
            if (distance < shortest) shortest = distance;
        }
        for (const Vertex &vertex : side_[side]) {
            float influence = libm::powf(
                Clamp((shortest + 0.0001f) / (Distance(vertex.source, ball) + 0.0001f), 0.0f, 1.0f),
                1.5f);
            const float tension =
                Clamp((std::fabs(ball.coordinates[0]) - kPitchHalfWidth) * 2.0f, 0.0f, 1.0f);
            influence *= tension;
            influence = libm::sinf(influence * kPi - 0.5f * kPi) * 0.5f + 0.5f;
            if (influence > 0.0f) {
                const float keep = 1.0f - influence;
                float *target = &buffer_[vertex.offset];
                for (int c = 0; c < 3; c++)
                    target[c] =
                        vertex.source.coordinates[c] * keep + ball.coordinates[c] * influence;
            }
        }
        resetNetting_ = true;
        return true;
    }
    if (resetNetting_) {
        for (int s = 0; s < 2; s++)
            for (const Vertex &vertex : side_[s])
                for (int c = 0; c < 3; c++)
                    buffer_[vertex.offset + c] = vertex.source.coordinates[c];
        resetNetting_ = false;
        return true;
    }
    return false;
}

}
}
}
