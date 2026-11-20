// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// The geometry steps after loading (GRF, Apache-2.0): AABB of base/geometry/aabb.cpp,
// GetTriangleMeshAABB of base/geometry/trianglemeshutils.cpp, GeometryData::GetAABB,
// utils/splitgeometry.cpp, GraphicsGeometry_GeometryInterpreter::OnLoad with the index buffer of
// OpenGLRenderer3D::CreateVertexBuffer, and the vertex sharing of
// HumanoidBase::PrepareFullbodyModel.

#include <cmath>
#include <cstring>
#include <limits>

#include "assets/assets.h"

namespace cuju::render::assets {
namespace {

constexpr int kElements = 5;

// Folded at compile time, so the subnormal FLT_MIN / 100 survives any FTZ mode.
constexpr float kResetMin = std::numeric_limits<float>::max() / 100;
constexpr float kResetMax = std::numeric_limits<float>::min() / 100;

Aabb MeshAabb(const std::vector<float> &vertices, const std::vector<std::uint32_t> &indices) {
    Aabb aabb;
    aabb.Reset();
    if (indices.empty()) {
        const unsigned int triangles = (unsigned int)vertices.size() / kElements / 3 / 3;
        for (unsigned int t = 0; t < triangles; t++)
            for (unsigned int v = 0; v < 3; v++)
                for (unsigned int i = 0; i < 3; i++) {
                    const float value = vertices[t * 9 + v * 3 + i];
                    if (value < aabb.min.v[i]) aabb.min.v[i] = value;
                    if (value > aabb.max.v[i]) aabb.max.v[i] = value;
                }
    } else {
        // GRF reads vertices[index + i], an index that is not scaled by 3.
        for (std::size_t t = 0; t < indices.size() / 3; t++)
            for (unsigned int v = 0; v < 3; v++)
                for (unsigned int i = 0; i < 3; i++) {
                    const std::size_t at = (std::size_t)indices[t * 3 + v] + i;
                    if (at >= vertices.size()) continue;
                    const float value = vertices[at];
                    if (value < aabb.min.v[i]) aabb.min.v[i] = value;
                    if (value > aabb.max.v[i]) aabb.max.v[i] = value;
                }
    }
    return aabb;
}

void Add(Aabb &aabb, const Aabb &add) {
    for (int i = 0; i < 3; i++)
        if (aabb.min.v[i] > add.min.v[i]) aabb.min.v[i] = add.min.v[i];
    for (int i = 0; i < 3; i++)
        if (aabb.max.v[i] < add.max.v[i]) aabb.max.v[i] = add.max.v[i];
}

}  // namespace

void Aabb::Reset() {
    for (int i = 0; i < 3; i++) {
        min.v[i] = kResetMin;
        max.v[i] = kResetMax;
    }
}

Aabb TriangleMeshAabb(const TriangleMesh &mesh) { return MeshAabb(mesh.vertices, mesh.indices); }

Aabb GeometryAabb(const GeometryData &data) {
    Aabb aabb;
    aabb.Reset();
    for (const TriangleMesh &mesh : data.meshes) Add(aabb, TriangleMeshAabb(mesh));
    return aabb;
}

std::vector<GridCell> SplitGeometry(const std::string &source_name, const GeometryData &source,
                                    float grid_size) {
    struct Index {
        float x, y;
    };
    std::vector<Index> keys;
    std::vector<GridCell> cells;
    for (const TriangleMesh &mesh : source.meshes) {
        const Aabb aabb = MeshAabb(mesh.vertices, {});
        const float centerX = (float)((double)(aabb.min.v[0] + aabb.max.v[0]) / 2.0);
        const float centerY = (float)((double)(aabb.min.v[1] + aabb.max.v[1]) / 2.0);
        const float x = grid_size * (float)(int)std::round(centerX / grid_size);
        const float y = grid_size * (float)(int)std::round(centerY / grid_size);
        std::size_t cell = 0;
        while (cell < keys.size() && !(keys[cell].x == x && keys[cell].y == y)) cell++;
        if (cell == keys.size()) {
            keys.push_back({x, y});
            const std::string at = std::to_string((int)x) + ", " + std::to_string((int)y);
            cells.push_back({source_name + " gridGeomData @ " + at,
                             source_name + " gridGeom @ " + at, GeometryData()});
        }
        TriangleMesh copy;
        copy.material = mesh.material;
        copy.vertices = mesh.vertices;
        cells[cell].data.meshes.push_back(std::move(copy));
    }
    return cells;
}

VertexBufferData ConcatenateMeshes(const GeometryData &data) {
    VertexBufferData out;
    std::size_t total = 0;
    for (const TriangleMesh &mesh : data.meshes) total += mesh.vertices.size();
    out.vertices.assign(total, 0.0f);
    const std::size_t elementSize = total / kElements;
    int startIndex = 0;
    int currentSize = 0;
    for (const TriangleMesh &mesh : data.meshes) {
        startIndex += currentSize;
        const std::size_t meshElementSize = mesh.vertices.size() / kElements;
        for (int e = 0; e < kElements; e++)
            if (meshElementSize)
                std::memcpy(&out.vertices[e * elementSize + startIndex],
                            &mesh.vertices[e * meshElementSize], meshElementSize * sizeof(float));
        for (std::uint32_t index : mesh.indices)
            out.indices.push_back((std::uint32_t)(startIndex / 3) + index);
        currentSize = (int)meshElementSize;
        IndexRange range;
        if (!out.indices.empty()) {
            range.start = (int)(out.indices.size() - mesh.indices.size());
            range.size = (int)mesh.indices.size();
        } else {
            range.start = startIndex / 3;
            range.size = currentSize / 3;
        }
        out.ranges.push_back(range);
    }
    out.uses_indices = !out.indices.empty();
    if (!out.uses_indices) {
        const unsigned int count = (unsigned int)total / kElements / 3;
        out.indices.reserve(count);
        for (unsigned int i = 0; i < count; i++) out.indices.push_back(i);
    }
    return out;
}

GeometryData IndexFullbody(const GeometryData &fullbody, float z_multiplier) {
    GeometryData out;
    for (const TriangleMesh &mesh : fullbody.meshes) {
        const int elementOffset = (int)mesh.vertices.size() / kElements;
        const float *data = mesh.vertices.data();
        // The first occurrence of each position and texcoord pair keeps all five elements.
        std::vector<int> unique;
        TriangleMesh result;
        result.material = mesh.material;
        for (int v = 0; v < elementOffset; v += 3) {
            const float *position = data + v;
            const float *texcoord = data + v + 2 * elementOffset;
            int index = -1;
            for (std::size_t i = 0; i < unique.size(); i++) {
                const float *p = data + unique[i];
                const float *t = data + unique[i] + 2 * elementOffset;
                if (p[0] == position[0] && p[1] == position[1] && p[2] == position[2] &&
                    t[0] == texcoord[0] && t[1] == texcoord[1] && t[2] == texcoord[2]) {
                    index = (int)i;
                    break;
                }
            }
            if (index < 0) {
                unique.push_back(v);
                index = (int)unique.size() - 1;
            }
            result.indices.push_back((std::uint32_t)index);
        }
        const int uniqueElementOffset = (int)unique.size() * 3;
        result.vertices.assign((std::size_t)uniqueElementOffset * kElements, 0.0f);
        for (std::size_t v = 0; v < unique.size(); v++)
            for (int e = 0; e < kElements; e++)
                for (int c = 0; c < 3; c++)
                    result.vertices[v * 3 + e * uniqueElementOffset + c] =
                        data[unique[v] + e * elementOffset + c];
        for (int v = 0; v < uniqueElementOffset; v++) result.vertices[v] *= z_multiplier;
        out.meshes.push_back(std::move(result));
    }
    return out;
}

}  // namespace cuju::render::assets
