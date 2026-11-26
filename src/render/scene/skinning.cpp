// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "scene/skinning.h"

#include <cmath>
#include <cstring>

namespace cuju {
namespace render {
namespace scene {
namespace {
struct Vec {
    float x, y, z;
};

inline Vec Load(const float *p) { return Vec{p[0], p[1], p[2]}; }
inline void Store(float *p, const Vec &v) {
    p[0] = v.x;
    p[1] = v.y;
    p[2] = v.z;
}
inline Vec FromVector(const Vector3 &v) {
    return Vec{v.coordinates[0], v.coordinates[1], v.coordinates[2]};
}

// GRF's Vector3::Rotate, two cross products written out.
inline Vec Rotate(Vec v, const Quaternion &quaternion) {
    const float *q = quaternion.elements;
    float uvx = v.z * q[1] - v.y * q[2];
    float uvy = v.x * q[2] - v.z * q[0];
    float uvz = v.y * q[0] - v.x * q[1];
    float uuvx = uvz * q[1] - uvy * q[2];
    float uuvy = uvx * q[2] - uvz * q[0];
    float uuvz = uvy * q[0] - uvx * q[1];
    const float w2 = 2.0f * q[3];
    uvx *= w2;
    uvy *= w2;
    uvz *= w2;
    uuvx *= 2.0f;
    uuvy *= 2.0f;
    uuvz *= 2.0f;
    v.x += uvx + uuvx;
    v.y += uvy + uuvy;
    v.z += uvz + uuvz;
    return v;
}

// GRF's Vector3::FastNormalize.
inline Vec FastNormalize(Vec v) {
    float x = v.x * v.x + v.y * v.y + v.z * v.z;
    const float xhalf = 0.5f * x;
    int32_t i;
    std::memcpy(&i, &x, 4);
    i = 0x5f3759df - (i >> 1);
    std::memcpy(&x, &i, 4);
    x *= 1.5f - xhalf * x * x;
    v.x *= x;
    v.y *= x;
    v.z *= x;
    return v;
}

inline Vec Scaled(const Vector3 &v, float s) {
    return Vec{v.coordinates[0] * s, v.coordinates[1] * s, v.coordinates[2] * s};
}

struct Joint {
    Quaternion orientation;
    Vector3 position;
};

void SkinSubmeshInto(const SkinSubmesh &submesh, const Joint *joints, const Vector3 *origin,
                     float zMultiplier, const float *source, float *out, size_t stride) {
    const int vertexCount = (int)submesh.weights.size();
    const size_t elementFloats = (size_t)3 * submesh.source.VertexCount();
    for (int v = 0; v < vertexCount; v++) {
        const SkinVertex &weighted = submesh.weights[v];
        const Vec origVertex = Load(&source[v * 3]);
        const Vec origNormal = Load(&source[v * 3 + elementFloats]);
        const Vec origTangent = Load(&source[v * 3 + elementFloats * 3]);
        const Vec origBitangent = Load(&source[v * 3 + elementFloats * 4]);
        Vec vertex, normal, tangent, bitangent;
        if (weighted.boneCount == 1) {
            const int j = weighted.bones[0].joint;
            const Vec o = Scaled(origin[j], zMultiplier);
            vertex = Vec{origVertex.x - o.x, origVertex.y - o.y, origVertex.z - o.z};
            vertex = Rotate(vertex, joints[j].orientation);
            const Vec p = Scaled(joints[j].position, zMultiplier);
            vertex.x += p.x;
            vertex.y += p.y;
            vertex.z += p.z;
            normal = Rotate(origNormal, joints[j].orientation);
            tangent = Rotate(origTangent, joints[j].orientation);
            bitangent = Rotate(origBitangent, joints[j].orientation);
        } else {
            vertex = normal = tangent = bitangent = Vec{0.0f, 0.0f, 0.0f};
            for (int b = 0; b < weighted.boneCount; b++) {
                const int j = weighted.bones[b].joint;
                const float w = weighted.bones[b].weight;
                const Vec o = Scaled(origin[j], zMultiplier);
                Vec adapted = Vec{origVertex.x - o.x, origVertex.y - o.y, origVertex.z - o.z};
                adapted = Rotate(adapted, joints[j].orientation);
                const Vec p = Scaled(joints[j].position, zMultiplier);
                adapted.x += p.x;
                adapted.y += p.y;
                adapted.z += p.z;
                vertex.x += adapted.x * w;
                vertex.y += adapted.y * w;
                vertex.z += adapted.z * w;
                const Vec n = Rotate(origNormal, joints[j].orientation);
                normal.x += n.x * w;
                normal.y += n.y * w;
                normal.z += n.z * w;
                const Vec t = Rotate(origTangent, joints[j].orientation);
                tangent.x += t.x * w;
                tangent.y += t.y * w;
                tangent.z += t.z * w;
                const Vec bt = Rotate(origBitangent, joints[j].orientation);
                bitangent.x += bt.x * w;
                bitangent.y += bt.y * w;
                bitangent.z += bt.z * w;
            }
            normal = FastNormalize(normal);
            tangent = FastNormalize(tangent);
            bitangent = FastNormalize(bitangent);
        }
        Store(&out[v * 3], vertex);
        Store(&out[v * 3 + stride], normal);
        Store(&out[v * 3 + stride * 3], tangent);
        Store(&out[v * 3 + stride * 4], bitangent);
    }
}

}  // namespace

const BindPose &FullbodyBindPose() {
    static const BindPose pose = {
        {Quaternion(0x1.0c6f7ap-20f, -0.0f, -0.0f, 0x1.0p+0f),
         Quaternion(-0x1.bec13ap-3f, -0.0f, -0.0f, 0x1.f3ab66p-1f),
         Quaternion(-0x1.de17c8p-3f, -0.0f, -0.0f, 0x1.f1da84p-1f),
         Quaternion(-0x1.83c5ecp-3f, 0x1.2e276p-2f, 0x1.57c68ap-3f, 0x1.d7bc92p-1f),
         Quaternion(0x1.3e9d06p-2f, 0x1.55ae3p-3f, 0x1.2ebf5p-2f, 0x1.c67ba6p-1f),
         Quaternion(-0x1.83c5eep-3f, -0x1.2e276ep-2f, -0x1.57c608p-3f, 0x1.d7bc94p-1f),
         Quaternion(0x1.3e9d08p-2f, -0x1.55ae8ap-3f, -0x1.2ebf2p-2f, 0x1.c67ba8p-1f),
         Quaternion(0x1.0496e8p-2f, 0x1.06f2dep-3f, -0x1.0d1654p-4f, 0x1.e99b44p-1f),
         Quaternion(-0x1.519edap-2f, 0x1.1a37ecp-4f, -0x1.037c0ap-3f, 0x1.ddb26cp-1f),
         Quaternion(-0x1.66ac1p-2f, 0x1.0eab54p-4f, -0x1.068aep-3f, 0x1.d9d886p-1f),
         Quaternion(0x1.0496e8p-2f, -0x1.06f2dep-3f, 0x1.0d1654p-4f, 0x1.e99b44p-1f),
         Quaternion(-0x1.519f0cp-2f, -0x1.1a37ccp-4f, 0x1.037c1p-3f, 0x1.ddb26p-1f),
         Quaternion(-0x1.66ac44p-2f, -0x1.0eab34p-4f, 0x1.068ae6p-3f, 0x1.d9d87cp-1f)},
        {Vector3(0.0f, 0.0f, 0x1.eb851ep-1f), Vector3(0.0f, 0x1.421f6p-22f, 0x1.1c28f6p+0f),
         Vector3(0.0f, -0x1.eb967p-3f, 0x1.8cb542p+0f),
         Vector3(0x1.47ae14p-3f, -0x1.b51608p-3f, 0x1.8a415cp+0f),
         Vector3(0x1.692dfp-2f, -0x1.039bb4p-3f, 0x1.495076p+0f),
         Vector3(-0x1.47ae14p-3f, -0x1.b51608p-3f, 0x1.8a415cp+0f),
         Vector3(-0x1.692dfp-2f, -0x1.039b9ep-3f, 0x1.495078p+0f),
         Vector3(0x1.645a1cp-4f, -0x1.5798eep-26f, 0x1.e66666p-1f),
         Vector3(0x1.a2272ep-3f, -0x1.9421cep-3f, 0x1.324d5cp-1f),
         Vector3(0x1.e1cfep-3f, 0x1.a21788p-5f, 0x1.df974cp-3f),
         Vector3(-0x1.645a1cp-4f, -0x1.5798eep-26f, 0x1.e66666p-1f),
         Vector3(-0x1.a2272ep-3f, -0x1.9421cep-3f, 0x1.324d5cp-1f),
         Vector3(-0x1.e1cfc4p-3f, 0x1.a218a8p-5f, 0x1.df977cp-3f)},
        {Vector3(0.0f, 0.0f, 0x1.eb851ep-1f), Vector3(0.0f, 0.0f, 0x1.1c28f6p+0f),
         Vector3(0.0f, -0x1.eb851ep-6f, 0x1.9c28f6p+0f),
         Vector3(0x1.47ae14p-3f, -0x1.47ae14p-7f, 0x1.970a3ep+0f),
         Vector3(0x1.333332p-3f, -0x1.47ae14p-7f, 0x1.428f5cp+0f),
         Vector3(-0x1.47ae14p-3f, -0x1.47ae14p-7f, 0x1.970a3ep+0f),
         Vector3(-0x1.333332p-3f, -0x1.47ae14p-7f, 0x1.428f5cp+0f),
         Vector3(0x1.645a1cp-4f, 0.0f, 0x1.e66666p-1f),
         Vector3(0x1.645a1cp-4f, 0.0f, 0x1.0f5c28p-1f),
         Vector3(0x1.645a1cp-4f, -0x1.47ae14p-5f, 0x1.70a3dp-4f),
         Vector3(-0x1.645a1cp-4f, 0.0f, 0x1.e66666p-1f),
         Vector3(-0x1.645a1cp-4f, 0.0f, 0x1.0f5c28p-1f),
         Vector3(-0x1.645a1cp-4f, -0x1.47ae14p-5f, 0x1.70a3dp-4f)}};
    return pose;
}

bool PositionLess::operator()(const Vector3 &a, const Vector3 &b) const {
    if (a.coordinates[0] == b.coordinates[0]) {
        if (a.coordinates[1] == b.coordinates[1]) return a.coordinates[2] < b.coordinates[2];
        return a.coordinates[1] < b.coordinates[1];
    }
    return a.coordinates[0] < b.coordinates[0];
}

size_t SkinnedModel::BufferFloats() const {
    size_t floats = 0;
    for (const SkinSubmesh &submesh : submeshes) floats += submesh.source.vertices.size();
    return floats;
}

bool PrepareFullbodyModel(const std::vector<PlanarMesh> &meshes, const VertexColorMap &colors,
                          float zMultiplier, const BindPose &pose, SkinnedModel &model, bool bake) {
    model = SkinnedModel();
    model.zMultiplier = zMultiplier;
    for (const PlanarMesh &mesh : meshes) {
        const int elementOffset = (int)(mesh.vertices.size() / MESH_ELEMENTS);
        const float *data = mesh.vertices.data();
        std::vector<int> uniqueOf;
        std::vector<uint32_t> indices;
        indices.reserve(elementOffset / 3);
        for (int v = 0; v < elementOffset; v += 3) {
            int index = -1;
            for (size_t u = 0; u < uniqueOf.size(); u++) {
                const float *a = &data[uniqueOf[u]];
                const float *t = &data[uniqueOf[u] + 2 * elementOffset];
                if (a[0] == data[v] && a[1] == data[v + 1] && a[2] == data[v + 2] &&
                    t[0] == data[v + 2 * elementOffset] &&
                    t[1] == data[v + 2 * elementOffset + 1] &&
                    t[2] == data[v + 2 * elementOffset + 2]) {
                    index = (int)u;
                    break;
                }
            }
            if (index < 0) {
                uniqueOf.push_back(v);
                index = (int)uniqueOf.size() - 1;
            }
            indices.push_back((uint32_t)index);
        }
        SkinSubmesh submesh;
        const int unique = (int)uniqueOf.size();
        const size_t uniqueElementOffset = (size_t)3 * unique;
        submesh.source.vertices.resize(uniqueElementOffset * MESH_ELEMENTS);
        submesh.source.indices = indices;
        float *out = submesh.source.vertices.data();
        for (int u = 0; u < unique; u++)
            for (int e = 0; e < MESH_ELEMENTS; e++)
                for (int c = 0; c < 3; c++)
                    out[u * 3 + e * uniqueElementOffset + c] =
                        data[uniqueOf[u] + e * elementOffset + c];
        submesh.weights.resize(unique);
        for (int u = 0; u < unique; u++) {
            const Vector3 position(out[u * 3], out[u * 3 + 1], out[u * 3 + 2]);
            auto found = colors.find(position);
            if (found == colors.end()) return false;
            const Vector3 &color = found->second;
            out[u * 3] *= zMultiplier;
            out[u * 3 + 1] *= zMultiplier;
            out[u * 3 + 2] *= zMultiplier;
            float totalWeight = 0.0f;
            SkinBone bones[3];
            for (int c = 0; c < 3; c++) {
                const int jointID = (int)std::floor(color.coordinates[c] * 0.1);
                const float weight = (float)((color.coordinates[c] - jointID * 10.0) / 9.0);
                bones[c] = SkinBone{jointID, weight};
                totalWeight += weight;
            }
            if (bones[0].weight == 0.0f) return false;
            SkinVertex &weighted = submesh.weights[u];
            weighted.boneCount = 0;
            for (int c = 0; c < 3; c++) {
                if (bones[c].weight > 0.01f) {
                    if (bones[c].joint < 0 || bones[c].joint >= RENDER_JOINTS) return false;
                    bones[c].weight /= totalWeight;
                    weighted.bones[weighted.boneCount++] = bones[c];
                }
            }
        }
        model.submeshes.push_back(std::move(submesh));
    }
    Joint joints[RENDER_JOINTS];
    for (int j = 0; j < RENDER_JOINTS; j++) {
        joints[j].orientation = pose.inverseRotation[j];
        joints[j].position = pose.straightPosition[j];
    }
    for (SkinSubmesh &submesh : model.submeshes) {
        if (!bake) continue;
        const std::vector<float> before = submesh.source.vertices;
        SkinSubmeshInto(submesh, joints, pose.basePosition, zMultiplier, before.data(),
                        submesh.source.vertices.data(), (size_t)3 * submesh.source.VertexCount());
    }
    for (int j = 0; j < RENDER_JOINTS; j++) model.origin[j] = pose.straightPosition[j];
    return true;
}

void SkinFullbody(const SkinnedModel &model, const RenderJoint renderJoints[RENDER_JOINTS],
                  float *buffer) {
    Joint joints[RENDER_JOINTS];
    for (int j = 0; j < RENDER_JOINTS; j++) {
        joints[j].orientation = renderJoints[j].orientation;
        joints[j].position = renderJoints[j].position;
    }
    const size_t stride = model.BufferFloats() / MESH_ELEMENTS;
    size_t start = 0;
    for (const SkinSubmesh &submesh : model.submeshes) {
        const size_t elementFloats = (size_t)3 * submesh.source.VertexCount();
        SkinSubmeshInto(submesh, joints, model.origin, model.zMultiplier,
                        submesh.source.vertices.data(), buffer + start, stride);
        std::memcpy(buffer + start + 2 * stride, submesh.source.vertices.data() + 2 * elementFloats,
                    elementFloats * sizeof(float));
        start += elementFloats;
    }
}

void FullbodyRestBuffer(const SkinnedModel &model, float *buffer) {
    const size_t stride = model.BufferFloats() / MESH_ELEMENTS;
    size_t start = 0;
    for (const SkinSubmesh &submesh : model.submeshes) {
        const size_t elementFloats = (size_t)3 * submesh.source.VertexCount();
        for (int e = 0; e < MESH_ELEMENTS; e++)
            std::memcpy(buffer + start + e * stride,
                        submesh.source.vertices.data() + e * elementFloats,
                        elementFloats * sizeof(float));
        start += elementFloats;
    }
}

}  // namespace scene
}  // namespace render
}  // namespace cuju
