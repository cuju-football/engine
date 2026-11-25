// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GRF's media as its loaders read them (GRF, Apache-2.0): images as IMG_LoadBmp leaves its SDL
// surfaces, ASE models as ASELoader builds their triangle meshes with GRF's tangent rule, object
// files as ObjectLoader walks their XML, and the pure geometry steps that follow loading,
// SplitGeometry's 24 m grid, the vertex buffer that GraphicsGeometry concatenates, and the indexed
// mesh of PrepareFullbodyModel before skinning.

#ifndef CUJU_RENDER_ASSETS_ASSETS_H
#define CUJU_RENDER_ASSETS_ASSETS_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuju::render::assets {
struct Surface {
    int w = 0;
    int h = 0;
    int pitch = 0;
    bool alpha = false;
    std::vector<std::uint8_t> pixels;
};

bool LoadSurface(const std::string &path, Surface *surface, std::string *error);

// GRF's float vector and its AABB with Reset's quirky start values.
struct Vec3 {
    float v[3] = {0, 0, 0};
};
struct Aabb {
    Vec3 min, max;
    void Reset();
};

struct Material {
    std::string diffuse, diffuse_path;
    std::string normal, normal_path;
    std::string specular, specular_path;
    std::string illumination, illumination_path;
    float shininess = 0.0f;
    float specular_amount = 0.0f;
    Vec3 self_illumination;
};

struct TriangleMesh {
    Material material;
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;
};

struct GeometryData {
    std::vector<TriangleMesh> meshes;
};

Aabb TriangleMeshAabb(const TriangleMesh &mesh);
Aabb GeometryAabb(const GeometryData &data);

bool LoadAse(const std::string &data_dir, const std::string &path, GeometryData *data,
             std::string *error);

struct ObjectGeometry {
    std::string filename;
    std::string name;
    Vec3 position;
    float rotation[4] = {0, 0, 0, 1};
    bool dynamic = false;
    bool absolute = false;
};
struct ObjectLight {
    std::string name;
    Vec3 position;
    Vec3 color;
    float radius = 0.0f;
    bool directional = false;
    bool shadow = false;
    bool absolute = false;
};
struct ObjectNode {
    std::string name;
    bool has_position = false, has_rotation = false;
    Vec3 position;
    float rotation[4] = {0, 0, 0, 1};
    struct Entry {
        enum Kind : std::uint8_t { kNode, kGeometry, kLight } kind;
        int index;
    };
    std::vector<Entry> entries;
    std::vector<std::unique_ptr<ObjectNode>> nodes;
    std::vector<ObjectGeometry> geometries;
    std::vector<ObjectLight> lights;
};
bool LoadObject(const std::string &data_dir, const std::string &path, ObjectNode *node,
                std::string *error);

struct GridCell {
    std::string data_name;
    std::string geometry_name;
    GeometryData data;
};
std::vector<GridCell> SplitGeometry(const std::string &source_name, const GeometryData &source,
                                    float grid_size);

struct IndexRange {
    int start = 0;
    int size = 0;
};
struct VertexBufferData {
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<IndexRange> ranges;
    bool uses_indices = false;
};
VertexBufferData ConcatenateMeshes(const GeometryData &data);

GeometryData IndexFullbody(const GeometryData &fullbody, float z_multiplier);

}

#endif
