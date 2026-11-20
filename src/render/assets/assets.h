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

// An SDL surface after IMG_LoadBmp's conversion: SDL_PIXELFORMAT_ABGR8888, bytes R G B A, when
// the BMP has an alpha mask, otherwise SDL_PIXELFORMAT_RGB24, bytes R G B. Rows top first, each
// `pitch` bytes, SDL's pitch: width times bytes per pixel rounded up to 4.
struct Surface {
    int w = 0;
    int h = 0;
    int pitch = 0;
    bool alpha = false;
    std::vector<std::uint8_t> pixels;  // pitch * h
};

// IMG_LoadBmp(path): reads `path` minus its last four characters plus ".bmp".
bool LoadSurface(const std::string &path, Surface *surface, std::string *error);

// GRF's float vector and its AABB with Reset's quirky start values.
struct Vec3 {
    float v[3] = {0, 0, 0};
};
struct Aabb {
    Vec3 min, max;
    void Reset();  // min = FLT_MAX / 100, max = FLT_MIN / 100, as GRF's AABB::Reset
};

// A material as GRF's Material holds it. Textures are named by the identity GRF's resource manager
// gives the surface, the file name without its directory ("skin.jpg"), and `*_path` is the path
// GRF fetched it with, relative to GRF's data directory
// ("media/objects/players/textures/skin.jpg"). Empty when the map is absent.
struct Material {
    std::string diffuse, diffuse_path;
    std::string normal, normal_path;
    std::string specular, specular_path;
    std::string illumination, illumination_path;
    float shininess = 0.0f;
    float specular_amount = 0.0f;
    Vec3 self_illumination;
};

// MaterializedTriangleMesh: vertices planar, five elements (position, normal, texcoord, tangent,
// bitangent), each 3 floats per vertex, so vertices.size() is verticesDataSize.
struct TriangleMesh {
    Material material;
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;  // empty for unindexed meshes
};

struct GeometryData {
    std::vector<TriangleMesh> meshes;
};

// GetTriangleMeshAABB, with GRF's indexing of indexed meshes kept as it is.
Aabb TriangleMeshAabb(const TriangleMesh &mesh);
// GeometryData::GetAABB: the union, by AABB::operator+=, of the meshes' AABBs from a reset AABB.
Aabb GeometryAabb(const GeometryData &data);

// ASELoader::Load of `path` under `data_dir` (GRF's data directory, holding media/).
bool LoadAse(const std::string &data_dir, const std::string &path, GeometryData *data,
             std::string *error);

// An object file as ObjectLoader::LoadObjectImpl walks it.
struct ObjectGeometry {
    std::string filename;  // the ase path relative to the data directory, dirpart included
    std::string name;
    Vec3 position;
    float rotation[4] = {0, 0, 0, 1};  // GetQuaternionFromString, x y z w
    bool dynamic = false;              // properties "dynamic"
    bool absolute = false;             // localmode absolute
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
    std::string name;  // "objectnode: <path>" until a name element renames it
    bool has_position = false, has_rotation = false;
    Vec3 position;
    float rotation[4] = {0, 0, 0, 1};
    // Children in the order LoadObjectImpl visits them: each entry is a node, a geometry or a
    // light. GRF's XML tree keeps children in a std::multimap keyed by tag, so entries come
    // sorted by tag, geometries before lights before nodes, in file order within one tag.
    struct Entry {
        enum Kind : std::uint8_t { kNode, kGeometry, kLight } kind;
        int index;  // into nodes, geometries or lights
    };
    std::vector<Entry> entries;
    std::vector<std::unique_ptr<ObjectNode>> nodes;
    std::vector<ObjectGeometry> geometries;
    std::vector<ObjectLight> lights;
};
bool LoadObject(const std::string &data_dir, const std::string &path, ObjectNode *node,
                std::string *error);

// SplitGeometry(source, 24): the grid cells in creation order. Each cell's GeometryData is named
// "<source name> gridGeomData @ <x>, <y>" and its Geometry "<source name> gridGeom @ <x>, <y>".
struct GridCell {
    std::string data_name;
    std::string geometry_name;
    GeometryData data;
};
std::vector<GridCell> SplitGeometry(const std::string &source_name, const GeometryData &source,
                                    float grid_size);

// GraphicsGeometry_GeometryInterpreter::OnLoad's concatenation: the vertex buffer floats, the
// element indices (GRF generates 0..n-1 when the meshes have none), and one index range per mesh.
struct IndexRange {
    int start = 0;
    int size = 0;
};
struct VertexBufferData {
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;  // what CreateVertexBuffer uploads
    std::vector<IndexRange> ranges;      // vertexBufferIndices
    bool uses_indices = false;
};
VertexBufferData ConcatenateMeshes(const GeometryData &data);

// PrepareFullbodyModel's indexed mesh before skinning: per mesh, the unique vertices (equal
// position and equal texcoord) with positions times `z_multiplier`, and their indices.
GeometryData IndexFullbody(const GeometryData &fullbody, float z_multiplier);

}  // namespace cuju::render::assets

#endif  // CUJU_RENDER_ASSETS_ASSETS_H
