// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// ASELoader of loaders/aseloader.cpp (GRF, Apache-2.0) over base/utils.cpp's tree_readblock, with
// Triangle::CalculateTangents of base/geometry/triangle.cpp. Numbers parse as glibc's atof, a
// double then rounded to float, and every float operation keeps GRF's order.

#include <cmath>
#include <memory>

#include "assets/assets.h"
#include "assets/text.h"
#include "cuju/vector3.h"

namespace cuju::render::assets {
namespace {
struct Tree;
struct TreeEntry {
    std::string name;
    std::vector<std::string> values;
    std::unique_ptr<Tree> subtree;
};
struct Tree {
    std::vector<TreeEntry> entries;
};

std::string Chomp(const std::string &input, char chomp) {
    const std::size_t at = input.find_first_not_of(chomp);
    if (at < input.length()) return input.substr(at);
    return "";
}

std::unique_ptr<Tree> ReadBlock(const std::string &data, std::size_t &pos) {
    auto content = std::make_unique<Tree>();
    bool quit = false;
    while (pos < data.size() && !quit) {
        const std::size_t start = pos;
        while (pos < data.size() && data[pos] != '\n') pos++;
        std::string line = data.substr(start, pos - start);
        if (pos < data.size()) pos++;
        const std::size_t nul = line.find('\0');
        if (nul != std::string::npos) line.resize(nul);
        if (line.length() > 1 && line[line.length() - 1] == '\r') line.pop_back();
        line = Chomp(line, '\t');
        line = Chomp(line, ' ');
        std::vector<std::string> tokens;
        text::Tokenize(line, tokens, " \t");
        if (tokens.empty()) continue;
        if (tokens[0] == "}") {
            quit = true;
            continue;
        }
        TreeEntry entry;
        entry.name = tokens[0].compare(0, 1, "*") == 0 ? tokens[0].substr(1) : tokens[0];
        for (std::size_t i = 1; i < tokens.size(); i++) entry.values.push_back(tokens[i]);
        if (tokens.back() == "{") {
            entry.values.pop_back();
            entry.subtree = ReadBlock(data, pos);
        }
        content->entries.push_back(std::move(entry));
    }
    return content;
}

const TreeEntry *FindEntry(const Tree *tree, const char *needle) {
    for (const TreeEntry &entry : tree->entries)
        if (entry.name == needle) return &entry;
    return nullptr;
}

const Tree *FindTree(const Tree *tree, const char *needle) {
    for (const TreeEntry &entry : tree->entries)
        if (entry.name == needle) return entry.subtree.get();
    return nullptr;
}

struct AseMaterial {
    std::string maps[4];
    std::string shininess;
    std::string specular_amount;
    Vec3 self_illumination;
};

struct Failure {
    std::string message;
};

[[noreturn]] void Fail(const std::string &message) { throw Failure{message}; }

const std::string &Value(const TreeEntry *entry, std::size_t index) {
    if (!entry || index >= entry->values.size()) Fail("missing value");
    return entry->values[index];
}

float ParseFloat(const std::string &s) { return (float)text::Atof(s); }

struct Tri {
    cuju::Vector3 vertices[3];
    cuju::Vector3 tangents[3];
    cuju::Vector3 bitangents[3];
    cuju::Vector3 texcoords[3];
    cuju::Vector3 normals[3];

    void CalculateTangents() {
        const cuju::Vector3 edge1 = vertices[1] - vertices[0];
        const cuju::Vector3 edge2 = vertices[2] - vertices[0];
        const cuju::Vector3 edge1uv = texcoords[1] - texcoords[0];
        const cuju::Vector3 edge2uv = texcoords[2] - texcoords[0];
        const float cp = edge1uv.coordinates[1] * edge2uv.coordinates[0] -
                         edge1uv.coordinates[0] * edge2uv.coordinates[1];
        if (cp != 0.0f) {
            const float mul = 1.0f / cp;
            for (int v = 0; v < 3; v++) {
                tangents[v] =
                    (edge1 * -edge2uv.coordinates[1] + edge2 * edge1uv.coordinates[1]) * mul;
                bitangents[v] =
                    (edge1 * -edge2uv.coordinates[0] + edge2 * edge1uv.coordinates[0]) * mul;
                tangents[v].Normalize();
                bitangents[v].Normalize();
                const cuju::Vector3 cross = normals[v].GetCrossProduct(tangents[v]);
                const float dot = cross.GetDotProduct(bitangents[v]);
                if (dot > 0.0f) bitangents[v] = -bitangents[v];
            }
        } else {
            for (int v = 0; v < 3; v++) {
                tangents[v].Set(1, 0, 0);
                bitangents[v].Set(0, 1, 0);
            }
        }
    }
};

void NormalizeRow(float v[3]) {
    const float dot = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const float f = 1.0f / std::sqrt(dot);
    v[0] *= f;
    v[1] *= f;
    v[2] *= f;
}

void SetMap(std::string *name, std::string *path, const std::string &fetched) {
    *path = fetched;
    *name = text::FileName(fetched);
}

TriangleMesh BuildTriangleMesh(const Tree *data, const std::vector<AseMaterial> &materials) {
    const Tree *nodeTm = FindTree(data, "NODE_TM");
    if (!nodeTm) Fail("subtree NODE_TM not found");
    const Tree *mesh = FindTree(data, "MESH");
    if (!mesh) Fail("subtree MESH not found");

    const TreeEntry *entry = FindEntry(mesh, "MESH_NUMVERTEX");
    if (!entry) Fail("tree entry MESH_NUMVERTEX not found");
    const int numvertex = text::Atoi(Value(entry, 0));
    entry = FindEntry(mesh, "MESH_NUMTVERTEX");
    if (!entry) Fail("tree entry MESH_NUMTVERTEX not found");
    const int numtvertex = text::Atoi(Value(entry, 0));
    entry = FindEntry(mesh, "MESH_NUMFACES");
    if (!entry) Fail("tree entry MESH_NUMFACES not found");
    const int numfaces = text::Atoi(Value(entry, 0));
    int numtfaces = 0;
    if (numtvertex > 0) {
        entry = FindEntry(mesh, "MESH_NUMTVFACES");
        if (!entry) Fail("tree entry MESH_NUMTFACES not found");
        numtfaces = text::Atoi(Value(entry, 0));
    }

    const Tree *vertexList = FindTree(mesh, "MESH_VERTEX_LIST");
    if (!vertexList) Fail("subtree MESH_VERTEX_LIST not found");
    const Tree *faceList = FindTree(mesh, "MESH_FACE_LIST");
    if (!faceList) Fail("subtree MESH_FACE_LIST not found");
    const Tree *tvertexList = nullptr;
    if (numtvertex > 0) {
        tvertexList = FindTree(mesh, "MESH_TVERTLIST");
        if (!tvertexList) Fail("subtree MESH_TVERTLIST not found");
    }
    const Tree *tfaceList = nullptr;
    if (numtfaces > 0) {
        tfaceList = FindTree(mesh, "MESH_TFACELIST");
        if (!tfaceList) Fail("subtree MESH_TFACELIST not found");
    }
    const Tree *normals = FindTree(mesh, "MESH_NORMALS");
    if (!normals) Fail("subtree MESH_NORMALS not found");

    if (numvertex < 0 || vertexList->entries.size() != (unsigned int)numvertex)
        Fail("numvertex and the vertex list differ");
    std::vector<cuju::Vector3> vertexCache((std::size_t)numvertex);
    for (const TreeEntry &e : vertexList->entries) {
        const int index = text::Atoi(Value(&e, 0));
        if (index < 0 || index >= numvertex) Fail("vertex index larger than numvertex");
        vertexCache[index].coordinates[0] = ParseFloat(Value(&e, 1));
        vertexCache[index].coordinates[1] = ParseFloat(Value(&e, 2));
        vertexCache[index].coordinates[2] = ParseFloat(Value(&e, 3));
    }

    if (numfaces < 0 || faceList->entries.size() != (unsigned int)numfaces)
        Fail("numfaces and the face list differ");
    std::vector<Tri> triangles(faceList->entries.size());
    for (std::size_t i = 0; i < faceList->entries.size(); i++) {
        const TreeEntry &e = faceList->entries[i];
        if (text::Atoi(Value(&e, 0)) >= numfaces) Fail("face index larger than numfaces");
        for (int v = 0; v < 3; v++) {
            const int index = text::Atoi(Value(&e, 2 + 2 * v));
            if (index < 0 || index >= numvertex) Fail("face vertex out of range");
            triangles[i].vertices[v] = vertexCache[index];
        }
    }

    if (nodeTm->entries.size() < 7) Fail("NODE_TM rows missing");
    float rotation[9];
    for (int row = 0; row < 3; row++)
        for (int c = 0; c < 3; c++)
            rotation[row * 3 + c] = ParseFloat(Value(&nodeTm->entries[4 + row], c));
    NormalizeRow(&rotation[0]);
    NormalizeRow(&rotation[3]);
    NormalizeRow(&rotation[6]);

    if (normals->entries.size() / 4 != (unsigned int)numfaces)
        Fail("numfaces and the normal list differ");
    for (std::size_t i = 0; i < normals->entries.size(); i += 4) {
        const TreeEntry &faceNormal = normals->entries[i];
        const int face = text::Atoi(Value(&faceNormal, 0));
        if (face < 0 || (std::size_t)face >= triangles.size()) Fail("normal face out of range");
        for (int v = 0; v < 3; v++) {
            if (i + v + 1 >= normals->entries.size()) Fail("vertex normal missing");
            const TreeEntry &vertexNormal = normals->entries[i + v + 1];
            const float x = ParseFloat(Value(&vertexNormal, 1));
            const float y = ParseFloat(Value(&vertexNormal, 2));
            const float z = ParseFloat(Value(&vertexNormal, 3));
            cuju::Vector3 normal(x * rotation[0] + y * rotation[3] + z * rotation[6],
                                 x * rotation[1] + y * rotation[4] + z * rotation[7],
                                 x * rotation[2] + y * rotation[5] + z * rotation[8]);
            normal.Normalize();
            triangles[face].normals[v] = normal;
        }
    }

    if (numtvertex > 0) {
        if (tvertexList->entries.size() != (unsigned int)numtvertex)
            Fail("numtvertex and the texture vertex list differ");
        std::vector<cuju::Vector3> tvertexCache((std::size_t)numtvertex);
        for (const TreeEntry &e : tvertexList->entries) {
            const int index = text::Atoi(Value(&e, 0));
            if (index < 0 || index >= numtvertex) Fail("tvertex index larger than numtvertex");
            tvertexCache[index].coordinates[0] = ParseFloat(Value(&e, 1));
            tvertexCache[index].coordinates[1] = (float)-text::Atof(Value(&e, 2));
            tvertexCache[index].coordinates[2] = ParseFloat(Value(&e, 3));
        }
        if (numtfaces > 0) {
            if (tfaceList->entries.size() != (unsigned int)numtfaces)
                Fail("numtfaces and the texture face list differ");
            for (const TreeEntry &e : tfaceList->entries) {
                const int face = text::Atoi(Value(&e, 0));
                if (face >= numtfaces) Fail("tface index larger than numtfaces");
                if (face < 0 || (std::size_t)face >= triangles.size()) Fail("tface out of range");
                for (int v = 0; v < 3; v++) {
                    const int index = text::Atoi(Value(&e, 1 + v));
                    if (index < 0 || index >= numtvertex) Fail("tface vertex out of range");
                    triangles[face].texcoords[v] = tvertexCache[index];
                }
            }
        }
    }

    TriangleMesh result;
    const std::size_t count = triangles.size();
    result.vertices.assign(count * 3 * 3 * 5, 0.0f);
    float *raw = result.vertices.data();
    for (std::size_t t = 0; t < count; t++) {
        Tri &triangle = triangles[t];
        triangle.CalculateTangents();
        for (int v = 0; v < 3; v++) {
            const cuju::Vector3 *elements[5] = {&triangle.vertices[v], &triangle.normals[v],
                                                &triangle.texcoords[v], &triangle.tangents[v],
                                                &triangle.bitangents[v]};
            for (int e = 0; e < 5; e++)
                for (int c = 0; c < 3; c++)
                    raw[e * count * 9 + t * 9 + v * 3 + c] = elements[e]->coordinates[c];
        }
    }

    int reference = -1;
    if (const TreeEntry *ref = FindEntry(data, "MATERIAL_REF"))
        reference = text::Atoi(Value(ref, 0));
    if (reference != -1) {
        if (reference < 0 || (std::size_t)reference >= materials.size())
            Fail("MATERIAL_REF out of range");
        const AseMaterial &source = materials[reference];
        Material &material = result.material;
        SetMap(&material.diffuse, &material.diffuse_path, source.maps[0]);
        if (!source.maps[1].empty())
            SetMap(&material.normal, &material.normal_path, source.maps[1]);
        if (!source.maps[2].empty())
            SetMap(&material.specular, &material.specular_path, source.maps[2]);
        if (!source.maps[3].empty())
            SetMap(&material.illumination, &material.illumination_path, source.maps[3]);
        material.shininess = (float)text::Atof(source.shininess);
        material.specular_amount = (float)text::Atof(source.specular_amount);
        material.self_illumination = source.self_illumination;
    }
    return result;
}

void Build(const Tree *data, GeometryData *geometry) {
    const Tree *scene = FindTree(data, "SCENE");
    if (!scene) Fail("SCENE not found");
    const Tree *materialList = FindTree(data, "MATERIAL_LIST");
    if (!materialList || materialList->entries.empty()) Fail("MATERIAL_LIST not found");
    const int materialCount = text::Atoi(Value(&materialList->entries[0], 0));
    std::vector<AseMaterial> materials;
    for (int m = 0; m < materialCount; m++) {
        if ((std::size_t)m + 1 >= materialList->entries.size() ||
            !materialList->entries[m + 1].subtree)
            Fail("material missing");
        const Tree *tree = materialList->entries[m + 1].subtree.get();
        const TreeEntry *shine = FindEntry(tree, "MATERIAL_SHINE");
        const TreeEntry *shineStrength = FindEntry(tree, "MATERIAL_SHINESTRENGTH");
        const TreeEntry *selfIllumination = FindEntry(tree, "MATERIAL_SELFILLUM");
        const char *mapNames[4] = {"MAP_DIFFUSE", "MAP_BUMP", "MAP_SHINE", "MAP_SELFILLUM"};
        AseMaterial material;
        for (int i = 0; i < 4; i++) {
            if (const Tree *map = FindTree(tree, mapNames[i])) {
                const std::string &bitmap = Value(FindEntry(map, "BITMAP"), 0);
                if (bitmap.length() < 2) Fail("BITMAP name");
                material.maps[i] = bitmap.substr(1, bitmap.length() - 2);
            } else if (i == 0) {
                material.maps[i] = "orange.jpg";
            }
        }
        material.shininess = Value(shine, 0);
        material.specular_amount = Value(shineStrength, 0);
        const float illumination = (float)text::Atof(Value(selfIllumination, 0));
        material.self_illumination.v[0] = illumination;
        material.self_illumination.v[1] = illumination;
        material.self_illumination.v[2] = illumination;
        materials.push_back(material);
    }
    for (const TreeEntry &entry : data->entries) {
        if (entry.name != "GEOMOBJECT") continue;
        if (entry.subtree && !entry.subtree->entries.empty())
            geometry->meshes.push_back(BuildTriangleMesh(entry.subtree.get(), materials));
    }
}

}

bool LoadAse(const std::string &data_dir, const std::string &path, GeometryData *data,
             std::string *error) {
    std::string contents;
    if (!text::ReadFile(text::DataPath(data_dir, path), &contents, error)) return false;
    std::size_t pos = 0;
    const std::unique_ptr<Tree> tree = ReadBlock(contents, pos);
    GeometryData result;
    try {
        Build(tree.get(), &result);
    } catch (const Failure &failure) {
        if (error) *error = path + ": " + failure.message;
        return false;
    }
    *data = std::move(result);
    return true;
}

}
