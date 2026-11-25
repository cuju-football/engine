// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "vertex_colors.h"

#include <cmath>
#include <vector>

#include "../assets/text.h"

namespace cuju::render::scene {
namespace {
struct Index3 {
    int index[3];
};

}

bool LoadVertexColors(const std::string &data_dir, VertexColorMap *colors, std::string *error) {
    std::string data;
    std::string path =
        assets::text::DataPath(data_dir, "media/objects/players/models/fullbody.ase");
    if (!assets::text::ReadFile(path, &data, error)) return false;
    std::vector<Vector3> vertices, vertexColors;
    std::vector<Index3> faces, colorFaces;
    std::size_t pos = 0, last = 0;
    std::vector<std::string> tokens;
    while (pos < data.size()) {
        while (pos < data.size() && data[pos] != '\n') pos++;
        std::string line = data.substr(last, pos - last);
        pos++;
        last = pos;
        tokens.clear();
        assets::text::Tokenize(line, tokens, " \t");
        if (tokens.empty()) continue;
        const std::string &key = tokens[0];
        if (key == "*MESH_NORMALS") {
            // GRF: the colours of an object complete at its normals block.
            for (std::size_t i = 0; i < colorFaces.size(); i++)
                for (int v = 0; v < 3; v++) {
                    const Vector3 &coord = vertices.at(faces.at(i).index[v]);
                    const Vector3 &color = vertexColors.at(colorFaces[i].index[v]);
                    colors->insert({coord, color});
                }
            vertices.clear();
            vertexColors.clear();
            faces.clear();
            colorFaces.clear();
        }
        if (key == "*MESH_VERTEX" && tokens.size() > 4) {
            vertices.emplace_back(static_cast<float>(assets::text::Atof(tokens[2])),
                                  static_cast<float>(assets::text::Atof(tokens[3])),
                                  static_cast<float>(assets::text::Atof(tokens[4])));
        }
        if (key == "*MESH_FACE" && tokens.size() > 7) {
            faces.push_back({{assets::text::Atoi(tokens[3]), assets::text::Atoi(tokens[5]),
                              assets::text::Atoi(tokens[7])}});
        }
        if (key == "*MESH_VERTCOL" && tokens.size() > 4) {
            Vector3 color(static_cast<float>(assets::text::Atof(tokens[2])),
                          static_cast<float>(assets::text::Atof(tokens[3])),
                          static_cast<float>(assets::text::Atof(tokens[4])));
            color *= 255.0f;
            for (float &c : color.coordinates) c = std::round(c);
            vertexColors.push_back(color);
        }
        if (key == "*MESH_CFACE" && tokens.size() > 4) {
            colorFaces.push_back({{assets::text::Atoi(tokens[2]), assets::text::Atoi(tokens[3]),
                                   assets::text::Atoi(tokens[4])}});
        }
    }
    if (colors->empty()) {
        *error = "no vertex colours in " + path;
        return false;
    }
    return true;
}

}
