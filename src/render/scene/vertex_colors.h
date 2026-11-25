// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// GRF's GetVertexColors of gamedefines.cpp, Apache-2.0: the vertex colours of the full body mesh,
// which name each vertex's bones.

#ifndef CUJU_RENDER_SCENE_VERTEX_COLORS_H
#define CUJU_RENDER_SCENE_VERTEX_COLORS_H

#include <string>

#include "skinning.h"

namespace cuju::render::scene {
bool LoadVertexColors(const std::string &data_dir, VertexColorMap *colors, std::string *error);

}

#endif
