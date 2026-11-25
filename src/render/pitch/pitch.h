// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GRF's procedural pitch of onthepitch/proceduralpitch.cpp, Apache-2.0. It makes the diffuse,
// specular and normal maps of the four pitch quadrants bit for bit, with the draws of GRF's
// non-deterministic Mersenne Twister and of libc `rand` in GRF's order. Threads split the work by
// columns and each starts its random stream at the draw GRF reaches there, so every thread count
// gives the same bytes.

#ifndef CUJU_RENDER_PITCH_PITCH_H
#define CUJU_RENDER_PITCH_PITCH_H

#include <cstdint>
#include <string>
#include <vector>

#include "bmp.h"

namespace cuju::render::pitch {
struct PitchSizes {
    int diffuseWidth = 2048, diffuseHeight = 1024;
    int specularWidth = 1024, specularHeight = 512;
    int normalWidth = 2048, normalHeight = 1024;
};

// GRF's state when the first rendering match of a process builds the pitch.
struct PitchOptions {
    PitchSizes sizes;
    std::uint32_t seed = 42;
    int drawsBefore = 23;
    float repeatMultiplier = 1.0f;
    float redToBlueRatio = 0.5f;
    int threads = 1;
};

struct PitchMedia {
    Image seamless;
    Image overlay;
};

struct PitchTextures {
    PitchSizes sizes;
    std::vector<std::uint8_t> diffuse[4];
    std::vector<std::uint8_t> specular[4];
    std::vector<std::uint8_t> normal[4];
};

bool LoadPitchMedia(const std::string &mediaDirectory, PitchMedia *media, std::string *error);

bool GeneratePitch(const PitchMedia &media, const PitchOptions &options, PitchTextures *textures,
                   std::string *error);

}

#endif
