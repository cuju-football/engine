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

// The sizes of GRF's call GeneratePitch(2048, 1024, 1024, 512, 2048, 1024).
struct PitchSizes {
    int diffuseWidth = 2048, diffuseHeight = 1024;
    int specularWidth = 1024, specularHeight = 512;
    int normalWidth = 2048, normalHeight = 1024;
};

// GRF's state when the first rendering match of a process builds the pitch. That match is the
// internal seed-42 match of GameEnv::start_game, so the defaults hold for every GRF process.
struct PitchOptions {
    PitchSizes sizes;
    // `randomize(seed)` of StartMatch seeds both Mersenne Twisters and `srand`, then drops one
    // value of `rand`.
    std::uint32_t seed = 42;
    // Draws of rng_non_deterministic before the pitch, one for each of the 23 adboards of
    // RandomizeAdboards.
    int drawsBefore = 23;
    // grassNormalRepeatMultiplier, the pitch's one draw from the deterministic generator.
    float repeatMultiplier = 1.0f;
    // The configuration value graphics_pitchredtoblueratio, which GRF leaves at its default.
    float redToBlueRatio = 0.5f;
    int threads = 1;
};

// The two source images, decoded as GRF reads them through SDL.
struct PitchMedia {
    Image seamless;  // seamlessgrass08.bmp, 1200x1200
    Image overlay;   // overlay.bmp, 4096x2048, the pitch lines with alpha
};

// Chunk i is GRF's chunk i + 1, the textures pitch_0N, pitch_specular_0N and pitch_normal_0N with
// N = i + 1. Each surface is RGBA8 in upload order, row 0 first, alpha 255, which is the buffer GRF
// passes to glTexSubImage2D with GL_RGBA and GL_UNSIGNED_BYTE.
struct PitchTextures {
    PitchSizes sizes;
    std::vector<std::uint8_t> diffuse[4];
    std::vector<std::uint8_t> specular[4];
    std::vector<std::uint8_t> normal[4];
};

// Reads textures/pitch/seamlessgrass08.bmp and textures/pitch/overlay.bmp under GRF's media
// directory, the directory GRF names "media".
bool LoadPitchMedia(const std::string &mediaDirectory, PitchMedia *media, std::string *error);

// False when the images do not have GRF's sizes.
bool GeneratePitch(const PitchMedia &media, const PitchOptions &options, PitchTextures *textures,
                   std::string *error);

}  // namespace cuju::render::pitch

#endif  // CUJU_RENDER_PITCH_PITCH_H
