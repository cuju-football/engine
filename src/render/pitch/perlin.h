// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GRF's Perlin noise of misc/perlin.cpp, Apache-2.0, after Ken Perlin's reference code, and the
// libc `rand` that fills its tables, the TYPE_3 additive generator of glibc's srandom_r and
// random_r.

#ifndef CUJU_RENDER_PITCH_PERLIN_H
#define CUJU_RENDER_PITCH_PERLIN_H

#include <cstdint>

namespace cuju::render::pitch {

// glibc's `rand` after `srand(seed)`, the degree-31 additive feedback generator.
class GlibcRand {
public:
    explicit GlibcRand(std::uint32_t seed);
    int Next();

private:
    std::int32_t state_[31];
    int front_ = 3;
    int rear_ = 0;
};

class Perlin {
public:
    static constexpr int kSampleSize = 1024;

    Perlin(int octaves, float frequency, float amplitude);
    // GRF fills the tables at the first Get, from the process's `rand` stream.
    void Init(GlibcRand &random);
    float Get(float x, float y) const;

private:
    float Noise2(const float vec[2]) const;

    int octaves_;
    float frequency_;
    float amplitude_;
    int p_[kSampleSize + kSampleSize + 2];
    float g2_[kSampleSize + kSampleSize + 2][2];
};

}  // namespace cuju::render::pitch

#endif  // CUJU_RENDER_PITCH_PERLIN_H
