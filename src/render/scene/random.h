// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GRF's rng_non_deterministic (boost mt19937 with uniform_real<float>) and its random_non_determ,
// and the float libm calls of GRF's graphics code.

#ifndef CUJU_RENDER_SCENE_RANDOM_H
#define CUJU_RENDER_SCENE_RANDOM_H

#include <cmath>
#include <cstdint>
#include <cstring>

#include "cuju/libm/sincosf.h"
#include "cuju/rng.h"
#include "cuju/vector3.h"

namespace cuju::render::scene {
class GrfRng {
public:
    explicit GrfRng(std::uint32_t seed = 0) { Seed(seed); }
    void Seed(std::uint32_t seed) {
        std::memset(bytes_, 0, sizeof(bytes_));
        const float minimum = 0.0f, maximum = 1.0f;
        std::memcpy(bytes_ + 2504, &minimum, 4);
        std::memcpy(bytes_ + 2508, &maximum, 4);
        RNGView{bytes_}.Seed(seed);
    }
    float Range(float min, float max) { return RandomInRange(bytes_, min, max); }

private:
    std::uint8_t bytes_[RNG_BYTES];
};

inline float SinFloat(float x) { return libm::sinf(x); }
inline float CosFloat(float x) { return libm::cosf(x); }

inline float GetDistance(const Vector3 &a, const Vector3 &b) {
    float v0 = a.coordinates[0] - b.coordinates[0];
    float v1 = a.coordinates[1] - b.coordinates[1];
    float v2 = a.coordinates[2] - b.coordinates[2];
    if (v0 == 0.0f && v1 == 0.0f && v2 == 0.0f) return 0.0f;
    double s =
        static_cast<double>(v0) * v0 + static_cast<double>(v1) * v1 + static_cast<double>(v2) * v2;
    float length = static_cast<float>(std::sqrt(s));
    if (length < 0.000001) length = 0;
    return length;
}

}  // namespace cuju::render::scene

#endif
