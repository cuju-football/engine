// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "perlin.h"

#include <cmath>

namespace cuju::render::pitch {
GlibcRand::GlibcRand(std::uint32_t seed) {
    if (seed == 0) seed = 1;
    std::int32_t word = (std::int32_t)seed;
    state_[0] = word;
    for (int i = 1; i < 31; i++) {
        const long hi = word / 127773, lo = word % 127773;
        word = (std::int32_t)(16807 * lo - 2836 * hi);
        if (word < 0) word += 2147483647;
        state_[i] = word;
    }
    for (int i = 0; i < 310; i++) Next();
}

int GlibcRand::Next() {
    const std::uint32_t value = (std::uint32_t)state_[front_] + (std::uint32_t)state_[rear_];
    state_[front_] = (std::int32_t)value;
    front_ = front_ == 30 ? 0 : front_ + 1;
    rear_ = rear_ == 30 ? 0 : rear_ + 1;
    return (int)(value >> 1);
}

Perlin::Perlin(int octaves, float frequency, float amplitude)
    : octaves_(octaves), frequency_(frequency), amplitude_(amplitude) {}

void Perlin::Init(GlibcRand &random) {
    constexpr int B = kSampleSize;
    int i, j, k;
    for (i = 0; i < B; i++) {
        p_[i] = i;
        random.Next();
        for (j = 0; j < 2; j++) g2_[i][j] = (float)((random.Next() % (B + B)) - B) / B;
        const float s = 1.0f / (float)std::sqrt(g2_[i][0] * g2_[i][0] + g2_[i][1] * g2_[i][1]);
        g2_[i][0] = g2_[i][0] * s;
        g2_[i][1] = g2_[i][1] * s;
        for (j = 0; j < 3; j++) random.Next();
    }
    while (--i) {
        k = p_[i];
        p_[i] = p_[j = random.Next() % B];
        p_[j] = k;
    }
    for (i = 0; i < B + 2; i++) {
        p_[B + i] = p_[i];
        g2_[B + i][0] = g2_[i][0];
        g2_[B + i][1] = g2_[i][1];
    }
}

float Perlin::Noise2(const float vec[2]) const {
    constexpr int BM = kSampleSize - 1;
    constexpr int N = 0x1000;
    float t = vec[0] + N;
    const int bx0 = ((int)t) & BM;
    const int bx1 = (bx0 + 1) & BM;
    const float rx0 = t - (int)t;
    const float rx1 = rx0 - 1.0f;
    t = vec[1] + N;
    const int by0 = ((int)t) & BM;
    const int by1 = (by0 + 1) & BM;
    const float ry0 = t - (int)t;
    const float ry1 = ry0 - 1.0f;
    const int i = p_[bx0];
    const int j = p_[bx1];
    const int b00 = p_[i + by0];
    const int b10 = p_[j + by0];
    const int b01 = p_[i + by1];
    const int b11 = p_[j + by1];
    const float sx = rx0 * rx0 * (3.0f - 2.0f * rx0);
    const float sy = ry0 * ry0 * (3.0f - 2.0f * ry0);
    const float *q = g2_[b00];
    float u = rx0 * q[0] + ry0 * q[1];
    q = g2_[b10];
    float v = rx1 * q[0] + ry0 * q[1];
    const float a = u + sx * (v - u);
    q = g2_[b01];
    u = rx0 * q[0] + ry1 * q[1];
    q = g2_[b11];
    v = rx1 * q[0] + ry1 * q[1];
    const float b = u + sx * (v - u);
    return a + sy * (b - a);
}

float Perlin::Get(float x, float y) const {
    float vec[2] = {x, y};
    float result = 0.0f;
    float amp = amplitude_;
    vec[0] *= frequency_;
    vec[1] *= frequency_;
    for (int i = 0; i < octaves_; i++) {
        result += Noise2(vec) * amp;
        vec[0] *= 2.0f;
        vec[1] *= 2.0f;
        amp *= 0.5f;
    }
    return result;
}

}
