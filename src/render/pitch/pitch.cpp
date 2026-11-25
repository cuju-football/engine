// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// Each expression keeps GRF's types, float or double, and its order, as GRF's build compiles it
// for x86-64 SSE without contraction. sinf and powf are cuju's ports of glibc's FMA build, which
// GRF calls on the reference machine. Only pure functions of a row or a column are hoisted.

#include "pitch.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <thread>

#include "../fpenv.h"
#include "cuju/libm/powf.h"
#include "cuju/libm/sincosf.h"
#include "cuju/rng.h"
#include "perlin.h"

namespace cuju::render::pitch {
namespace {
constexpr float kPitchHalfW = 55;
constexpr float kPitchHalfH = 36;
constexpr float kPitchFullHalfW = 60;
constexpr float kPitchFullHalfH = 40;
constexpr float kPi = 3.1415926535897932384626433832795028841972f;

constexpr int kPerlinTexW = 1600;
constexpr int kPerlinTexH = 1000;
constexpr int kSeamlessTexW = 1200;
constexpr int kSeamlessTexH = 1200;
constexpr int kOverlayTexW = 4096;
constexpr int kOverlayTexH = 2048;

float Clamp(const float value, const float min, const float max) {
    if (min > value) return min;
    if (max < value) return max;
    return value;
}

float NormalizedClamp(const float value, const float min, const float max) {
    float banana = Clamp(value, min, max);
    banana = (banana - min) / (max - min);
    return banana;
}

float Curve(float source, float bias) {
    return (libm::sinf((source - 0.5f) * kPi) * 0.5f + 0.5f) * bias + source * (1.0f - bias);
}

std::uint8_t ToByte(float value) {
    if (!(value >= -2147483648.0f && value < 2147483648.0f)) return 0;
    return (std::uint8_t)(std::int32_t)value;
}

void Store(std::uint8_t *pixel, float r, float g, float b) {
    pixel[0] = ToByte(r);
    pixel[1] = ToByte(g);
    pixel[2] = ToByte(b);
    pixel[3] = 255;
}

struct Rgb {
    float x, y, z;
};

Rgb BilinearRgb(const std::uint8_t *tex, float x, float y, int w, int h) {
    const int intX1 = int(std::floor(x));
    const int intY1 = int(std::floor(y));
    const int intX2 = (int(std::floor(x)) + 1) % w;
    const int intY2 = (int(std::floor(y)) + 1) % h;
    const std::uint8_t *a = tex + 4 * ((std::size_t)intY1 * w + intX1);
    const std::uint8_t *b = tex + 4 * ((std::size_t)intY1 * w + intX2);
    const std::uint8_t *c = tex + 4 * ((std::size_t)intY2 * w + intX1);
    const std::uint8_t *d = tex + 4 * ((std::size_t)intY2 * w + intX2);
    const float xBias = x - intX1;
    const float yBias = y - intY1;
    const float xRest = 1.0f - xBias, yRest = 1.0f - yBias;
    Rgb result;
    result.x = ((float)a[0] * xRest + (float)b[0] * xBias) * yRest +
               ((float)c[0] * xRest + (float)d[0] * xBias) * yBias;
    result.y = ((float)a[1] * xRest + (float)b[1] * xBias) * yRest +
               ((float)c[1] * xRest + (float)d[1] * xBias) * yBias;
    result.z = ((float)a[2] * xRest + (float)b[2] * xBias) * yRest +
               ((float)c[2] * xRest + (float)d[2] * xBias) * yBias;
    return result;
}

float BilinearAlpha(const std::uint8_t *tex, float x, float y, int w, int h) {
    const int intX1 = int(std::floor(x));
    const int intY1 = int(std::floor(y));
    const int intX2 = (int(std::floor(x)) + 1) % w;
    const int intY2 = (int(std::floor(y)) + 1) % h;
    const float x1y1 = tex[4 * ((std::size_t)intY1 * w + intX1) + 3] / 256.0f;
    const float x2y1 = tex[4 * ((std::size_t)intY1 * w + intX2) + 3] / 256.0f;
    const float x1y2 = tex[4 * ((std::size_t)intY2 * w + intX1) + 3] / 256.0f;
    const float x2y2 = tex[4 * ((std::size_t)intY2 * w + intX2) + 3] / 256.0f;
    const float xBias = x - intX1;
    const float yBias = y - intY1;
    return (x1y1 * (1.0f - xBias) + x2y1 * xBias) * (1.0f - yBias) +
           (x1y2 * (1.0f - xBias) + x2y2 * xBias) * yBias;
}

float BilinearFloat(const float *tex, float x, float y, int w, int h) {
    const int intX1 = int(std::floor(x));
    const int intY1 = int(std::floor(y));
    const int intX2 = (int(std::floor(x)) + 1) % w;
    const int intY2 = (int(std::floor(y)) + 1) % h;
    const float x1y1 = tex[intY1 * w + intX1];
    const float x2y1 = tex[intY1 * w + intX2];
    const float x1y2 = tex[intY2 * w + intX1];
    const float x2y2 = tex[intY2 * w + intX2];
    const float xBias = x - intX1;
    const float yBias = y - intY1;
    return (x1y1 * (1.0f - xBias) + x2y1 * xBias) * (1.0f - yBias) +
           (x1y2 * (1.0f - xBias) + x2y2 * xBias) * yBias;
}

float GetSmoothGrassDirection(float coord, float repeat, int transitionSharpness) {
    const float iteration = std::floor(coord / repeat);
    float bias = coord - repeat * iteration;
    bias = libm::sinf(bias * 2 * kPi) * 0.5f + 0.5f;
    for (int i = 0; i < transitionSharpness; i++) bias = Curve(bias, 1.0f);
    bias *= 2.0f;
    bias -= 1.0f;
    return bias;
}

// GRF's random_non_determ, boost::uniform_real<float> over mt19937 and then min + u * (max - min).
class Stream {
public:
    static Stream FromView(RNGView view) {
        Stream stream;
        for (int i = 0; i < 624; i++) stream.words_[i] = view.Word(i);
        stream.index_ = (int)view.Index();
        return stream;
    }

    int Index() const { return index_; }
    void SetIndex(int index) { index_ = index; }
    std::uint32_t Word(int i) const { return words_[i]; }

    void Twist() {
        const int stateSize = 624, shiftSize = 397;
        const std::uint32_t upperMask = ~0u << 31, lowerMask = ~upperMask, twistMask = 0x9908b0dfu;
        std::uint32_t *w = words_;
        for (int i = 0; i < stateSize - shiftSize; i++) {
            const std::uint32_t mixed = (w[i] & upperMask) | (w[i + 1] & lowerMask);
            w[i] = w[i + shiftSize] ^ (mixed >> 1) ^ ((w[i + 1] & 1) * twistMask);
        }
        for (int i = stateSize - shiftSize; i < stateSize - 1; i++) {
            const std::uint32_t mixed = (w[i] & upperMask) | (w[i + 1] & lowerMask);
            w[i] = w[i - (stateSize - shiftSize)] ^ (mixed >> 1) ^ ((w[i + 1] & 1) * twistMask);
        }
        const std::uint32_t mixed = (w[stateSize - 1] & upperMask) | (w[0] & lowerMask);
        w[stateSize - 1] = w[shiftSize - 1] ^ (mixed >> 1) ^ ((w[0] & 1) * twistMask);
        index_ = 0;
    }

    std::uint32_t Next() {
        if (index_ == 624) Twist();
        return Temper(words_[index_++]);
    }
    float Uniform() {
        const float minimum = 0.0f, maximum = 1.0f;
        for (;;) {
            const float numerator = (float)(Next() - 0u);
            const float divisor = (float)(0xffffffffu - 0u) + 1;
            const float result = numerator / divisor * (maximum - minimum) + minimum;
            if (result < maximum) return result;
        }
    }
    float Range(float min, float max) {
        const float stretch = max - min;
        return min + (Uniform() * stretch);
    }

    static std::uint32_t Temper(std::uint32_t tempered) {
        tempered ^= (tempered >> 11) & 0xffffffffu;
        tempered ^= (tempered << 7) & 0x9d2c5680u;
        tempered ^= (tempered << 15) & 0xefc60000u;
        tempered ^= (tempered >> 18);
        return tempered;
    }

private:
    std::uint32_t words_[624];
    int index_ = 624;
};

bool Rejected(std::uint32_t word) {
    const float numerator = (float)(word - 0u);
    const float divisor = (float)(0xffffffffu - 0u) + 1;
    const float result = numerator / divisor * (1.0f - 0.0f) + 0.0f;
    return !(result < 1.0f);
}

void RunParallel(int threads, int tasks, const std::function<void(int)> &task) {
    std::atomic<int> next{0};
    auto worker = [&] {
        FpScope scope(FpMode::kIeee);
        for (int i; (i = next.fetch_add(1)) < tasks;) task(i);
    };
    const int extra = std::max(0, std::min(threads, tasks) - 1);
    std::vector<std::thread> pool;
    pool.reserve(extra);
    for (int t = 0; t < extra; t++) pool.emplace_back(worker);
    worker();
    for (std::thread &thread : pool) thread.join();
}

class Plan {
public:
    static constexpr int kBlocksPerSnapshot = 8;

    Plan(const Stream &start, std::uint64_t draws)
        : start_(start), startIndex_(start.Index()), draws_(draws) {}

    void Extend() { ExtendTo(draws_ + 4096); }

    void FindRejections(int threads) {
        for (;;) {
            Scan(threads);
            if (rejected_.size() <= words_ - draws_) return;
            ExtendTo(draws_ + 2 * rejected_.size() + 4096);
        }
    }

    Stream At(std::uint64_t draw) const {
        std::uint64_t word = draw;
        for (std::uint64_t r : rejected_) {
            if (r > word) break;
            word++;
        }
        const std::uint64_t position = (std::uint64_t)startIndex_ + word;
        const std::uint64_t twists = position / 624;
        Stream stream = snapshots_[twists / kBlocksPerSnapshot];
        for (std::uint64_t t = twists / kBlocksPerSnapshot * kBlocksPerSnapshot; t < twists; t++)
            stream.Twist();
        stream.SetIndex((int)(position % 624));
        return stream;
    }

private:
    void ExtendTo(std::uint64_t words) {
        const std::uint64_t twists = ((std::uint64_t)startIndex_ + words) / 624 + 1;
        if (snapshots_.empty()) snapshots_.push_back(start_);
        Stream current = snapshots_.back();
        std::uint64_t done = (snapshots_.size() - 1) * kBlocksPerSnapshot;
        while (done < twists) {
            for (int b = 0; b < kBlocksPerSnapshot; b++) current.Twist();
            done += kBlocksPerSnapshot;
            snapshots_.push_back(current);
        }
        words_ = done * 624 - startIndex_;
    }

    void Scan(int threads) {
        const int segments = (int)(snapshots_.size() - 1);
        std::vector<std::vector<std::uint64_t>> found(segments);
        RunParallel(threads, segments, [&](int s) {
            Stream stream = snapshots_[s];
            for (int b = 0; b < kBlocksPerSnapshot; b++) {
                const std::uint64_t block = (std::uint64_t)s * kBlocksPerSnapshot + b;
                if (b > 0) stream.Twist();
                for (int i = 0; i < 624; i++) {
                    const std::uint64_t position = block * 624 + i;
                    if (position < (std::uint64_t)startIndex_) continue;
                    if (Rejected(Stream::Temper(stream.Word(i))))
                        found[s].push_back(position - startIndex_);
                }
            }
        });
        rejected_.clear();
        for (auto &list : found) rejected_.insert(rejected_.end(), list.begin(), list.end());
    }

    Stream start_;
    int startIndex_;
    std::uint64_t draws_;
    std::uint64_t words_ = 0;
    std::vector<Stream> snapshots_;
    std::vector<std::uint64_t> rejected_;
};

struct Shared {
    const PitchOptions *options;
    const std::uint8_t *seamless;
    const std::uint8_t *overlay;
    const float *perlinTex;
    const Plan *plan;
    PitchTextures *textures;
    float r0, g0, b0;
};

enum Stage { kDiffuse, kSpecular, kNormal };

float Coordinate(int index, int resolution, float halfSize, int offset) {
    return (float)(index / (resolution * 1.0) * halfSize + halfSize * offset);
}

struct ChunkGeometry {
    int offsetW, offsetH;
};

ChunkGeometry Geometry(int chunk) {
    const int i = chunk + 1;
    ChunkGeometry g;
    g.offsetW = (i == 1 || i == 3) ? -1 : 0;
    g.offsetH = (i == 1 || i == 2) ? -1 : 0;
    return g;
}

void DiffuseColumns(const Shared &s, int chunk, int x0, int x1, std::uint64_t firstDraw) {
    const PitchSizes &z = s.options->sizes;
    const int resX = z.diffuseWidth, resY = z.diffuseHeight;
    const ChunkGeometry geometry = Geometry(chunk);
    std::uint8_t *out = s.textures->diffuse[chunk].data();
    Stream stream = s.plan->At(firstDraw);

    const float texMultiplier = 0.3f;
    const float texScale = 0.32f;
    const float perlinNoiseMultiplier = 0.4f;
    const float randomSpread = 2.5f;

    std::vector<float> seamlessYs(resY), perlYs(resY), overlayYs(resY), lightYs(resY);
    for (int y = 0; y < resY; y++) {
        const float yCoord = Coordinate(y, resY, kPitchFullHalfH, geometry.offsetH);
        float seamlessY =
            (float)(((yCoord / kPitchFullHalfH) * 0.5 + 0.5) * kSeamlessTexH * 12.0 * texScale);
        seamlessY = (float)std::fmod(seamlessY, kSeamlessTexH);
        seamlessYs[y] = seamlessY;
        perlYs[y] = (float)(((yCoord / kPitchFullHalfH) * 0.5 + 0.5) * kPerlinTexH);
        overlayYs[y] = (float)(((yCoord / kPitchFullHalfH) * 0.5 + 0.5) * kOverlayTexH);
        lightYs[y] = 0 - yCoord / kPitchHalfH;
    }

    for (int x = x0; x < x1; x++) {
        const float xCoord = Coordinate(x, resX, kPitchFullHalfW, geometry.offsetW);
        float seamlessX =
            (float)(((xCoord / kPitchFullHalfW) * 0.5 + 0.5) * kSeamlessTexW * 18.0 * texScale);
        seamlessX = (float)std::fmod(seamlessX, kSeamlessTexW);
        const float perlX0 = (float)(((xCoord / kPitchFullHalfW) * 0.5 + 0.5) * kPerlinTexW);
        const float overlayX = (float)(((xCoord / kPitchFullHalfW) * 0.5 + 0.5) * kOverlayTexW);
        const float lightX = 0 - xCoord / kPitchHalfW;
        for (int y = 0; y < resY; y++) {
            float r = s.r0, g = s.g0, b = s.b0;
            const Rgb tex =
                BilinearRgb(s.seamless, seamlessX, seamlessYs[y], kSeamlessTexW, kSeamlessTexH);
            r = r * (1.0f - texMultiplier) + tex.x * texMultiplier;
            g = g * (1.0f - texMultiplier) + tex.y * texMultiplier;
            b = b * (1.0f - texMultiplier) + tex.z * texMultiplier;

            const float randomX = stream.Range(-1, 1);
            const float perlX = Clamp(perlX0 + randomX * randomSpread, 0, kPerlinTexW - 1);
            const float randomY = stream.Range(-1, 1);
            const float perlY = Clamp(perlYs[y] + randomY * randomSpread, 0, kPerlinTexH - 1);
            const float perlinNoise =
                BilinearFloat(s.perlinTex, perlX, perlY, kPerlinTexW, kPerlinTexH) - 0.5f;

            const float randomNoise = 0.0f, randomNoiseMultiplier = 0.0f;
            r += ((perlinNoise * perlinNoiseMultiplier) + (randomNoise * randomNoiseMultiplier)) *
                 40.0f;
            g += ((perlinNoise * perlinNoiseMultiplier) + (randomNoise * randomNoiseMultiplier)) *
                 40.0f;
            b += ((perlinNoise * perlinNoiseMultiplier) + (randomNoise * randomNoiseMultiplier)) *
                 40.0f;

            const double lx = lightX, ly = lightYs[y], lz = 0.0f - 0.0f;
            float length = (float)std::sqrt(lx * lx + ly * ly + lz * lz);
            if (length < 0.000001) length = 0;
            const float darkness = 1.0f - libm::powf(Clamp(length * 0.7f, 0.0, 1.0), 1.5f) * 0.18f;
            r *= darkness;
            g *= darkness;
            b *= darkness;

            const float overlayY = overlayYs[y];
            const Rgb overlay =
                BilinearRgb(s.overlay, overlayX, overlayY, kOverlayTexW, kOverlayTexH);
            const float overlay_alpha =
                BilinearAlpha(s.overlay, overlayX, overlayY, kOverlayTexW, kOverlayTexH);
            r = Clamp((float)(r * (1.0 - overlay_alpha) + overlay.x * overlay_alpha), 0, 255);
            g = Clamp((float)(g * (1.0 - overlay_alpha) + overlay.y * overlay_alpha), 0, 255);
            b = Clamp((float)(b * (1.0 - overlay_alpha) + overlay.z * overlay_alpha), 0, 255);
            Store(out + 4 * ((std::size_t)y * resX + x), r, g, b);
        }
    }
}

void SpecularColumns(const Shared &s, int chunk, int x0, int x1, std::uint64_t firstDraw) {
    const PitchSizes &z = s.options->sizes;
    const int resX = z.specularWidth, resY = z.specularHeight;
    const ChunkGeometry geometry = Geometry(chunk);
    std::uint8_t *out = s.textures->specular[chunk].data();
    Stream stream = s.plan->At(firstDraw);
    const float base = 2.0f;
    const float noisefac = 18.0f;
    const float randomSpread = 2.5f;
    std::vector<float> perlYs(resY);
    for (int y = 0; y < resY; y++) {
        const float yCoord = Coordinate(y, resY, kPitchFullHalfH, geometry.offsetH);
        perlYs[y] = (float)(((yCoord / kPitchFullHalfH) * 0.5 + 0.5) * kPerlinTexH);
    }
    for (int x = x0; x < x1; x++) {
        const float xCoord = Coordinate(x, resX, kPitchFullHalfW, geometry.offsetW);
        const float perlX0 = (float)(((xCoord / kPitchFullHalfW) * 0.5 + 0.5) * kPerlinTexW);
        for (int y = 0; y < resY; y++) {
            const float randomX = stream.Range(-1, 1);
            const float perlX = Clamp(perlX0 + randomX * randomSpread, 0, kPerlinTexW - 1);
            const float randomY = stream.Range(-1, 1);
            const float perlY = Clamp(perlYs[y] + randomY * randomSpread, 0, kPerlinTexH - 1);
            const float noise =
                base +
                BilinearFloat(s.perlinTex, perlX, perlY, kPerlinTexW, kPerlinTexH) * noisefac;
            Store(out + 4 * ((std::size_t)y * resX + x), noise, noise, noise);
        }
    }
}

struct NormalTables {
    std::vector<float> rowTerm;
    std::vector<std::uint8_t> rowInside;
};

int Sharpness(float repeatMultiplier) {
    int transitionSharpness = 5;
    if (repeatMultiplier > 0.75f) transitionSharpness = 7;
    return transitionSharpness;
}

float RowTerm(float yCoord, float repeatMultiplier) {
    const float yRepeat = 11.0f * repeatMultiplier;
    const float yStrength = (float)0.1;
    return GetSmoothGrassDirection(yCoord / yRepeat, 1.0f, Sharpness(repeatMultiplier)) * yStrength;
}

float ColumnTerm(float xCoord, float repeatMultiplier) {
    const float xRepeat = 11.0f * repeatMultiplier;
    const float xStrength = (float)0.12;
    return GetSmoothGrassDirection(xCoord / xRepeat, 1.0f, Sharpness(repeatMultiplier)) * xStrength;
}

void NormalColumns(const Shared &s, const NormalTables &tables, int chunk, int x0, int x1,
                   std::uint64_t firstDraw) {
    const PitchSizes &z = s.options->sizes;
    const int resX = z.normalWidth, resY = z.normalHeight;
    const ChunkGeometry geometry = Geometry(chunk);
    const float repeatMultiplier = s.options->repeatMultiplier;
    std::uint8_t *out = s.textures->normal[chunk].data();
    Stream stream = s.plan->At(firstDraw);
    const float noisefac = 0.06f;
    for (int x = x0; x < x1; x++) {
        const float xCoord = Coordinate(x, resX, kPitchFullHalfW, geometry.offsetW);
        const bool columnInside = std::fabs(xCoord) < kPitchHalfW;
        const float columnTerm = columnInside ? ColumnTerm(xCoord, repeatMultiplier) : 0.0f;
        for (int y = 0; y < resY; y++) {
            float n0 = 0, n1 = 0, n2 = 1;
            if (columnInside && tables.rowInside[y]) {
                n0 += tables.rowTerm[y];
                n1 += columnTerm;
                n2 += 0;
            }
            n0 += stream.Range(-1, 1) * noisefac;
            n1 += stream.Range(-1, 1) * noisefac;
            const float f = 1.0f / std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);
            n0 *= f;
            n1 *= f;
            n2 *= f;
            n0 = n0 * 0.5f + 0.5f;
            n1 = n1 * 0.5f + 0.5f;
            n2 = n2 * 0.5f + 0.5f;
            Store(out + 4 * ((std::size_t)y * resX + x), n0 * 255, n1 * 255, n2 * 255);
        }
    }
}

void PerlinColumns(const Perlin &perlin1, const Perlin &perlin2, const float *ynoise,
                   float *perlinTex, int x0, int x1) {
    const float noiseFactor = (float)0.15;
    const float sinScale = 4.0f;
    for (int x = x0; x < x1; x++) {
        const float xnoise = (libm::sinf(x / (float)kPerlinTexW * 15 * sinScale) +
                              libm::sinf(x / (float)kPerlinTexW * 41 * sinScale) +
                              libm::sinf(x / (float)kPerlinTexW * 109 * sinScale) +
                              libm::sinf(x / (float)kPerlinTexW * 241 * sinScale)) *
                             0.25f;
        for (int y = 0; y < kPerlinTexH; y++) {
            float noise = xnoise * 0.65f + ynoise[y] * 0.35f;
            noise = Curve(noise * 0.5f + 0.5f, 0.4f) * 2.0f - 1.0f;
            float value =
                perlin1.Get((float)x, (float)y) * 0.4f + perlin2.Get((float)x, (float)y) * 0.6f;
            value = value * 1.7f + 0.5f;
            value += (NormalizedClamp(noise, -0.9f, 0.9f) * 2.0f - 1.0f) * noiseFactor;
            value = Clamp(value, 0.2f, 0.8f);
            value = Curve(value, 0.4f);
            perlinTex[y * kPerlinTexW + x] = value;
        }
    }
}

constexpr int kColumnsPerTask = 16;

struct Task {
    Stage stage;
    int chunk;
    int x0, x1;
    std::uint64_t firstDraw;
};

void Generate(const PitchMedia &media, const PitchOptions &options, PitchTextures *textures) {
    const PitchSizes &z = options.sizes;
    const int threads = std::max(1, options.threads);

    const float scale = 0.06f;
    Perlin perlin1(4, (float)(0.06 * scale), 0.5f);
    Perlin perlin2(4, (float)(0.14 * scale), 0.5f);
    {
        GlibcRand random(options.seed);
        random.Next();
        perlin1.Init(random);
        perlin2.Init(random);
    }
    const float sinScale = 4.0f;
    std::vector<float> ynoise(kPerlinTexH);
    for (int y = 0; y < kPerlinTexH; y++)
        ynoise[y] = (libm::sinf(y / (float)kPerlinTexH * 13 * sinScale) +
                     libm::sinf(y / (float)kPerlinTexH * 43 * sinScale) +
                     libm::sinf(y / (float)kPerlinTexH * 107 * sinScale) +
                     libm::sinf(y / (float)kPerlinTexH * 245 * sinScale)) *
                    0.25f;
    std::vector<float> perlinTex((std::size_t)kPerlinTexW * kPerlinTexH);
    auto perlinTask = [&](int t) {
        const int x0 = t * kColumnsPerTask;
        PerlinColumns(perlin1, perlin2, ynoise.data(), perlinTex.data(), x0,
                      std::min(kPerlinTexW, x0 + kColumnsPerTask));
    };
    const int perlinTasks = (kPerlinTexW + kColumnsPerTask - 1) / kColumnsPerTask;

    std::uint8_t bytes[RNG_BYTES] = {};
    const float minimum = 0.0f, maximum = 1.0f;
    std::memcpy(bytes + 2504, &minimum, 4);
    std::memcpy(bytes + 2508, &maximum, 4);
    RNGView view{bytes};
    view.Seed(options.seed);
    for (int i = 0; i < options.drawsBefore; i++) view.Uniform();
    const std::uint64_t diffuseDraws = 2ull * z.diffuseWidth * z.diffuseHeight;
    const std::uint64_t specularDraws = 2ull * z.specularWidth * z.specularHeight;
    const std::uint64_t normalDraws = 2ull * z.normalWidth * z.normalHeight;
    const std::uint64_t chunkDraws = diffuseDraws + specularDraws + normalDraws;
    Plan plan(Stream::FromView(view), 4 * chunkDraws);

    if (threads > 1) {
        std::thread twister([&plan] { plan.Extend(); });
        RunParallel(threads - 1, perlinTasks, perlinTask);
        twister.join();
    } else {
        plan.Extend();
        RunParallel(1, perlinTasks, perlinTask);
    }
    plan.FindRejections(threads);

    textures->sizes = z;
    for (int c = 0; c < 4; c++) {
        textures->diffuse[c].assign((std::size_t)z.diffuseWidth * z.diffuseHeight * 4, 0);
        textures->specular[c].assign((std::size_t)z.specularWidth * z.specularHeight * 4, 0);
        textures->normal[c].assign((std::size_t)z.normalWidth * z.normalHeight * 4, 0);
    }

    Shared shared;
    shared.options = &options;
    shared.seamless = media.seamless.rgba.data();
    shared.overlay = media.overlay.rgba.data();
    shared.perlinTex = perlinTex.data();
    shared.plan = &plan;
    shared.textures = textures;
    const float contrast = 0.4f;
    const float brightness = 2.0f;
    const float rToB = options.redToBlueRatio * 2.0f;
    shared.r0 = ((35 - contrast * 10) * rToB) * brightness;
    shared.g0 = 46 * brightness;
    shared.b0 = ((25 - contrast * 10) * (2.0f - rToB)) * brightness;

    NormalTables normalTables[4];
    RunParallel(threads, 4, [&](int c) {
        const int offsetH = Geometry(c).offsetH;
        NormalTables &table = normalTables[c];
        table.rowTerm.assign(z.normalHeight, 0.0f);
        table.rowInside.assign(z.normalHeight, 0);
        for (int y = 0; y < z.normalHeight; y++) {
            const float yCoord = Coordinate(y, z.normalHeight, kPitchFullHalfH, offsetH);
            table.rowInside[y] = std::fabs(yCoord) < kPitchHalfH;
            if (table.rowInside[y]) table.rowTerm[y] = RowTerm(yCoord, options.repeatMultiplier);
        }
    });

    std::vector<Task> tasks;
    for (int c = 0; c < 4; c++) {
        const std::uint64_t base = c * chunkDraws;
        const struct {
            Stage stage;
            int width, height;
            std::uint64_t first;
        } stages[3] = {
            {kDiffuse, z.diffuseWidth, z.diffuseHeight, base},
            {kSpecular, z.specularWidth, z.specularHeight, base + diffuseDraws},
            {kNormal, z.normalWidth, z.normalHeight, base + diffuseDraws + specularDraws}};
        for (const auto &st : stages)
            for (int x0 = 0; x0 < st.width; x0 += kColumnsPerTask)
                tasks.push_back({st.stage, c, x0, std::min(st.width, x0 + kColumnsPerTask),
                                 st.first + 2ull * x0 * st.height});
    }
    RunParallel(threads, (int)tasks.size(), [&](int t) {
        const Task &task = tasks[t];
        if (task.stage == kDiffuse)
            DiffuseColumns(shared, task.chunk, task.x0, task.x1, task.firstDraw);
        else if (task.stage == kSpecular)
            SpecularColumns(shared, task.chunk, task.x0, task.x1, task.firstDraw);
        else
            NormalColumns(shared, normalTables[task.chunk], task.chunk, task.x0, task.x1,
                          task.firstDraw);
    });
}

}

bool LoadPitchMedia(const std::string &mediaDirectory, PitchMedia *media, std::string *error) {
    const std::string pitch = mediaDirectory + "/textures/pitch/";
    return LoadBmp(pitch + "seamlessgrass08.bmp", &media->seamless, error) &&
           LoadBmp(pitch + "overlay.bmp", &media->overlay, error);
}

bool GeneratePitch(const PitchMedia &media, const PitchOptions &options, PitchTextures *textures,
                   std::string *error) {
    if (media.seamless.width != kSeamlessTexW || media.seamless.height != kSeamlessTexH ||
        media.overlay.width != kOverlayTexW || media.overlay.height != kOverlayTexH) {
        if (error) *error = "the pitch images do not have the sizes GRF asserts";
        return false;
    }
    const PitchSizes &z = options.sizes;
    if (z.diffuseWidth <= 0 || z.diffuseHeight <= 0 || z.specularWidth <= 0 ||
        z.specularHeight <= 0 || z.normalWidth <= 0 || z.normalHeight <= 0) {
        if (error) *error = "the pitch sizes must be positive";
        return false;
    }
    FpScope scope(FpMode::kIeee);
    Generate(media, options, textures);
    return true;
}

}
