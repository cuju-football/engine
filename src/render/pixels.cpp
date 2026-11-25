// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports of OpenCV's RGB2Gray<uchar> and of resize with INTER_AREA for 8-bit images, with the
// results of OpenCV 5.0 on x86-64. The algorithms are OpenCV's (Apache-2.0 licence, Copyright
// Intel Corporation, Willow Garage Inc., Itseez Inc., OpenCV Foundation and others). Every float
// operation is OpenCV's, in its order, under round to nearest even. The vector paths change only
// how values move, so they give the bits of the scalar paths.

#include "pixels.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#define CUJU_PIXELS_AVX2 1
#endif
#define CUJU_PIXELS_RESTRICT __restrict

namespace cuju::render {
namespace {
constexpr int kLanes = AreaResize::kLanes;
constexpr int kCoefScale = 1 << 11;

int Floor(double x) {
    int i = static_cast<int>(x);
    return i - (static_cast<double>(i) > x);
}

int Ceil(double x) {
    int i = static_cast<int>(x);
    return i + (static_cast<double>(i) < x);
}

float RoundEven(float x) {
    const float magic = 12582912.0f;
    return (x + magic) - magic;
}

std::uint8_t SaturateU8(float x) {
    float r = RoundEven(x);
    r = r < 0.0f ? 0.0f : r;
    r = r > 255.0f ? 255.0f : r;
    return static_cast<std::uint8_t>(r);
}

std::int16_t SaturateS16(float x) {
    int r = static_cast<int>(std::nearbyint(x));
    return static_cast<std::int16_t>(std::clamp(r, -32768, 32767));
}

struct Term {
    int destination;
    int source;
    float weight;
};

std::vector<Term> AreaTerms(int source_size, int destination_size, double scale) {
    std::vector<Term> terms;
    for (int d = 0; d < destination_size; ++d) {
        double f1 = d * scale;
        double f2 = f1 + scale;
        double cell = std::min(scale, source_size - f1);
        int s1 = Ceil(f1), s2 = Floor(f2);
        s2 = std::min(s2, source_size - 1);
        s1 = std::min(s1, s2);
        if (s1 - f1 > 1e-3) {
            terms.push_back({d, s1 - 1, static_cast<float>((s1 - f1) / cell)});
        }
        for (int s = s1; s < s2; ++s) {
            terms.push_back({d, s, static_cast<float>(1.0 / cell)});
        }
        if (f2 - s2 > 1e-3) {
            terms.push_back(
                {d, s2, static_cast<float>(std::min(std::min(f2 - s2, 1.0), cell) / cell)});
        }
    }
    return terms;
}

void TransposeRows(const std::uint8_t *const *rows, int n, std::uint8_t *out) {
    int e = 0;
#if CUJU_PIXELS_AVX2
    for (; e + 16 <= n; e += 16) {
        __m128i a[8];
        for (int l = 0; l < 8; ++l) {
            a[l] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rows[l] + e));
        }
        __m128i t[8], u[8];
        for (int l = 0; l < 4; ++l) {
            t[2 * l] = _mm_unpacklo_epi8(a[2 * l], a[2 * l + 1]);
            t[2 * l + 1] = _mm_unpackhi_epi8(a[2 * l], a[2 * l + 1]);
        }
        for (int h = 0; h < 2; ++h) {
            u[4 * h] = _mm_unpacklo_epi16(t[4 * h], t[4 * h + 2]);
            u[4 * h + 1] = _mm_unpackhi_epi16(t[4 * h], t[4 * h + 2]);
            u[4 * h + 2] = _mm_unpacklo_epi16(t[4 * h + 1], t[4 * h + 3]);
            u[4 * h + 3] = _mm_unpackhi_epi16(t[4 * h + 1], t[4 * h + 3]);
        }
        __m128i *o = reinterpret_cast<__m128i *>(out + static_cast<std::size_t>(e) * kLanes);
        for (int q = 0; q < 4; ++q) {
            _mm_storeu_si128(o + 2 * q, _mm_unpacklo_epi32(u[q], u[q + 4]));
            _mm_storeu_si128(o + 2 * q + 1, _mm_unpackhi_epi32(u[q], u[q + 4]));
        }
    }
#endif
    for (; e < n; ++e) {
        for (int l = 0; l < kLanes; ++l) {
            out[static_cast<std::size_t>(e) * kLanes + l] = rows[l][e];
        }
    }
}

template <int CN>
void Columns(const std::uint8_t *lanes, int dst_width, int channels, const int *first,
             const int *source, const float *weight, float *out) {
    const int cn = CN ? CN : channels;
#if CUJU_PIXELS_AVX2
    auto load = [](const std::uint8_t *p) {
        __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
        return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b));
    };
    for (int dx = 0; dx < dst_width; ++dx) {
        if constexpr (CN == 3) {
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0;
            for (int t = first[dx]; t < first[dx + 1]; ++t) {
                const std::uint8_t *p = lanes + source[t];
                const __m256 w = _mm256_set1_ps(weight[t]);
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(load(p), w));
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(load(p + kLanes), w));
                a2 = _mm256_add_ps(a2, _mm256_mul_ps(load(p + 2 * kLanes), w));
            }
            float *o = out + static_cast<std::size_t>(dx) * 3 * kLanes;
            _mm256_storeu_ps(o, a0);
            _mm256_storeu_ps(o + kLanes, a1);
            _mm256_storeu_ps(o + 2 * kLanes, a2);
        } else {
            for (int c = 0; c < cn; ++c) {
                __m256 a = _mm256_setzero_ps();
                for (int t = first[dx]; t < first[dx + 1]; ++t) {
                    const __m256 w = _mm256_set1_ps(weight[t]);
                    a = _mm256_add_ps(a, _mm256_mul_ps(load(lanes + source[t] + c * kLanes), w));
                }
                _mm256_storeu_ps(out + static_cast<std::size_t>(dx * cn + c) * kLanes, a);
            }
        }
    }
#else
    for (int dx = 0; dx < dst_width; ++dx) {
        for (int c = 0; c < cn; ++c) {
            float a[kLanes] = {};
            for (int t = first[dx]; t < first[dx + 1]; ++t) {
                const std::uint8_t *p = lanes + source[t] + c * kLanes;
                const float w = weight[t];
                for (int l = 0; l < kLanes; ++l) {
                    a[l] = a[l] + static_cast<float>(p[l]) * w;
                }
            }
            std::memcpy(out + static_cast<std::size_t>(dx * cn + c) * kLanes, a, sizeof a);
        }
    }
#endif
}

void SumRows(const std::uint8_t *rows, std::ptrdiff_t step, int count, int n,
             std::int32_t *CUJU_PIXELS_RESTRICT column) {
    for (int i = 0; i < n; ++i) {
        column[i] = rows[i];
    }
    for (int y = 1; y < count; ++y) {
        const std::uint8_t *CUJU_PIXELS_RESTRICT r = rows + y * step;
        for (int i = 0; i < n; ++i) {
            column[i] += r[i];
        }
    }
}

template <int CN>
void SumColumns(const std::int32_t *column, int dst_width, int block, int channels,
                std::int32_t *CUJU_PIXELS_RESTRICT sums) {
    if constexpr (CN == 0) {
        for (int dx = 0; dx < dst_width; ++dx) {
            const std::int32_t *p = column + static_cast<std::size_t>(dx) * block * channels;
            for (int c = 0; c < channels; ++c) {
                std::int32_t sum = 0;
                for (int x = 0; x < block; ++x) {
                    sum += p[x * channels + c];
                }
                sums[dx * channels + c] = sum;
            }
        }
    } else {
        for (int dx = 0; dx < dst_width; ++dx) {
            const std::int32_t *p = column + static_cast<std::size_t>(dx) * block * CN;
            std::int32_t acc[CN];
            for (int c = 0; c < CN; ++c) {
                acc[c] = p[c];
            }
            for (int x = 1; x < block; ++x) {
                for (int c = 0; c < CN; ++c) {
                    acc[c] += p[x * CN + c];
                }
            }
            for (int c = 0; c < CN; ++c) {
                sums[dx * CN + c] = acc[c];
            }
        }
    }
}

void UntransposeRows(const float *group, int width, int count, float *const *rows) {
    int d = 0;
#if CUJU_PIXELS_AVX2
    for (; d + 8 <= width; d += 8) {
        __m256 r[8], t[8], s[8];
        for (int i = 0; i < 8; ++i) {
            r[i] = _mm256_loadu_ps(group + static_cast<std::size_t>(d + i) * kLanes);
        }
        for (int i = 0; i < 4; ++i) {
            t[2 * i] = _mm256_unpacklo_ps(r[2 * i], r[2 * i + 1]);
            t[2 * i + 1] = _mm256_unpackhi_ps(r[2 * i], r[2 * i + 1]);
        }
        for (int h = 0; h < 2; ++h) {
            s[4 * h] = _mm256_shuffle_ps(t[4 * h], t[4 * h + 2], 0x44);
            s[4 * h + 1] = _mm256_shuffle_ps(t[4 * h], t[4 * h + 2], 0xEE);
            s[4 * h + 2] = _mm256_shuffle_ps(t[4 * h + 1], t[4 * h + 3], 0x44);
            s[4 * h + 3] = _mm256_shuffle_ps(t[4 * h + 1], t[4 * h + 3], 0xEE);
        }
        __m256 o[8];
        for (int i = 0; i < 4; ++i) {
            o[i] = _mm256_permute2f128_ps(s[i], s[i + 4], 0x20);
            o[i + 4] = _mm256_permute2f128_ps(s[i], s[i + 4], 0x31);
        }
        for (int l = 0; l < count; ++l) {
            _mm256_storeu_ps(rows[l] + d, o[l]);
        }
    }
#endif
    for (; d < width; ++d) {
        for (int l = 0; l < count; ++l) {
            rows[l][d] = group[static_cast<std::size_t>(d) * kLanes + l];
        }
    }
}

}

void RgbToGray(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
               std::ptrdiff_t dst_step, int width, int height) {
#if CUJU_PIXELS_AVX2
    const __m256i rg_mask =
        _mm256_setr_epi8(0, -1, 1, -1, 3, -1, 4, -1, 6, -1, 7, -1, 9, -1, 10, -1, 0, -1, 1, -1, 3,
                         -1, 4, -1, 6, -1, 7, -1, 9, -1, 10, -1);
    const __m256i b_mask =
        _mm256_setr_epi8(2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1, 2, -1, -1, -1,
                         5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1);
    const __m256i one = _mm256_set1_epi32(1 << 16);
    const __m256i rg_coef = _mm256_set1_epi32((19235 << 16) | 9798);
    const __m256i b_coef = _mm256_set1_epi32((16384 << 16) | 3735);
    auto eight = [&](const std::uint8_t *p) {
        __m256i v = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p))),
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 12)), 1);
        __m256i rg = _mm256_shuffle_epi8(v, rg_mask);
        __m256i b = _mm256_or_si256(_mm256_shuffle_epi8(v, b_mask), one);
        __m256i y = _mm256_add_epi32(_mm256_madd_epi16(rg, rg_coef), _mm256_madd_epi16(b, b_coef));
        return _mm256_srli_epi32(y, 15);
    };
#endif
    for (int y = 0; y < height; ++y) {
        const std::uint8_t *s = src + y * src_step;
        std::uint8_t *d = dst + y * dst_step;
        int x = 0;
#if CUJU_PIXELS_AVX2
        for (; 3 * x + 52 <= 3 * width; x += 16) {
            __m256i lo = eight(s + 3 * x), hi = eight(s + 3 * x + 24);
            __m256i w = _mm256_permute4x64_epi64(_mm256_packus_epi32(lo, hi), 0xD8);
            __m128i b = _mm_packus_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256(w, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(d + x), b);
        }
#endif
        for (; x < width; ++x) {
            int v = s[3 * x] * 9798 + s[3 * x + 1] * 19235 + s[3 * x + 2] * 3735 + (1 << 14);
            d[x] = static_cast<std::uint8_t>(v >> 15);
        }
    }
}

AreaResize::AreaResize(int src_width, int src_height, int dst_width, int dst_height, int channels)
    : src_width_(src_width),
      src_height_(src_height),
      dst_width_(dst_width),
      dst_height_(dst_height),
      channels_(channels) {
    const int cn = channels;
    const int width = dst_width * cn;
    if (src_width == dst_width && src_height == dst_height) {
        mode_ = Mode::kCopy;
        return;
    }
    double inv_scale_x = static_cast<double>(dst_width) / src_width;
    double inv_scale_y = static_cast<double>(dst_height) / src_height;
    double scale_x = 1.0 / inv_scale_x, scale_y = 1.0 / inv_scale_y;
    int iscale_x = static_cast<int>(std::nearbyint(scale_x));
    int iscale_y = static_cast<int>(std::nearbyint(scale_y));
    bool integer =
        std::abs(scale_x - iscale_x) < DBL_EPSILON && std::abs(scale_y - iscale_y) < DBL_EPSILON;
    if (scale_x >= 1 && scale_y >= 1 && integer) {
        mode_ = Mode::kBlock;
        block_x_ = iscale_x;
        block_y_ = iscale_y;
        block_sum_.assign(static_cast<std::size_t>(src_width + dst_width) * cn, 0);
        return;
    }
    if (scale_x >= 1 && scale_y >= 1) {
        mode_ = Mode::kArea;
        std::vector<Term> xs = AreaTerms(src_width, dst_width, scale_x);
        column_first_.assign(dst_width + 1, 0);
        for (const Term &t : xs) {
            ++column_first_[t.destination + 1];
            column_source_.push_back(t.source * cn * kLanes);
            column_weight_.push_back(t.weight);
        }
        for (int d = 0; d < dst_width; ++d) {
            column_first_[d + 1] += column_first_[d];
        }
        std::vector<Term> ys = AreaTerms(src_height, dst_height, scale_y);
        row_first_.assign(dst_height + 1, 0);
        for (const Term &t : ys) {
            ++row_first_[t.destination + 1];
            row_source_.push_back(t.source);
            row_weight_.push_back(t.weight);
        }
        int span = 0;
        for (int d = 0; d < dst_height; ++d) {
            row_first_[d + 1] += row_first_[d];
            span =
                std::max(span, row_source_[row_first_[d + 1] - 1] - row_source_[row_first_[d]] + 1);
        }
        ring_rows_ = span + kLanes;
        lanes_.assign(static_cast<std::size_t>(src_width) * cn * kLanes, 0);
        group_.assign(static_cast<std::size_t>(width) * kLanes, 0.0f);
        ring_.assign(static_cast<std::size_t>(ring_rows_) * width, 0.0f);
        sum_.assign(width, 0.0f);
        return;
    }

    mode_ = Mode::kLinear;
    x_offset_.assign(width, 0);
    x_alpha_.assign(static_cast<std::size_t>(width) * 2, 0);
    x_max_ = dst_width;
    for (int dx = 0; dx < dst_width; ++dx) {
        int sx = Floor(dx * scale_x);
        float fx = static_cast<float>((dx + 1) - (sx + 1) * inv_scale_x);
        fx = fx <= 0 ? 0.0f : fx - static_cast<float>(Floor(fx));
        if (sx < 0) {
            fx = 0, sx = 0;
        }
        if (sx + 1 >= src_width) {
            x_max_ = std::min(x_max_, dx);
            if (sx >= src_width - 1) {
                fx = 0, sx = src_width - 1;
            }
        }
        std::int16_t a0 = SaturateS16((1.0f - fx) * kCoefScale);
        std::int16_t a1 = SaturateS16(fx * kCoefScale);
        for (int c = 0; c < cn; ++c) {
            x_offset_[dx * cn + c] = sx * cn + c;
            x_alpha_[(dx * cn + c) * 2] = a0;
            x_alpha_[(dx * cn + c) * 2 + 1] = a1;
        }
    }
    y_offset_.assign(dst_height, 0);
    y_beta_.assign(static_cast<std::size_t>(dst_height) * 2, 0);
    for (int dy = 0; dy < dst_height; ++dy) {
        int sy = Floor(dy * scale_y);
        float fy = static_cast<float>((dy + 1) - (sy + 1) * inv_scale_y);
        fy = fy <= 0 ? 0.0f : fy - static_cast<float>(Floor(fy));
        y_offset_[dy] = sy;
        y_beta_[dy * 2] = SaturateS16((1.0f - fy) * kCoefScale);
        y_beta_[dy * 2 + 1] = SaturateS16(fy * kCoefScale);
    }
    line0_.assign(width, 0);
    line1_.assign(width, 0);
}

void AreaResize::Run(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                     std::ptrdiff_t dst_step) {
    switch (mode_) {
        case Mode::kCopy:
            for (int y = 0; y < src_height_; ++y) {
                std::memcpy(dst + y * dst_step, src + y * src_step,
                            static_cast<std::size_t>(src_width_) * channels_);
            }
            return;
        case Mode::kBlock:
            RunBlock(src, src_step, dst, dst_step);
            return;
        case Mode::kArea:
            RunArea(src, src_step, dst, dst_step, false);
            return;
        case Mode::kLinear:
            RunLinear(src, src_step, dst, dst_step);
            return;
    }
}

void AreaResize::RunGray(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                         std::ptrdiff_t dst_step) {
    if (mode_ == Mode::kArea) {
        gray_.resize(static_cast<std::size_t>(kLanes) * src_width_);
        RunArea(src, src_step, dst, dst_step, true);
        return;
    }
    gray_.resize(static_cast<std::size_t>(src_height_) * src_width_);
    RgbToGray(src, src_step, gray_.data(), src_width_, src_width_, src_height_);
    Run(gray_.data(), src_width_, dst, dst_step);
}

void AreaResize::RunBlock(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                          std::ptrdiff_t dst_step) {
    const int cn = channels_, bx = block_x_, by = block_y_;
    const int n = dst_width_ * bx * cn;
    const float scale = 1.0f / static_cast<float>(bx * by);
    const bool shift = bx == 2 && by == 2;
    const int width = dst_width_ * cn;
    std::int32_t *column = block_sum_.data();
    std::int32_t *sums = column + static_cast<std::size_t>(src_width_) * cn;
    for (int dy = 0; dy < dst_height_; ++dy) {
        const std::uint8_t *s = src + static_cast<std::ptrdiff_t>(dy) * by * src_step;
        SumRows(s, src_step, by, n, column);
        if (cn == 1) {
            SumColumns<1>(column, dst_width_, bx, cn, sums);
        } else if (cn == 3) {
            SumColumns<3>(column, dst_width_, bx, cn, sums);
        } else {
            SumColumns<0>(column, dst_width_, bx, cn, sums);
        }
        std::uint8_t *d = dst + dy * dst_step;
        if (shift) {
            for (int x = 0; x < width; ++x) {
                d[x] = static_cast<std::uint8_t>((sums[x] + 2) >> 2);
            }
        } else {
            for (int x = 0; x < width; ++x) {
                d[x] = SaturateU8(static_cast<float>(sums[x]) * scale);
            }
        }
    }
}

void AreaResize::RunArea(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                         std::ptrdiff_t dst_step, bool from_rgb) {
    const int cn = channels_;
    const int width = dst_width_ * cn;
    const int n = src_width_ * cn;
    float *sum = sum_.data();
    int dy = 0;
    for (int r0 = 0; r0 < src_height_; r0 += kLanes) {
        const int count = std::min(kLanes, src_height_ - r0);
        const std::uint8_t *rows[kLanes];
        float *out[kLanes];
        if (from_rgb) {
            RgbToGray(src + r0 * src_step, src_step, gray_.data(), src_width_, src_width_, count);
        }
        for (int l = 0; l < kLanes; ++l) {
            const int r = std::min(l, count - 1);
            rows[l] = from_rgb ? gray_.data() + static_cast<std::size_t>(r) * src_width_
                               : src + (r0 + r) * src_step;
            out[l] = ring_.data() + static_cast<std::size_t>((r0 + l) % ring_rows_) * width;
        }
        TransposeRows(rows, n, lanes_.data());
        if (cn == 1) {
            Columns<1>(lanes_.data(), dst_width_, cn, column_first_.data(), column_source_.data(),
                       column_weight_.data(), group_.data());
        } else if (cn == 3) {
            Columns<3>(lanes_.data(), dst_width_, cn, column_first_.data(), column_source_.data(),
                       column_weight_.data(), group_.data());
        } else {
            Columns<0>(lanes_.data(), dst_width_, cn, column_first_.data(), column_source_.data(),
                       column_weight_.data(), group_.data());
        }
        UntransposeRows(group_.data(), width, count, out);
        for (; dy < dst_height_ && row_source_[row_first_[dy + 1] - 1] < r0 + count; ++dy) {
            for (int j = row_first_[dy]; j < row_first_[dy + 1]; ++j) {
                const float *h =
                    ring_.data() + static_cast<std::size_t>(row_source_[j] % ring_rows_) * width;
                const float beta = row_weight_[j];
                if (j == row_first_[dy]) {
                    for (int d = 0; d < width; ++d) {
                        sum[d] = beta * h[d];
                    }
                } else {
                    for (int d = 0; d < width; ++d) {
                        sum[d] = sum[d] + beta * h[d];
                    }
                }
            }
            std::uint8_t *o = dst + dy * dst_step;
            for (int x = 0; x < width; ++x) {
                o[x] = SaturateU8(sum[x]);
            }
        }
    }
}

void AreaResize::RunLinear(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                           std::ptrdiff_t dst_step) {
    const int cn = channels_;
    const int width = dst_width_ * cn;
    const int x_max = x_max_ * cn;
    auto horizontal = [&](int sy, std::int32_t *out) {
        const std::uint8_t *s = src + sy * src_step;
        for (int d = 0; d < width; ++d) {
            int sx = x_offset_[d];
            if (d < x_max) {
                out[d] = s[sx] * x_alpha_[2 * d] + s[sx + cn] * x_alpha_[2 * d + 1];
            } else {
                out[d] = s[sx] * kCoefScale;
            }
        }
    };
    for (int dy = 0; dy < dst_height_; ++dy) {
        int sy0 = std::clamp(y_offset_[dy], 0, src_height_ - 1);
        int sy1 = std::clamp(y_offset_[dy] + 1, 0, src_height_ - 1);
        horizontal(sy0, line0_.data());
        horizontal(sy1, line1_.data());
        const int b0 = y_beta_[2 * dy], b1 = y_beta_[2 * dy + 1];
        std::uint8_t *d = dst + dy * dst_step;
        for (int x = 0; x < width; ++x) {
            int t0 = ((line0_[x] >> 4) * b0) >> 16;
            int t1 = ((line1_[x] >> 4) * b1) >> 16;
            d[x] = static_cast<std::uint8_t>(std::clamp((t0 + t1 + 2) >> 2, 0, 255));
        }
    }
}

}
