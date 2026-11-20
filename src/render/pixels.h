// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The two OpenCV calls GRF makes on its frames, cv2.cvtColor with COLOR_RGB2GRAY and cv2.resize
// with INTER_AREA, for 8-bit images, with OpenCV 5.0's results. The algorithms are OpenCV's
// (Apache-2.0 licence, Copyright Intel Corporation, Willow Garage Inc., Itseez Inc., OpenCV
// Foundation and others): modules/imgproc/src/color_rgb.simd.hpp and resize.cpp.

#ifndef CUJU_RENDER_PIXELS_H
#define CUJU_RENDER_PIXELS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cuju::render {

// cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) of a packed RGB image. Steps are in bytes.
void RgbToGray(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
               std::ptrdiff_t dst_step, int width, int height);

// cv2.resize(src, (dst_width, dst_height), interpolation=cv2.INTER_AREA) for 1 or 3 interleaved
// channels and any sizes of at least one pixel. The tables depend only on the sizes, so one
// object serves every frame of a size. Run is not reentrant, one object per thread.
class AreaResize {
public:
    static constexpr int kLanes = 8;

    AreaResize(int src_width, int src_height, int dst_width, int dst_height, int channels);

    void Run(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
             std::ptrdiff_t dst_step);

    // Run on RgbToGray of a packed RGB image, for one channel: GRF's pixels_gray.
    void RunGray(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                 std::ptrdiff_t dst_step);

    int src_width() const { return src_width_; }
    int src_height() const { return src_height_; }
    int dst_width() const { return dst_width_; }
    int dst_height() const { return dst_height_; }
    int channels() const { return channels_; }

private:
    enum class Mode { kCopy, kBlock, kArea, kLinear };

    void RunBlock(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                  std::ptrdiff_t dst_step);
    void RunArea(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                 std::ptrdiff_t dst_step, bool from_rgb);
    void RunLinear(const std::uint8_t *src, std::ptrdiff_t src_step, std::uint8_t *dst,
                   std::ptrdiff_t dst_step);

    int src_width_, src_height_, dst_width_, dst_height_, channels_;
    Mode mode_;
    int block_x_ = 0, block_y_ = 0;
    std::vector<std::int32_t> block_sum_;

    // Area: per destination column its terms, a source offset into the lane buffer and a weight;
    // per destination row its source rows and their weights.
    std::vector<int> column_first_, column_source_;
    std::vector<float> column_weight_;
    std::vector<int> row_first_, row_source_;
    std::vector<float> row_weight_;
    int ring_rows_ = 0;
    std::vector<std::uint8_t> lanes_, gray_;
    std::vector<float> group_, ring_, sum_;

    // Linear: OpenCV's fixed-point offsets and coefficients.
    std::vector<int> x_offset_, y_offset_;
    std::vector<std::int16_t> x_alpha_, y_beta_;
    int x_max_ = 0;
    std::vector<std::int32_t> line0_, line1_;
};

}  // namespace cuju::render

#endif  // CUJU_RENDER_PIXELS_H
