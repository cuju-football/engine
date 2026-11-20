// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Texture storage. Rows are in upload order, so row 0 is texture coordinate t = 0. Formats are
// llvmpipe's pipe formats for GRF's internal formats.

#ifndef CUJU_RENDER_TEXTURE_H
#define CUJU_RENDER_TEXTURE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cuju::render {

enum class Format : std::uint8_t {
    R8G8B8X8_UNORM,      // GL_RGB8, alpha reads 1
    R8G8B8X8_SRGB,       // GL_SRGB8, alpha reads 1
    R8G8B8A8_SRGB,       // GL_SRGB8_ALPHA8
    R8G8B8A8_UNORM,      // GL_RGBA8
    R16G16B16A16_FLOAT,  // GL_RGBA16F, half bits
    Z16_UNORM,           // GL_DEPTH_COMPONENT16
    B10G10R10X2_UNORM,   // the default framebuffer, B in bits 0 to 9
};

inline int BytesPerTexel(Format f) {
    switch (f) {
        case Format::R16G16B16A16_FLOAT:
            return 8;
        case Format::Z16_UNORM:
            return 2;
        default:
            return 4;
    }
}

// Texel bytes that copies share until one of them writes. Reads go through the const accessors;
// the non-const data() makes the bytes this object's own first, so a writer never changes what
// another texture or a cache holds.
class TexelBuffer {
public:
    std::size_t size() const { return bytes_ ? bytes_->size() : 0; }
    bool empty() const { return size() == 0; }
    const std::uint8_t *data() const { return bytes_ ? bytes_->data() : nullptr; }
    std::uint8_t *data() {
        if (bytes_ && bytes_.use_count() > 1) {
            bytes_ = std::make_shared<std::vector<std::uint8_t>>(*bytes_);
        }
        return bytes_ ? bytes_->data() : nullptr;
    }
    const std::uint8_t &operator[](std::size_t i) const { return (*bytes_)[i]; }
    std::uint8_t &operator[](std::size_t i) { return data()[i]; }
    void assign(std::size_t n, std::uint8_t value) {
        bytes_ = std::make_shared<std::vector<std::uint8_t>>(n, value);
    }

private:
    std::shared_ptr<std::vector<std::uint8_t>> bytes_;
};

struct TextureLevel {
    int width = 0;
    int height = 0;
    TexelBuffer texels;  // width * height * BytesPerTexel, rows packed

    std::uint8_t *Row(int y, Format f) {
        return texels.data() + static_cast<std::size_t>(y) * width * BytesPerTexel(f);
    }
    const std::uint8_t *Row(int y, Format f) const {
        return texels.data() + static_cast<std::size_t>(y) * width * BytesPerTexel(f);
    }
};

struct Texture {
    Format format = Format::R8G8B8A8_UNORM;
    std::vector<TextureLevel> levels;  // level 0 first

    int Width(int level = 0) const { return levels[level].width; }
    int Height(int level = 0) const { return levels[level].height; }
};

}  // namespace cuju::render

#endif  // CUJU_RENDER_TEXTURE_H
