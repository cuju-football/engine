// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The mipmap chains of a process, shared by the contexts of its renderers. A chain is a function
// of the format and level 0 alone, so a texture whose level 0 equals a kept one, byte for byte,
// takes the kept levels instead of filtering them again.

#ifndef CUJU_RENDER_GL_MIP_CACHE_H
#define CUJU_RENDER_GL_MIP_CACHE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../texture.h"

namespace cuju::render::gl {
class MipCache {
public:
    explicit MipCache(std::size_t budget = std::size_t{1} << 30);

    void Generate(Texture &texture);

    std::size_t hits() const;
    std::size_t misses() const;
    std::size_t bytes() const;

private:
    struct Entry {
        Format format;
        std::vector<TextureLevel> levels;
    };
    const Entry *Find(std::uint64_t hash, const Texture &texture) const;

    std::size_t budget_;
    mutable std::mutex mutex_;
    std::unordered_multimap<std::uint64_t, std::unique_ptr<Entry>> entries_;
    std::size_t bytes_ = 0, hits_ = 0, misses_ = 0;
};

}  // namespace cuju::render::gl

#endif
