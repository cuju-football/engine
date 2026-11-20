// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include "mip_cache.h"

#include <cstring>

#include "../mipmap.h"

namespace cuju::render::gl {

namespace {

std::uint64_t Hash(const TextureLevel &level, Format format) {
    std::uint64_t h = 0x9E3779B97F4A7C15ull ^ (static_cast<std::uint64_t>(format) << 48) ^
                      (static_cast<std::uint64_t>(level.width) << 24) ^
                      static_cast<std::uint64_t>(level.height);
    const std::uint8_t *p = level.texels.data();
    const std::size_t n = level.texels.size();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 0xFF51AFD7ED558CCDull;
        h ^= h >> 32;
    }
    for (; i < n; ++i) {
        h = (h ^ p[i]) * 0xC4CEB9FE1A85EC53ull;
    }
    return h;
}

bool SameLevel(const TextureLevel &a, const TextureLevel &b) {
    return a.width == b.width && a.height == b.height && a.texels.size() == b.texels.size() &&
           std::memcmp(a.texels.data(), b.texels.data(), a.texels.size()) == 0;
}

std::size_t Bytes(const std::vector<TextureLevel> &levels) {
    std::size_t n = 0;
    for (const TextureLevel &l : levels) n += l.texels.size();
    return n;
}

}  // namespace

MipCache::MipCache(std::size_t budget) : budget_(budget) {}

const MipCache::Entry *MipCache::Find(std::uint64_t hash, const Texture &texture) const {
    auto range = entries_.equal_range(hash);
    for (auto it = range.first; it != range.second; ++it) {
        const Entry &e = *it->second;
        if (e.format == texture.format && SameLevel(e.levels[0], texture.levels[0])) return &e;
    }
    return nullptr;
}

void MipCache::Generate(Texture &texture) {
    const std::uint64_t hash = Hash(texture.levels[0], texture.format);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (const Entry *e = Find(hash, texture)) {
            // Speed: level 0 and the chain share the kept bytes, equal to what filtering gives.
            texture.levels = e->levels;
            ++hits_;
            return;
        }
        ++misses_;
    }
    GenerateMipmaps(texture);
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t size = Bytes(texture.levels);
    if (bytes_ + size > budget_ || Find(hash, texture)) return;
    auto e = std::make_unique<Entry>();
    e->format = texture.format;
    e->levels = texture.levels;
    bytes_ += size;
    entries_.emplace(hash, std::move(e));
}

std::size_t MipCache::hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hits_;
}

std::size_t MipCache::misses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return misses_;
}

std::size_t MipCache::bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_;
}

}  // namespace cuju::render::gl
