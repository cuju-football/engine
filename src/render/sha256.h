// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// SHA-256 of FIPS 180-4, for the renderer's self-check digests.

#ifndef CUJU_RENDER_SHA256_H
#define CUJU_RENDER_SHA256_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace cuju::render {

class Sha256 {
public:
    Sha256() { Reset(); }
    void Reset();
    void Update(const void *data, std::size_t size);
    // The digest in lowercase hexadecimal; the object starts over afterwards.
    std::string HexDigest();

private:
    void Block(const std::uint8_t *block);
    std::uint32_t state_[8];
    std::uint8_t buffer_[64];
    std::size_t buffered_ = 0;
    std::uint64_t length_ = 0;
};

inline std::string Sha256Hex(const void *data, std::size_t size) {
    Sha256 h;
    h.Update(data, size);
    return h.HexDigest();
}

}  // namespace cuju::render

#endif  // CUJU_RENDER_SHA256_H
