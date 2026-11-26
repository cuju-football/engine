// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// A BMP reader that gives the results of SDL2's SDL_LoadBMP_RW for the files GRF reads,
// uncompressed 24-bit and 32-bit images, BI_RGB or BI_BITFIELDS with 8-bit channels, bottom-up or
// top-down.

#ifndef CUJU_RENDER_PITCH_BMP_H
#define CUJU_RENDER_PITCH_BMP_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuju::render::pitch {
struct Image {
    int width = 0;
    int height = 0;
    bool hasAlpha = false;
    std::vector<std::uint8_t> rgba;
};

bool DecodeBmp(const std::uint8_t *data, std::size_t size, Image *image, std::string *error);
bool LoadBmp(const std::string &path, Image *image, std::string *error);

}  // namespace cuju::render::pitch

#endif
