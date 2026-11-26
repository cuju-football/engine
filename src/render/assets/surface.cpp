// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// IMG_LoadBmp of utils/gui2/widgets/image.cpp (GRF, Apache-2.0) on Linux: SDL_LoadBMP_RW, then
// ARGB8888 to ABGR8888 and BGR24 to RGB24. Both conversions only reorder bytes, and the new
// surface's rows are padded with zeros to SDL's pitch.

#include "assets/assets.h"

#include <cstring>

#include "assets/text.h"
#include "pitch/bmp.h"

namespace cuju::render::assets {
bool LoadSurface(const std::string &path, Surface *surface, std::string *error) {
    const std::string name = path.substr(0, path.size() >= 4 ? path.size() - 4 : 0) + ".bmp";
    std::string bytes;
    if (!text::ReadFile(name, &bytes, error)) return false;
    const auto *data = reinterpret_cast<const std::uint8_t *>(bytes.data());
    pitch::Image image;
    std::string reason;
    if (!pitch::DecodeBmp(data, bytes.size(), &image, &reason)) {
        if (error) *error = name + ": " + reason;
        return false;
    }
    const int bits = data[28] | data[29] << 8;
    if (bits == 32 && !image.hasAlpha) {
        if (error) *error = name + ": 32-bit BMP without an alpha mask";
        return false;
    }
    const int bpp = image.hasAlpha ? 4 : 3;
    surface->w = image.width;
    surface->h = image.height;
    surface->alpha = image.hasAlpha;
    surface->pitch = (image.width * bpp + 3) & ~3;
    surface->pixels.assign((std::size_t)surface->pitch * image.height, 0);
    for (int y = 0; y < image.height; y++) {
        const std::uint8_t *src = image.rgba.data() + (std::size_t)y * image.width * 4;
        std::uint8_t *dst = surface->pixels.data() + (std::size_t)y * surface->pitch;
        if (bpp == 4) {
            std::memcpy(dst, src, (std::size_t)image.width * 4);
        } else {
            for (int x = 0; x < image.width; x++, src += 4, dst += 3) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
            }
        }
    }
    return true;
}

}  // namespace cuju::render::assets
