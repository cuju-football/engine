// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include "bmp.h"

#include <fstream>

#include "cuju/file_path.h"

namespace cuju::render::pitch {
namespace {
std::uint32_t Read16(const std::uint8_t *p) { return p[0] | (std::uint32_t)p[1] << 8; }
std::uint32_t Read32(const std::uint8_t *p) {
    return p[0] | (std::uint32_t)p[1] << 8 | (std::uint32_t)p[2] << 16 | (std::uint32_t)p[3] << 24;
}

bool Fail(std::string *error, const char *message) {
    if (error) *error = message;
    return false;
}

int ByteShift(std::uint32_t mask) {
    for (int shift = 0; shift <= 24; shift += 8)
        if (mask == 0xFFu << shift) return shift;
    return -1;
}

}  // namespace

bool DecodeBmp(const std::uint8_t *data, std::size_t size, Image *image, std::string *error) {
    if (size < 54 || data[0] != 'B' || data[1] != 'M') return Fail(error, "not a BMP file");
    const std::uint32_t pixelOffset = Read32(data + 10);
    const std::uint32_t headerSize = Read32(data + 14);
    if (headerSize < 40 || 14 + (std::size_t)headerSize > size) return Fail(error, "BMP header");
    const std::int32_t width = (std::int32_t)Read32(data + 18);
    const std::int32_t rawHeight = (std::int32_t)Read32(data + 22);
    const int bits = (int)Read16(data + 28);
    const std::uint32_t compression = Read32(data + 30);
    if (width <= 0 || rawHeight == 0 || rawHeight == INT32_MIN) return Fail(error, "BMP size");
    const bool topDown = rawHeight < 0;
    const int height = topDown ? -rawHeight : rawHeight;
    std::uint32_t rMask = 0, gMask = 0, bMask = 0, aMask = 0;
    bool correctAlpha = false;
    if (compression == 0) {
        if (bits == 24) {
            rMask = 0x00FF0000u, gMask = 0x0000FF00u, bMask = 0x000000FFu;
        } else if (bits == 32) {
            aMask = 0xFF000000u, rMask = 0x00FF0000u, gMask = 0x0000FF00u, bMask = 0x000000FFu;
            correctAlpha = true;
        } else {
            return Fail(error, "BMP bit depth");
        }
    } else if (compression == 3 && (bits == 32 || bits == 24)) {
        const std::size_t at = 14 + 40;
        if (at + 12 > size) return Fail(error, "BMP masks");
        rMask = Read32(data + at), gMask = Read32(data + at + 4), bMask = Read32(data + at + 8);
        if (headerSize >= 56 && at + 16 <= size) aMask = Read32(data + at + 12);
    } else {
        return Fail(error, "BMP compression");
    }
    const int rShift = ByteShift(rMask), gShift = ByteShift(gMask), bShift = ByteShift(bMask);
    const int aShift = aMask ? ByteShift(aMask) : 0;
    if (rShift < 0 || gShift < 0 || bShift < 0 || aShift < 0) return Fail(error, "BMP masks");
    const int bytes = bits / 8;
    const std::size_t pitch = ((std::size_t)width * bytes + 3) & ~(std::size_t)3;
    if ((std::size_t)pixelOffset + pitch * (height - 1) + (std::size_t)width * bytes > size)
        return Fail(error, "BMP pixel data");
    image->width = width;
    image->height = height;
    image->hasAlpha = aMask != 0;
    image->rgba.assign((std::size_t)width * height * 4, 0);
    bool anyAlpha = false;
    for (int row = 0; row < height; row++) {
        const int fileRow = topDown ? row : height - 1 - row;
        const std::uint8_t *src = data + pixelOffset + pitch * fileRow;
        std::uint8_t *dst = image->rgba.data() + (std::size_t)row * width * 4;
        for (int x = 0; x < width; x++, src += bytes, dst += 4) {
            std::uint32_t pixel = src[0] | (std::uint32_t)src[1] << 8 | (std::uint32_t)src[2] << 16;
            if (bytes == 4) pixel |= (std::uint32_t)src[3] << 24;
            dst[0] = (std::uint8_t)(pixel >> rShift);
            dst[1] = (std::uint8_t)(pixel >> gShift);
            dst[2] = (std::uint8_t)(pixel >> bShift);
            dst[3] = aMask ? (std::uint8_t)(pixel >> aShift) : 255;
            anyAlpha |= aMask && dst[3] != 0;
        }
    }
    if (correctAlpha && !anyAlpha)
        for (std::size_t i = 3; i < image->rgba.size(); i += 4) image->rgba[i] = 255;
    return true;
}

bool LoadBmp(const std::string &path, Image *image, std::string *error) {
    std::ifstream file(cuju::FilePath(path), std::ios::binary);
    if (!file) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    file.seekg(0, std::ios::end);
    std::vector<std::uint8_t> bytes((std::size_t)file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(bytes.data()), (std::streamsize)bytes.size());
    if (!file) {
        if (error) *error = "cannot read " + path;
        return false;
    }
    std::string reason;
    if (!DecodeBmp(bytes.data(), bytes.size(), image, &reason)) {
        if (error) *error = path + ": " + reason;
        return false;
    }
    return true;
}

}  // namespace cuju::render::pitch
