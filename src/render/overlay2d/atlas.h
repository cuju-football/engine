// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// GRF's caption text. GRF opens Alegreya Sans SC ExtraBold at size 32 through SDL_ttf 2.22.0, once
// plain and once with outline 2, and renders each caption with TTF_RenderUTF8_Blended. The atlas
// holds every glyph of the font as SDL_ttf loads it through FreeType 2.13.2, filled and stroked,
// with HarfBuzz 8.3.0's advances, kerning, ligatures and contextual forms for every pair of
// characters. ci/make/make_glyph_atlas.py writes it. The composition here gives SDL_ttf's surface
// byte for byte for every string over the atlas's characters, which are the font's cmap without
// control characters, default ignorables, combining marks and the Greek spacing accents.
//
// File layout, little-endian:
//   char[8] "CUJGLYF1", u8[32] SHA-256 of the font,
//   i32 ptsize, outline, height, ascent, u32 glyphs, characters, ligatures, forms, kerns, pixels,
//   per glyph: i32 advance in 26.6, then for the filled and the stroked glyph
//     i16 left, i16 top, u16 width, u16 rows, u32 offset of its rows in the pixels,
//   per character: u32 codepoint, u32 glyph, ascending,
//   per ligature: u32 first, u32 second, u32 glyph, ascending,
//   per contextual form: u32 character, u32 next character, u32 glyph, ascending,
//   per kern: u16 left glyph, u16 right glyph, i32 adjustment in 26.6, ascending,
//   u8 pixels[], coverage rows without padding.

#ifndef CUJU_RENDER_OVERLAY2D_ATLAS_H
#define CUJU_RENDER_OVERLAY2D_ATLAS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "surface.h"

namespace cuju::render::overlay2d {
class GlyphAtlas {
public:
    static constexpr int kFill = 0;
    static constexpr int kOutline = 1;

    bool Load(const std::uint8_t *data, std::size_t size, std::string *error);
    bool LoadFile(const std::string &path, std::string *error);

    bool Covers(std::string_view utf8) const;

    void Size(int font, std::string_view utf8, int &w, int &h) const;

    Surface RenderBlended(int font, std::string_view utf8, std::uint8_t r, std::uint8_t g,
                          std::uint8_t b) const;

    int ptsize() const { return ptsize_; }
    int outline() const { return outline_; }
    const std::uint8_t *font_sha256() const { return sha_; }

private:
    struct Image {
        std::int16_t left, top;
        std::uint16_t width, rows;
        std::uint32_t offset;
    };
    struct Glyph {
        std::int32_t advance;
        Image image[2];
    };
    struct Placed {
        std::uint32_t glyph;
        std::int32_t pen;
    };
    struct Layout {
        int minx, miny, w, h;
    };

    std::vector<std::uint32_t> Decode(std::string_view utf8, bool *valid) const;
    std::int32_t Nominal(std::uint32_t cp) const;
    std::int32_t Pair(const std::vector<std::uint64_t> &keys,
                      const std::vector<std::uint32_t> &values, std::uint32_t a,
                      std::uint32_t b) const;
    std::int32_t Kern(std::uint32_t a, std::uint32_t b) const;
    Layout Measure(int font, std::string_view utf8, std::vector<Placed> &placed) const;

    int ptsize_ = 0, outline_ = 0, height_ = 0, ascent_ = 0;
    std::uint8_t sha_[32] = {};
    std::vector<Glyph> glyphs_;
    std::vector<std::uint32_t> cmap_cp_, cmap_glyph_;
    std::vector<std::uint64_t> merge_key_, subst_key_;
    std::vector<std::uint32_t> merge_glyph_, subst_glyph_;
    std::vector<std::uint32_t> kern_key_;
    std::vector<std::int32_t> kern_value_;
    std::vector<std::uint8_t> pixels_;
};

}  // namespace cuju::render::overlay2d

#endif
