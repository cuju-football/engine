// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Reproduces the layout and blending of SDL_ttf 2.22.0's TTF_RenderUTF8_Blended (zlib licence,
// Copyright 2001-2024 Sam Lantinga) as measured against it: pen positions in 26.6 from HarfBuzz's
// advances and kerning, each glyph at the floor of its pen plus its bitmap offset, the surface
// spanning the pen origin, the last pen and every bitmap of the plain font, 2 * outline larger for
// the outlined font, and coverage combined by bitwise or.

#include "atlas.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>

#include "cuju/file_path.h"

namespace cuju::render::overlay2d {
namespace {
const char kMagic[8] = {'C', 'U', 'J', 'G', 'L', 'Y', 'F', '1'};

class Reader {
public:
    Reader(const std::uint8_t *p, std::size_t n) : p_(p), n_(n) {}
    bool Has(std::size_t k) const { return n_ - at_ >= k; }
    template <typename T>
    T Get() {
        T v;
        std::memcpy(&v, p_ + at_, sizeof v);
        at_ += sizeof v;
        return v;
    }
    const std::uint8_t *Here() const { return p_ + at_; }
    void Skip(std::size_t k) { at_ += k; }
    std::size_t Left() const { return n_ - at_; }

private:
    const std::uint8_t *p_;
    std::size_t n_, at_ = 0;
};

inline int Floor26(std::int32_t v) { return v >= 0 ? v >> 6 : -((-v + 63) >> 6); }

inline std::uint64_t PairKey(std::uint32_t a, std::uint32_t b) {
    return (static_cast<std::uint64_t>(a) << 32) | b;
}

}  // namespace

bool GlyphAtlas::Load(const std::uint8_t *data, std::size_t size, std::string *error) {
    auto fail = [&](const char *what) {
        if (error) *error = what;
        return false;
    };
    Reader r(data, size);
    if (!r.Has(8 + 32 + 40) || std::memcmp(r.Here(), kMagic, 8) != 0)
        return fail("not a glyph atlas");
    r.Skip(8);
    std::memcpy(sha_, r.Here(), 32);
    r.Skip(32);
    ptsize_ = r.Get<std::int32_t>();
    outline_ = r.Get<std::int32_t>();
    height_ = r.Get<std::int32_t>();
    ascent_ = r.Get<std::int32_t>();
    const std::uint32_t nglyphs = r.Get<std::uint32_t>(), nchars = r.Get<std::uint32_t>();
    const std::uint32_t nmerges = r.Get<std::uint32_t>(), nsubsts = r.Get<std::uint32_t>();
    const std::uint32_t nkerns = r.Get<std::uint32_t>(), npixels = r.Get<std::uint32_t>();
    const std::size_t need = std::size_t{28} * nglyphs + std::size_t{8} * nchars +
                             std::size_t{12} * (nmerges + nsubsts) + std::size_t{8} * nkerns +
                             npixels;
    if (r.Left() != need) return fail("glyph atlas has the wrong size");
    glyphs_.resize(nglyphs);
    for (auto &g : glyphs_) {
        g.advance = r.Get<std::int32_t>();
        for (auto &im : g.image) {
            im.left = r.Get<std::int16_t>();
            im.top = r.Get<std::int16_t>();
            im.width = r.Get<std::uint16_t>();
            im.rows = r.Get<std::uint16_t>();
            im.offset = r.Get<std::uint32_t>();
            if (std::size_t{im.offset} + std::size_t{im.width} * im.rows > npixels)
                return fail("glyph atlas has a glyph outside its pixels");
        }
    }
    cmap_cp_.resize(nchars);
    cmap_glyph_.resize(nchars);
    for (std::uint32_t i = 0; i < nchars; ++i) {
        cmap_cp_[i] = r.Get<std::uint32_t>();
        cmap_glyph_[i] = r.Get<std::uint32_t>();
        if (cmap_glyph_[i] >= nglyphs) return fail("glyph atlas maps a character to no glyph");
    }
    auto pairs = [&](std::uint32_t n, std::vector<std::uint64_t> &keys,
                     std::vector<std::uint32_t> &values) {
        keys.resize(n);
        values.resize(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t a = r.Get<std::uint32_t>(), b = r.Get<std::uint32_t>();
            keys[i] = PairKey(a, b);
            values[i] = r.Get<std::uint32_t>();
            if (values[i] >= nglyphs) return false;
        }
        return true;
    };
    if (!pairs(nmerges, merge_key_, merge_glyph_) || !pairs(nsubsts, subst_key_, subst_glyph_))
        return fail("glyph atlas names no glyph");
    kern_key_.resize(nkerns);
    kern_value_.resize(nkerns);
    for (std::uint32_t i = 0; i < nkerns; ++i) {
        const std::uint32_t a = r.Get<std::uint16_t>(), b = r.Get<std::uint16_t>();
        kern_key_[i] = (a << 16) | b;
        kern_value_[i] = r.Get<std::int32_t>();
    }
    pixels_.assign(r.Here(), r.Here() + npixels);
    if (!std::is_sorted(cmap_cp_.begin(), cmap_cp_.end()) ||
        !std::is_sorted(merge_key_.begin(), merge_key_.end()) ||
        !std::is_sorted(subst_key_.begin(), subst_key_.end()) ||
        !std::is_sorted(kern_key_.begin(), kern_key_.end()))
        return fail("glyph atlas tables are not sorted");
    return true;
}

bool GlyphAtlas::LoadFile(const std::string &path, std::string *error) {
    std::ifstream f(cuju::FilePath(path), std::ios::binary);
    if (!f) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
    return Load(bytes.data(), bytes.size(), error);
}

std::vector<std::uint32_t> GlyphAtlas::Decode(std::string_view s, bool *valid) const {
    std::vector<std::uint32_t> out;
    bool ok = true;
    std::size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<std::uint8_t>(s[i]);
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size()) {
            ok = false;
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        std::uint32_t cp = n == 1 ? c : c & (0x7F >> n);
        for (int k = 1; k < n; ++k) {
            const auto d = static_cast<std::uint8_t>(s[i + k]);
            if ((d >> 6) != 2) ok = false;
            cp = (cp << 6) | (d & 0x3F);
        }
        out.push_back(cp);
        i += n;
    }
    if (valid) *valid = ok;
    return out;
}

std::int32_t GlyphAtlas::Nominal(std::uint32_t cp) const {
    auto it = std::lower_bound(cmap_cp_.begin(), cmap_cp_.end(), cp);
    if (it == cmap_cp_.end() || *it != cp) return -1;
    return static_cast<std::int32_t>(cmap_glyph_[it - cmap_cp_.begin()]);
}

std::int32_t GlyphAtlas::Pair(const std::vector<std::uint64_t> &keys,
                              const std::vector<std::uint32_t> &values, std::uint32_t a,
                              std::uint32_t b) const {
    const std::uint64_t k = PairKey(a, b);
    auto it = std::lower_bound(keys.begin(), keys.end(), k);
    if (it == keys.end() || *it != k) return -1;
    return static_cast<std::int32_t>(values[it - keys.begin()]);
}

std::int32_t GlyphAtlas::Kern(std::uint32_t a, std::uint32_t b) const {
    const std::uint32_t k = (a << 16) | b;
    auto it = std::lower_bound(kern_key_.begin(), kern_key_.end(), k);
    if (it == kern_key_.end() || *it != k) return 0;
    return kern_value_[it - kern_key_.begin()];
}

bool GlyphAtlas::Covers(std::string_view utf8) const {
    bool valid;
    const auto cps = Decode(utf8, &valid);
    if (!valid) return false;
    for (auto c : cps)
        if (Nominal(c) < 0) return false;
    return true;
}

GlyphAtlas::Layout GlyphAtlas::Measure(int font, std::string_view utf8,
                                       std::vector<Placed> &placed) const {
    const auto cps = Decode(utf8, nullptr);
    std::vector<std::uint32_t> glyphs;
    for (std::size_t i = 0; i < cps.size();) {
        const bool more = i + 1 < cps.size();
        std::int32_t g = more ? Pair(merge_key_, merge_glyph_, cps[i], cps[i + 1]) : -1;
        if (g >= 0) {
            i += 2;
        } else {
            g = more ? Pair(subst_key_, subst_glyph_, cps[i], cps[i + 1]) : -1;
            if (g < 0) g = std::max(Nominal(cps[i]), 0);
            ++i;
        }
        glyphs.push_back(static_cast<std::uint32_t>(g));
    }
    placed.clear();
    std::int32_t pen = 0;
    int minx = 0, maxx = 0, miny = 0, maxy = height_;
    for (std::size_t j = 0; j < glyphs.size(); ++j) {
        const Glyph &g = glyphs_[glyphs[j]];
        const Image &im = g.image[kFill];
        placed.push_back({glyphs[j], pen});
        const int x = Floor26(pen) + im.left, y = ascent_ - im.top;
        minx = std::min(minx, x);
        maxx = std::max(maxx, x + im.width);
        miny = std::min(miny, y);
        maxy = std::max(maxy, y + im.rows);
        pen += g.advance;
        if (j + 1 < glyphs.size()) pen += Kern(glyphs[j], glyphs[j + 1]);
    }
    maxx = std::max(maxx, Floor26(pen));
    const int pad = font == kOutline ? 2 * outline_ : 0;
    return {minx, miny, maxx - minx + pad, maxy - miny + pad};
}

void GlyphAtlas::Size(int font, std::string_view utf8, int &w, int &h) const {
    std::vector<Placed> placed;
    const Layout l = Measure(font, utf8, placed);
    w = l.w;
    h = l.h;
}

Surface GlyphAtlas::RenderBlended(int font, std::string_view utf8, std::uint8_t r, std::uint8_t g,
                                  std::uint8_t b) const {
    std::vector<Placed> placed;
    const Layout l = Measure(font, utf8, placed);
    Surface s(l.w, l.h, Order::kBGRA);
    for (std::size_t i = 0; i < s.px.size(); i += 4) {
        s.px[i] = b;
        s.px[i + 1] = g;
        s.px[i + 2] = r;
    }
    const int shift = font == kOutline ? outline_ : 0;
    for (const Placed &p : placed) {
        const Image &im = glyphs_[p.glyph].image[font];
        const int x0 = Floor26(p.pen) + im.left + shift - l.minx;
        const int y0 = ascent_ - im.top + shift - l.miny;
        const int ya = std::max(0, -y0), yb = std::min<int>(im.rows, l.h - y0);
        const int xa = std::max(0, -x0), xb = std::min<int>(im.width, l.w - x0);
        for (int y = ya; y < yb; ++y) {
            const std::uint8_t *src = pixels_.data() + im.offset + std::size_t{im.width} * y;
            std::uint8_t *dst = s.Row(y0 + y) + 4 * x0 + 3;
            for (int x = xa; x < xb; ++x) dst[4 * x] |= src[x];
        }
    }
    return s;
}

}  // namespace cuju::render::overlay2d
