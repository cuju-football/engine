# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Renders the glyph atlas of GRF's caption font with the libraries GRF loads."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import struct
import sys

LIBS = {
    'sdl': 'libSDL2-2.0.so.0',
    'ttf': 'libSDL2_ttf-2.0.so.0',
    'hb': 'libharfbuzz.so.0',
    'ft': 'libfreetype.so.6',
}
VERSIONS = {'ttf': (2, 22, 0), 'ft': (2, 13, 2), 'hb': (8, 3, 0)}
PTSIZE = 32
OUTLINE = 2
MAGIC = b'CUJGLYF1'
OUTPUT = os.path.join('fonts', 'caption_glyphs.bin')
GLYPH_CLASS_MARK = 3


class Surface(ctypes.Structure):
    _fields_ = [
        ('flags', ctypes.c_uint32),
        ('format', ctypes.c_void_p),
        ('w', ctypes.c_int),
        ('h', ctypes.c_int),
        ('pitch', ctypes.c_int),
        ('pixels', ctypes.c_void_p),
    ]


class Color(ctypes.Structure):
    _fields_ = [
        ('r', ctypes.c_uint8),
        ('g', ctypes.c_uint8),
        ('b', ctypes.c_uint8),
        ('a', ctypes.c_uint8),
    ]


class Version(ctypes.Structure):
    _fields_ = [('major', ctypes.c_uint8), ('minor', ctypes.c_uint8), ('patch', ctypes.c_uint8)]


class GlyphInfo(ctypes.Structure):
    _fields_ = [
        ('codepoint', ctypes.c_uint32),
        ('mask', ctypes.c_uint32),
        ('cluster', ctypes.c_uint32),
        ('var1', ctypes.c_uint32),
        ('var2', ctypes.c_uint32),
    ]


class GlyphPosition(ctypes.Structure):
    _fields_ = [
        ('x_advance', ctypes.c_int32),
        ('y_advance', ctypes.c_int32),
        ('x_offset', ctypes.c_int32),
        ('y_offset', ctypes.c_int32),
        ('var', ctypes.c_uint32),
    ]


class Feature(ctypes.Structure):
    _fields_ = [
        ('tag', ctypes.c_uint32),
        ('value', ctypes.c_uint32),
        ('start', ctypes.c_uint),
        ('end', ctypes.c_uint),
    ]


class FTGeneric(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('finalizer', ctypes.c_void_p)]


class FTVector(ctypes.Structure):
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


class FTBitmap(ctypes.Structure):
    _fields_ = [
        ('rows', ctypes.c_uint),
        ('width', ctypes.c_uint),
        ('pitch', ctypes.c_int),
        ('buffer', ctypes.c_void_p),
        ('num_grays', ctypes.c_ushort),
        ('pixel_mode', ctypes.c_ubyte),
        ('palette_mode', ctypes.c_ubyte),
        ('palette', ctypes.c_void_p),
    ]


class FTGlyphSlot(ctypes.Structure):
    _fields_ = [
        ('library', ctypes.c_void_p),
        ('face', ctypes.c_void_p),
        ('next', ctypes.c_void_p),
        ('glyph_index', ctypes.c_uint),
        ('generic', FTGeneric),
        ('metrics', ctypes.c_long * 8),
        ('linearHoriAdvance', ctypes.c_long),
        ('linearVertAdvance', ctypes.c_long),
        ('advance', FTVector),
        ('format', ctypes.c_int),
        ('bitmap', FTBitmap),
        ('bitmap_left', ctypes.c_int),
        ('bitmap_top', ctypes.c_int),
    ]


class FTFace(ctypes.Structure):
    _fields_ = [
        ('num_faces', ctypes.c_long),
        ('face_index', ctypes.c_long),
        ('face_flags', ctypes.c_long),
        ('style_flags', ctypes.c_long),
        ('num_glyphs', ctypes.c_long),
        ('family_name', ctypes.c_char_p),
        ('style_name', ctypes.c_char_p),
        ('num_fixed_sizes', ctypes.c_int),
        ('available_sizes', ctypes.c_void_p),
        ('num_charmaps', ctypes.c_int),
        ('charmaps', ctypes.c_void_p),
        ('generic', FTGeneric),
        ('bbox', ctypes.c_long * 4),
        ('units_per_EM', ctypes.c_ushort),
        ('ascender', ctypes.c_short),
        ('descender', ctypes.c_short),
        ('height', ctypes.c_short),
        ('max_advance_width', ctypes.c_short),
        ('max_advance_height', ctypes.c_short),
        ('underline_position', ctypes.c_short),
        ('underline_thickness', ctypes.c_short),
        ('glyph', ctypes.POINTER(FTGlyphSlot)),
    ]


class FTBitmapGlyph(ctypes.Structure):
    _fields_ = [
        ('library', ctypes.c_void_p),
        ('clazz', ctypes.c_void_p),
        ('format', ctypes.c_int),
        ('advance', FTVector),
        ('left', ctypes.c_int),
        ('top', ctypes.c_int),
        ('bitmap', FTBitmap),
    ]


def tag(s: str) -> int:
    return (ord(s[0]) << 24) | (ord(s[1]) << 16) | (ord(s[2]) << 8) | ord(s[3])


def utf8(cps) -> bytes:
    if isinstance(cps, int):
        cps = (cps,)
    return ''.join(chr(c) for c in cps).encode('utf-8')


def bitmap_bytes(bm: FTBitmap) -> bytes:
    if bm.rows == 0 or bm.width == 0:
        return b''
    raw = ctypes.string_at(bm.buffer, abs(bm.pitch) * bm.rows)
    p = abs(bm.pitch)
    return b''.join(raw[y * p : y * p + bm.width] for y in range(bm.rows))


class Reference:
    """GRF's two fonts through SDL_ttf, and FreeType and HarfBuzz set up as SDL_ttf sets them up."""

    def __init__(self, font_path: str):
        self.lib = {k: ctypes.CDLL(v) for k, v in LIBS.items()}
        sdl, ttf, hb, ft = (self.lib[k] for k in ('sdl', 'ttf', 'hb', 'ft'))
        sdl.SDL_Init(0)
        ttf.TTF_Init()
        ttf.TTF_Linked_Version.restype = ctypes.POINTER(Version)
        v = ttf.TTF_Linked_Version().contents
        found = {'ttf': (v.major, v.minor, v.patch)}
        a, b, c = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        ttf.TTF_GetFreeTypeVersion(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c))
        found['ft'] = (a.value, b.value, c.value)
        ttf.TTF_GetHarfBuzzVersion(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c))
        found['hb'] = (a.value, b.value, c.value)
        if found != VERSIONS:
            raise SystemExit(f'library versions {found}, the reference has {VERSIONS}')
        self.versions = found
        with open(font_path, 'rb') as f:
            self.font_bytes = f.read()
        self.font_buf = ctypes.create_string_buffer(self.font_bytes, len(self.font_bytes))
        n = len(self.font_bytes)
        sdl.SDL_RWFromConstMem.restype = ctypes.c_void_p
        sdl.SDL_RWFromConstMem.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ttf.TTF_OpenFontIndexRW.restype = ctypes.c_void_p
        ttf.TTF_OpenFontIndexRW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self.fonts = []
        for _ in range(2):
            rw = sdl.SDL_RWFromConstMem(self.font_buf, n)
            self.fonts.append(ttf.TTF_OpenFontIndexRW(rw, 0, PTSIZE, 0))
        ttf.TTF_SetFontOutline.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ttf.TTF_SetFontOutline(self.fonts[1], OUTLINE)
        ttf.TTF_RenderUTF8_Blended.restype = ctypes.POINTER(Surface)
        ttf.TTF_RenderUTF8_Blended.argtypes = [ctypes.c_void_p, ctypes.c_char_p, Color]
        ttf.TTF_GlyphMetrics32.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [
            ctypes.POINTER(ctypes.c_int)
        ] * 5
        for name in ('TTF_FontHeight', 'TTF_FontAscent'):
            getattr(ttf, name).argtypes = [ctypes.c_void_p]
        sdl.SDL_FreeSurface.argtypes = [ctypes.c_void_p]

        self.ft_lib = ctypes.c_void_p()
        ft.FT_Init_FreeType(ctypes.byref(self.ft_lib))
        face = ctypes.c_void_p()
        ft.FT_New_Memory_Face.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ft.FT_New_Memory_Face(self.ft_lib, self.font_buf, n, 0, ctypes.byref(face))
        ft.FT_Set_Char_Size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        ft.FT_Set_Char_Size(face, 0, PTSIZE * 64, 0, 0)
        self.face = face
        self.face_rec = ctypes.cast(face, ctypes.POINTER(FTFace)).contents
        ft.FT_Load_Glyph.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int32]
        ft.FT_Render_Glyph.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ft.FT_Get_Glyph.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        ft.FT_Stroker_New.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        ft.FT_Stroker_Set.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        ft.FT_Stroker_Done.argtypes = [ctypes.c_void_p]
        ft.FT_Glyph_Stroke.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_ubyte,
        ]
        ft.FT_Glyph_To_Bitmap.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ubyte,
        ]
        ft.FT_Done_Glyph.argtypes = [ctypes.c_void_p]

        hb.hb_ft_font_create_referenced.restype = ctypes.c_void_p
        hb.hb_ft_font_create_referenced.argtypes = [ctypes.c_void_p]
        self.hb_font = hb.hb_ft_font_create_referenced(face)
        hb.hb_ft_font_set_load_flags.argtypes = [ctypes.c_void_p, ctypes.c_int]
        hb.hb_ft_font_set_load_flags(self.hb_font, 0)  # FT_LOAD_DEFAULT | FT_LOAD_TARGET_NORMAL
        hb.hb_font_get_face.restype = ctypes.c_void_p
        hb.hb_font_get_face.argtypes = [ctypes.c_void_p]
        hb.hb_buffer_create.restype = ctypes.c_void_p
        for name in ('hb_buffer_clear_contents', 'hb_set_destroy'):
            getattr(hb, name).argtypes = [ctypes.c_void_p]
        hb.hb_buffer_set_direction.argtypes = [ctypes.c_void_p, ctypes.c_int]
        hb.hb_buffer_set_script.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        hb.hb_buffer_add_utf8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_int,
        ]
        hb.hb_shape.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Feature),
            ctypes.c_uint,
        ]
        hb.hb_buffer_get_glyph_infos.restype = ctypes.POINTER(GlyphInfo)
        hb.hb_buffer_get_glyph_infos.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        hb.hb_buffer_get_glyph_positions.restype = ctypes.POINTER(GlyphPosition)
        hb.hb_buffer_get_glyph_positions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        hb.hb_set_create.restype = ctypes.c_void_p
        hb.hb_face_collect_unicodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        hb.hb_set_next.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        hb.hb_set_next.restype = ctypes.c_int
        hb.hb_font_get_glyph_h_advance.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        hb.hb_font_get_glyph_h_advance.restype = ctypes.c_int32
        hb.hb_ot_layout_get_glyph_class.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.buffer = hb.hb_buffer_create()
        self.kern = Feature(tag('kern'), 1, 0, 0xFFFFFFFF)

    def height(self) -> int:
        return self.lib['ttf'].TTF_FontHeight(self.fonts[0])

    def ascent(self) -> int:
        return self.lib['ttf'].TTF_FontAscent(self.fonts[0])

    def num_glyphs(self) -> int:
        return self.face_rec.num_glyphs

    def codepoints(self) -> list[int]:
        hb = self.lib['hb']
        s = hb.hb_set_create()
        hb.hb_face_collect_unicodes(hb.hb_font_get_face(self.hb_font), s)
        out = []
        cp = ctypes.c_uint32(0xFFFFFFFF)
        while hb.hb_set_next(s, ctypes.byref(cp)):
            out.append(cp.value)
        hb.hb_set_destroy(s)
        return out

    def is_mark(self, glyph: int) -> bool:
        hb = self.lib['hb']
        return (
            hb.hb_ot_layout_get_glyph_class(hb.hb_font_get_face(self.hb_font), glyph)
            == GLYPH_CLASS_MARK
        )

    def advance(self, glyph: int) -> int:
        return self.lib['hb'].hb_font_get_glyph_h_advance(self.hb_font, glyph)

    def shape(self, text: bytes) -> list[tuple[int, int, int, int, int]]:
        """The glyph, x_advance, y_advance, x_offset and y_offset of each glyph of the text."""
        hb = self.lib['hb']
        b = self.buffer
        hb.hb_buffer_clear_contents(b)
        hb.hb_buffer_set_direction(b, 4)  # HB_DIRECTION_LTR
        hb.hb_buffer_set_script(b, tag('Zzzz'))  # HB_SCRIPT_UNKNOWN
        hb.hb_buffer_add_utf8(b, text, -1, 0, -1)
        hb.hb_shape(self.hb_font, b, ctypes.byref(self.kern), 1)
        n = ctypes.c_uint()
        info = hb.hb_buffer_get_glyph_infos(b, ctypes.byref(n))
        pos = hb.hb_buffer_get_glyph_positions(b, ctypes.byref(n))
        return [
            (
                info[i].codepoint,
                pos[i].x_advance,
                pos[i].y_advance,
                pos[i].x_offset,
                pos[i].y_offset,
            )
            for i in range(n.value)
        ]

    def glyph(self, glyph: int):
        """The left, top, width, rows and bytes of the glyph filled, then the same stroked."""
        ft = self.lib['ft']
        if ft.FT_Load_Glyph(self.face, glyph, 0):
            raise SystemExit(f'FT_Load_Glyph failed for glyph {glyph}')
        slot = self.face_rec.glyph
        g = ctypes.c_void_p()
        ft.FT_Get_Glyph(slot, ctypes.byref(g))
        ft.FT_Render_Glyph(slot, 0)
        bm = slot.contents.bitmap
        fill = (
            slot.contents.bitmap_left,
            slot.contents.bitmap_top,
            bm.width,
            bm.rows,
            bitmap_bytes(bm),
        )
        stroker = ctypes.c_void_p()
        ft.FT_Stroker_New(self.ft_lib, ctypes.byref(stroker))
        ft.FT_Stroker_Set(stroker, OUTLINE * 64, 1, 0, 0)  # round caps, round joins
        ft.FT_Glyph_Stroke(ctypes.byref(g), stroker, 1)
        ft.FT_Stroker_Done(stroker)
        ft.FT_Glyph_To_Bitmap(ctypes.byref(g), 0, None, 1)
        bg = ctypes.cast(g, ctypes.POINTER(FTBitmapGlyph)).contents
        outline = (bg.left, bg.top, bg.bitmap.width, bg.bitmap.rows, bitmap_bytes(bg.bitmap))
        ft.FT_Done_Glyph(g)
        return fill, outline

    def render(self, k: int, text: bytes, color=(255, 255, 255)):
        """The width, height and row-major ARGB8888 bytes of TTF_RenderUTF8_Blended."""
        ttf, sdl = self.lib['ttf'], self.lib['sdl']
        s = ttf.TTF_RenderUTF8_Blended(self.fonts[k], text, Color(*color, 0))
        if not s:
            return None
        w, h, pitch = s.contents.w, s.contents.h, s.contents.pitch
        raw = ctypes.string_at(s.contents.pixels, pitch * h)
        sdl.SDL_FreeSurface(s)
        return w, h, b''.join(raw[y * pitch : y * pitch + 4 * w] for y in range(h))

    def metrics(self, cp: int) -> tuple[int, int, int, int, int]:
        vals = [ctypes.c_int() for _ in range(5)]
        self.lib['ttf'].TTF_GlyphMetrics32(self.fonts[0], cp, *[ctypes.byref(v) for v in vals])
        return tuple(v.value for v in vals)


class Atlas:
    """The tables of the file, and the composition that src/render/overlay2d/atlas.cpp ports."""

    def __init__(self, height, ascent, glyphs, advances, cmap, merges, substs, kerns):
        self.height, self.ascent = height, ascent
        self.glyphs, self.advances = glyphs, advances
        self.cmap, self.merges, self.substs, self.kerns = cmap, merges, substs, kerns

    def layout(self, cps):
        """The glyphs of a string with their 26.6 pen positions, and the pen after the last."""
        out = []
        i = 0
        while i < len(cps):
            c = cps[i]
            nxt = cps[i + 1] if i + 1 < len(cps) else None
            if nxt is not None and (c, nxt) in self.merges:
                g = self.merges[(c, nxt)]
                i += 2
            else:
                g = self.substs.get((c, nxt), self.cmap.get(c, 0))
                i += 1
            out.append(g)
        pens = []
        pen = 0
        for j, g in enumerate(out):
            pens.append(pen)
            pen += self.advances[g]
            if j + 1 < len(out):
                pen += self.kerns.get((g, out[j + 1]), 0)
        return list(zip(out, pens, strict=True)), pen

    def compose(self, k: int, cps):
        """(width, height, alpha bytes) of TTF_RenderUTF8_Blended for font k."""
        placed, end = self.layout(cps)
        minx = maxx = miny = 0
        maxy = self.height
        for g, pen in placed:
            left, top, width, rows, _ = self.glyphs[g][0]
            x = (pen >> 6) + left
            y = self.ascent - top
            minx, maxx = min(minx, x), max(maxx, x + width)
            miny, maxy = min(miny, y), max(maxy, y + rows)
        maxx = max(maxx, end >> 6)
        w = maxx - minx + 2 * OUTLINE * k
        h = maxy - miny + 2 * OUTLINE * k
        img = bytearray(w * h)
        for g, pen in placed:
            left, top, width, rows, data = self.glyphs[g][k]
            x0 = (pen >> 6) + left + OUTLINE * k - minx
            y0 = self.ascent - top + OUTLINE * k - miny
            for y in range(max(0, -y0), min(rows, h - y0)):
                row = (y0 + y) * w + x0
                for x in range(max(0, -x0), min(width, w - x0)):
                    img[row + x] |= data[y * width + x]
        return w, h, bytes(img)


def check_render(ref: Reference, atlas: Atlas, cps, text: bytes):
    for k in range(2):
        w, h, argb = ref.render(k, text)
        mine = atlas.compose(k, cps)
        if mine != (w, h, argb[3::4]):
            return f'font {k}: composed {mine[:2]}, SDL_ttf {w, h}'
    return None


def build(ref: Reference) -> tuple[bytes, dict]:
    stats = {}
    n = ref.num_glyphs()
    glyphs = [ref.glyph(g) for g in range(n)]
    advances = [ref.advance(g) for g in range(n)]
    nominal = {}
    for cp in ref.codepoints():
        if cp < 0x20 or 0x7F <= cp < 0xA0:
            continue
        s = ref.shape(utf8(cp))
        if len(s) != 1 or s[0][2:] != (0, 0, 0) or s[0][1] != advances[s[0][0]]:
            stats.setdefault('hidden', []).append(cp)
        elif not ref.is_mark(s[0][0]):
            nominal[cp] = s[0][0]
    cps = sorted(nominal)
    reshaped = set()
    shaped = {}
    for a in cps:
        ta = utf8(a)
        ga = nominal[a]
        for b in cps:
            s = ref.shape(ta + utf8(b))
            if (
                len(s) == 2
                and s[0][0] == ga
                and s[1][0] == nominal[b]
                and s[0][1] == advances[ga]
                and s[1][1] == advances[s[1][0]]
                and not any(p[2] or p[3] or p[4] for p in s)
            ):
                continue
            shaped[(a, b)] = s
            if len(s) == 2 and s[1][0] != nominal[b]:
                reshaped.add(b)
    for c in reshaped:
        del nominal[c]
    cps = sorted(nominal)
    stats['reshaped'] = sorted(reshaped)
    merges, substs, kerns = {}, {}, {}
    for (a, b), s in sorted(shaped.items()):
        if a in nominal and b in nominal:
            if any(p[2] or p[3] or p[4] for p in s):
                raise SystemExit(f'pair U+{a:04X} U+{b:04X} has offsets {s}')
            if len(s) == 1:
                merges[(a, b)] = s[0][0]
                if s[0][1] != advances[s[0][0]]:
                    raise SystemExit(f'ligature U+{a:04X} U+{b:04X} moves {s}')
                continue
            if len(s) != 2 or s[1][0] != nominal[b] or s[1][1] != advances[s[1][0]]:
                raise SystemExit(f'pair U+{a:04X} U+{b:04X} shapes to {s}')
            if s[0][0] != nominal[a]:
                substs[(a, b)] = s[0][0]
            d = s[0][1] - advances[s[0][0]]
            if d:
                kerns[(s[0][0], nominal[b])] = d
    del shaped
    special = {}
    for (a, b), g in merges.items():
        special.setdefault(g, (a, b))
    for (a, b), g in substs.items():
        special.setdefault(g, (a, b))
    for g, rep in sorted(special.items()):
        tail = rep if g in merges.values() else rep[:1]
        for c in cps:
            if (c, rep[0]) not in merges:
                s = ref.shape(utf8((c, *rep)))
                if len(s) >= 2 and s[1][0] == g:
                    d = s[0][1] - advances[s[0][0]]
                    if d:
                        kerns[(s[0][0], g)] = d
            if tail == rep:
                s = ref.shape(utf8((*rep, c)))
                if len(s) == 2 and s[0][0] == g:
                    d = s[0][1] - advances[g]
                    if d:
                        kerns[(g, s[1][0])] = d
    atlas = Atlas(ref.height(), ref.ascent(), glyphs, advances, nominal, merges, substs, kerns)
    for cp in cps:
        minx, maxx, miny, maxy, _ = ref.metrics(cp)
        left, top, width, rows, _ = glyphs[nominal[cp]][0]
        if (minx, maxx, miny, maxy) != (left, left + width, top - rows, top):
            raise SystemExit(f'U+{cp:04X}: SDL_ttf metrics {minx, maxx, miny, maxy}')
        err = check_render(ref, atlas, [cp], utf8(cp))
        if err:
            raise SystemExit(f'U+{cp:04X}: {err}')
    for a, b in merges:
        err = check_render(ref, atlas, [a, b], utf8((a, b)))
        if err:
            raise SystemExit(f'U+{a:04X} U+{b:04X}: {err}')
    stats.update(
        glyphs=n, characters=len(cps), merges=len(merges), substs=len(substs), kerns=len(kerns)
    )

    out = bytearray(MAGIC)
    out += hashlib.sha256(ref.font_bytes).digest()
    pixels = bytearray()
    records = bytearray()
    for g in range(n):
        records += struct.pack('<i', advances[g])
        for left, top, width, rows, data in glyphs[g]:
            records += struct.pack('<hhHHI', left, top, width, rows, len(pixels))
            pixels += data
    out += struct.pack(
        '<4i6I',
        PTSIZE,
        OUTLINE,
        atlas.height,
        atlas.ascent,
        n,
        len(cps),
        len(merges),
        len(substs),
        len(kerns),
        len(pixels),
    )
    out += records
    for cp in cps:
        out += struct.pack('<II', cp, nominal[cp])
    for (a, b), g in sorted(merges.items()):
        out += struct.pack('<III', a, b, g)
    for (a, b), g in sorted(substs.items()):
        out += struct.pack('<III', a, b, g)
    for (a, b), v in sorted(kerns.items()):
        out += struct.pack('<HHi', a, b, v)
    out += pixels
    return bytes(out), stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('font')
    parser.add_argument('output')
    args = parser.parse_args()
    ref = Reference(args.font)
    data, stats = build(ref)
    path = os.path.join(args.output, OUTPUT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    print(f'{path}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}')
    print(' '.join(f'{k} {len(v) if isinstance(v, list) else v}' for k, v in stats.items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
