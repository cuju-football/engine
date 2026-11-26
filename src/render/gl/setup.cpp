// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Triangle setup and binning, ported from Mesa's llvmpipe (MIT licence): lp_setup_tri.c
// (calc_fixed_position, triangle_cw/ccw/both, do_triangle_ccw with its SSE plane code,
// lp_setup_bin_triangle), lp_setup.c (lp_setup_add_scissor_planes) and the setup JIT of
// lp_state_setup.c as its IR shows it.

#include <algorithm>
#include <cstring>

#include "../fpenv.h"
#include "../numerics.h"
#include "pipeline.h"

namespace cuju::render::gl {
namespace {
struct FtzOps {
    static float Add(float a, float b) { return ftz::Add(a, b); }
    static float Sub(float a, float b) { return ftz::Sub(a, b); }
    static float Mul(float a, float b) { return ftz::Mul(a, b); }
    static float Div(float a, float b) { return ftz::Div(a, b); }
};

struct IeeeOps {
    static float Add(float a, float b) { return a + b; }
    static float Sub(float a, float b) { return a - b; }
    static float Mul(float a, float b) { return a * b; }
    static float Div(float a, float b) { return a / b; }
};

template <class O>
CUJU_GL_NOINLINE void SetupCoefficients(const float (*v0)[4], const float (*v1)[4],
                                        const float (*v2)[4], const RasterState *rs, int num_slots,
                                        float *coefs) {
    float *a0 = coefs;
    float *dadx = coefs + 4 * num_slots;
    float *dady = coefs + 8 * num_slots;
    const float x0c = O::Sub(v0[0][0], 0.5f);
    const float y0c = O::Sub(v0[0][1], 0.5f);
    const float dx01 = O::Sub(v0[0][0], v1[0][0]);
    const float dy01 = O::Sub(v0[0][1], v1[0][1]);
    const float dx20 = O::Sub(v2[0][0], v0[0][0]);
    const float dy20 = O::Sub(v2[0][1], v0[0][1]);
    const float det = O::Sub(O::Mul(dx01, dy20), O::Mul(dy01, dx20));
    const float ooa = O::Div(1.0f, det);
    const float dy20_ooa = O::Mul(dy20, ooa);
    const float dy01_ooa = O::Mul(dy01, ooa);
    const float dx20_ooa = O::Mul(dx20, ooa);
    const float dx01_ooa = O::Mul(dx01, ooa);
    for (int s = 0; s < num_slots; ++s) {
        const bool perspective = s > 0 && rs->interp[s - 1] == Interp::kPerspective;
        for (int c = 0; c < 4; ++c) {
            float p0 = v0[s][c], p1 = v1[s][c], p2 = v2[s][c];
            if (perspective) {
                p0 = O::Mul(p0, v0[0][3]);
                p1 = O::Mul(p1, v1[0][3]);
                p2 = O::Mul(p2, v2[0][3]);
            }
            const float da01 = O::Sub(p0, p1);
            const float da20 = O::Sub(p2, p0);
            const float dx = O::Sub(O::Mul(da01, dy20_ooa), O::Mul(da20, dy01_ooa));
            const float dy = O::Sub(O::Mul(da20, dx01_ooa), O::Mul(da01, dx20_ooa));
            a0[4 * s + c] = O::Sub(p0, O::Add(O::Mul(dx, x0c), O::Mul(dy, y0c)));
            dadx[4 * s + c] = dx;
            dady[4 * s + c] = dy;
        }
    }
    a0[0] = 0.0f;
}

struct FixedPosition {
    std::int32_t x[4];
    std::int32_t y[4];
    std::int32_t dx01, dy01, dx20, dy20;
};

inline std::int32_t Wrap32(std::int64_t v) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(static_cast<std::uint64_t>(v)));
}
inline std::int32_t SubWrap(std::int32_t a, std::int32_t b) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) - static_cast<std::uint32_t>(b));
}

inline std::int32_t Snap(float v) { return CvtPs2Dq((v - 0.5f) * 256.0f); }

int CalcFixedPosition(FixedPosition *p, const float (*v0)[4], const float (*v1)[4],
                      const float (*v2)[4]) {
    const std::int32_t X0 = Snap(v0[0][0]), Y0 = Snap(v0[0][1]);
    const std::int32_t X1 = Snap(v1[0][0]), Y1 = Snap(v1[0][1]);
    const std::int32_t X2 = Snap(v2[0][0]), Y2 = Snap(v2[0][1]);
    p->dx01 = SubWrap(X0, X1);
    p->dy01 = SubWrap(Y0, Y1);
    p->dx20 = SubWrap(X2, X0);
    p->dy20 = SubWrap(Y2, Y0);
    p->x[0] = X0;
    p->x[1] = X1;
    p->x[2] = X2;
    p->x[3] = X0;
    p->y[0] = Y0;
    p->y[1] = Y1;
    p->y[2] = Y2;
    p->y[3] = Y0;
    const std::uint64_t area =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(p->dx01) * p->dy20) -
        static_cast<std::uint64_t>(static_cast<std::int64_t>(p->dx20) * p->dy01);
    return area == 0 ? 0 : (area & (1ull << 63)) ? -1 : 1;
}

void RotateFixedPosition01(FixedPosition *p) {
    const std::int32_t x = p->x[1], y = p->y[1];
    p->x[1] = p->x[0];
    p->y[1] = p->y[0];
    p->x[0] = x;
    p->y[0] = y;
    p->dx01 = SubWrap(0, p->dx01);
    p->dy01 = SubWrap(0, p->dy01);
    p->dx20 = SubWrap(p->x[2], p->x[0]);
    p->dy20 = SubWrap(p->y[2], p->y[0]);
}

struct Rect {
    int x0, y0, x1, y1;
};

bool RectIntersects(const Rect &a, const Rect &b) {
    return !(a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0 || a.x1 < a.x0 ||
             a.y1 < a.y0 || b.x1 < b.x0 || b.y1 < b.y0);
}

inline std::int32_t AsrInt(std::int32_t v, int s) { return v >> s; }

std::uint32_t FloorPot(std::uint32_t n) {
    if (n == 0) {
        return 0;
    }
    std::uint32_t r = 1;
    while (n >>= 1) {
        r <<= 1;
    }
    return r;
}

void BinTriangle(const RasterState &rs, Scene &scene, std::uint32_t tri_index, bool use_32bits,
                 const Rect &bbox, int nr_planes) {
    const Rect fb = {0, 0, rs.fb_width - 1, rs.fb_height - 1};
    const int dx = static_cast<int>(
        FloorPot(static_cast<std::uint32_t>((bbox.x0 ^ bbox.x1) | (bbox.y0 ^ bbox.y1))));
    const int max_sz = (bbox.x1 - (bbox.x0 & ~3)) | (bbox.y1 - (bbox.y0 & ~3));
    const int sz = static_cast<int>(FloorPot(static_cast<std::uint32_t>(max_sz)));
    Rect trimmed = bbox;
    if (trimmed.x0 < fb.x0) trimmed.x0 = fb.x0;
    if (trimmed.x1 > fb.x1) trimmed.x1 = fb.x1;
    if (trimmed.y0 < fb.y0) trimmed.y0 = fb.y0;
    if (trimmed.y1 > fb.y1) trimmed.y1 = fb.y1;
    ++scene.triangles_binned;

    if (dx < kTileSize) {
        const int ix0 = bbox.x0 / kTileSize;
        const int iy0 = bbox.y0 / kTileSize;
        int px = bbox.x0 & 63 & ~3;
        int py = bbox.y0 & 63 & ~3;
        if (nr_planes == 3) {
            if (sz < 4) {
                if (use_32bits) {
                    scene.Bin(ix0, iy0, Op::kTri32_3_4, static_cast<std::uint16_t>(px | (py << 8)),
                              tri_index);
                } else {
                    scene.Bin(ix0, iy0, Op::kTri64, 7, tri_index);
                }
                return;
            }
            if (sz < 16) {
                px = std::min(px, kTileSize - 16);
                py = std::min(py, kTileSize - 16);
                if (use_32bits) {
                    scene.Bin(ix0, iy0, Op::kTri32_3_16, static_cast<std::uint16_t>(px | (py << 8)),
                              tri_index);
                } else {
                    scene.Bin(ix0, iy0, Op::kTri64, 7, tri_index);
                }
                return;
            }
        } else if (nr_planes == 4 && sz < 16) {
            px = std::min(px, kTileSize - 16);
            py = std::min(py, kTileSize - 16);
            if (use_32bits) {
                scene.Bin(ix0, iy0, Op::kTri32_4_16, static_cast<std::uint16_t>(px | (py << 8)),
                          tri_index);
            } else {
                scene.Bin(ix0, iy0, Op::kTri64, 15, tri_index);
            }
            return;
        }
        scene.Bin(ix0, iy0, use_32bits ? Op::kTri32 : Op::kTri64,
                  static_cast<std::uint16_t>((1u << nr_planes) - 1), tri_index);
        return;
    }

    const SceneTriangle &tri = scene.Triangle(tri_index);
    std::int64_t c[kMaxPlanes], ei[kMaxPlanes], eo[kMaxPlanes], xstep[kMaxPlanes],
        ystep[kMaxPlanes];
    const int ix0 = trimmed.x0 / kTileSize;
    const int iy0 = trimmed.y0 / kTileSize;
    const int ix1 = trimmed.x1 / kTileSize;
    const int iy1 = trimmed.y1 / kTileSize;
    for (int i = 0; i < nr_planes; ++i) {
        const Plane &pl = tri.plane[i];
        c[i] = pl.c + static_cast<std::int64_t>(pl.dcdy) * iy0 * kTileSize -
               static_cast<std::int64_t>(pl.dcdx) * ix0 * kTileSize;
        ei[i] = (static_cast<std::int64_t>(pl.dcdy) - pl.dcdx - static_cast<std::int64_t>(pl.eo))
                << kTileOrder;
        eo[i] = static_cast<std::int64_t>(pl.eo) << kTileOrder;
        xstep[i] = -(static_cast<std::int64_t>(pl.dcdx) << kTileOrder);
        ystep[i] = static_cast<std::int64_t>(pl.dcdy) << kTileOrder;
    }
    for (int y = iy0; y <= iy1; ++y) {
        bool in = false;
        std::int64_t cx[kMaxPlanes];
        for (int i = 0; i < nr_planes; ++i) {
            cx[i] = c[i];
        }
        for (int x = ix0; x <= ix1; ++x) {
            int out = 0, partial = 0;
            for (int i = 0; i < nr_planes; ++i) {
                const std::int64_t planeout = cx[i] + eo[i];
                const std::int64_t planepartial = cx[i] + ei[i] - 1;
                out |= static_cast<int>(planeout >> 63);
                partial |= static_cast<int>(planepartial >> 63) & (1 << i);
            }
            if (out) {
                if (in) {
                    break;
                }
            } else if (partial) {
                in = true;
                scene.Bin(x, y, use_32bits ? Op::kTri32 : Op::kTri64,
                          static_cast<std::uint16_t>(partial), tri_index);
            } else {
                in = true;
                scene.Bin(x, y, Op::kShadeTile, 0, tri_index);
            }
            for (int i = 0; i < nr_planes; ++i) {
                cx[i] += xstep[i];
            }
        }
        for (int i = 0; i < nr_planes; ++i) {
            c[i] += ystep[i];
        }
    }
}

void DoTriangleCcw(const RasterState &rs, const DrawState *state, Scene &scene, bool ieee,
                   const FixedPosition &pos, const float (*v0)[4], const float (*v1)[4],
                   const float (*v2)[4], bool frontfacing) {
    const int adj = rs.bottom_edge_rule ? 1 : 0;
    Rect bbox;
    {
        const std::int32_t xmin = std::min(std::min(pos.x[0], pos.x[1]), pos.x[2]);
        const std::int32_t xmax = std::max(std::max(pos.x[0], pos.x[1]), pos.x[2]);
        const std::int32_t ymin = std::min(std::min(pos.y[0], pos.y[1]), pos.y[2]);
        const std::int32_t ymax = std::max(std::max(pos.y[0], pos.y[1]), pos.y[2]);
        bbox.x0 = AsrInt(xmin, kFixedOrder);
        bbox.x1 = AsrInt(SubWrap(xmax, 1), kFixedOrder);
        bbox.y0 = AsrInt(SubWrap(ymin, -adj), kFixedOrder);
        bbox.y1 = AsrInt(SubWrap(SubWrap(ymax, 1), -adj), kFixedOrder);
    }
    const Rect fb = {0, 0, rs.fb_width - 1, rs.fb_height - 1};
    if (!RectIntersects(fb, bbox)) {
        return;
    }
    const int max_szorig = SubWrap(bbox.x1, bbox.x0 & ~3) | SubWrap(bbox.y1, bbox.y0 & ~3);
    const bool use_32bits = max_szorig <= kMaxFixedLength32;
    bbox.x0 = std::max(bbox.x0, 0);
    bbox.y0 = std::max(bbox.y0, 0);
    const bool s_planes[4] = {bbox.x0<fb.x0, bbox.x1> fb.x1, bbox.y0<fb.y0, bbox.y1> fb.y1};
    const int nr_planes = 3 + s_planes[0] + s_planes[1] + s_planes[2] + s_planes[3];

    const int num_slots = 1 + rs.num_inputs;
    const std::uint32_t index = scene.NewTriangle(state, num_slots, frontfacing);
    SceneTriangle &tri = scene.Triangle(index);
    tri.nr_planes = static_cast<std::uint8_t>(nr_planes);
    float *coefs = scene.Coefs(tri.coef);
    if (ieee) {
        SetupCoefficients<IeeeOps>(v0, v1, v2, &rs, num_slots, coefs);
    } else {
        SetupCoefficients<FtzOps>(v0, v1, v2, &rs, num_slots, coefs);
    }

    const bool top_left = !rs.bottom_edge_rule;
    for (int i = 0; i < 3; ++i) {
        const int j = i == 2 ? 0 : i + 1;
        const std::int32_t dcdx = SubWrap(pos.y[i], pos.y[j]);
        const std::int32_t dcdy = SubWrap(pos.x[i], pos.x[j]);
        const bool dcdx_neg = dcdx < 0;
        const bool dcdy_neg = dcdy < 0;
        const bool c_dec = dcdx_neg || (dcdx == 0 && (dcdy_neg != top_left));
        const std::uint64_t c =
            static_cast<std::uint64_t>(static_cast<std::int64_t>(dcdx) * pos.x[i]) -
            static_cast<std::uint64_t>(static_cast<std::int64_t>(dcdy) * pos.y[i]) +
            (c_dec ? 1u : 0u);
        const std::int32_t sdcdx =
            static_cast<std::int32_t>(static_cast<std::uint32_t>(dcdx) << kFixedOrder);
        const std::int32_t sdcdy =
            static_cast<std::int32_t>(static_cast<std::uint32_t>(dcdy) << kFixedOrder);
        const std::uint32_t eo = (dcdy_neg ? 0u : static_cast<std::uint32_t>(sdcdy)) -
                                 (dcdx_neg ? static_cast<std::uint32_t>(sdcdx) : 0u);
        tri.plane[i].c = static_cast<std::int64_t>(c);
        tri.plane[i].dcdx = sdcdx;
        tri.plane[i].dcdy = sdcdy;
        tri.plane[i].eo = eo;
    }
    Plane *ps = &tri.plane[3];
    if (s_planes[0]) {
        const int x0 = fb.x0 - 1;
        ps->dcdx = static_cast<std::int32_t>(~0u << 8);
        ps->dcdy = 0;
        ps->c = -(static_cast<std::int64_t>(x0) << 8);
        ps->eo = 1u << 8;
        ++ps;
    }
    if (s_planes[1]) {
        ps->dcdx = 1 << 8;
        ps->dcdy = 0;
        ps->c = (static_cast<std::int64_t>(fb.x1) << 8) + 127;
        ps->eo = 0;
        ++ps;
    }
    if (s_planes[2]) {
        const int y0 = fb.y0 - 1;
        ps->dcdx = 0;
        ps->dcdy = 1 << 8;
        ps->c = -(static_cast<std::int64_t>(y0) << 8);
        ps->eo = 1u << 8;
        ++ps;
    }
    if (s_planes[3]) {
        ps->dcdx = 0;
        ps->dcdy = static_cast<std::int32_t>(~0u << 8);
        ps->c = (static_cast<std::int64_t>(fb.y1) << 8) + 127;
        ps->eo = 0;
        ++ps;
    }
    BinTriangle(rs, scene, index, use_32bits, bbox, nr_planes);
}

}  // namespace

void SetupTriangle(const RasterState &rs, const DrawState *state, Scene &scene, bool ieee,
                   const float (*v0)[4], const float (*v1)[4], const float (*v2)[4]) {
    const std::uint32_t cull = rs.cull_face;
    if (cull == 3) {
        return;
    }
    FixedPosition pos;
    const int area = CalcFixedPosition(&pos, v0, v1, v2);
    const bool ccw_front = rs.front_ccw;
    bool keep_ccw, keep_cw;
    if (cull == 0) {
        keep_ccw = keep_cw = true;
    } else if (cull == 2) {
        keep_ccw = ccw_front;
        keep_cw = !ccw_front;
    } else {
        keep_ccw = !ccw_front;
        keep_cw = ccw_front;
    }
    if (area > 0 && keep_ccw) {
        DoTriangleCcw(rs, state, scene, ieee, pos, v0, v1, v2, ccw_front);
    } else if (area < 0 && keep_cw) {
        RotateFixedPosition01(&pos);
        DoTriangleCcw(rs, state, scene, ieee, pos, v1, v0, v2, !ccw_front);
    }
}

}  // namespace cuju::render::gl
