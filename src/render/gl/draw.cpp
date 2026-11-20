// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The draw module of Mesa's gallium (MIT licence) as llvmpipe runs it for GRF: the vsplit front
// end (draw_pt_vsplit.c, draw_split_tmp.h), the fetch-shade-pipeline middle end with its LLVM
// vertex function (draw_pt_fetch_shade_pipeline_llvm.c, draw_llvm.c: fetch, clip mask, viewport),
// the emit path (draw_pt_emit.c), and the pipeline stages clip (draw_pipe_clip.c), cull
// (draw_pipe_cull.c) and vbuf (draw_pipe_vbuf.c) feeding llvmpipe's setup (lp_setup_vbuf.c).
//
// draw_vbo runs under FTZ and DAZ; pipeline triangles still buffered when it returns are set up
// by llvmpipe_draw_vbo's final draw_flush under the application's IEEE environment.

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "../fpenv.h"
#include "../numerics.h"
#include "pipeline.h"

namespace cuju::render::gl {

namespace {

constexpr std::uint32_t kUndefinedVertexId = 0xFFFF;
constexpr int kSegmentSize = 1024;
constexpr int kMaxVerticesLinear = 4096;
constexpr int kMapSize = 256;
constexpr int kMaxClippedVertices = 2 * (6 + 8) + 1;
constexpr int kVbufBytes = 4096;
constexpr int kVbufIndices = 1020;

// draw's vertex_header with its data: clip_pos and data[0] the window position, data[1..] the
// fragment inputs.
struct PVertex {
    std::uint32_t clipmask;
    std::uint32_t vertex_id;
    float clip[4];
    float data[kMaxSlots][4];
};

inline bool IsInfOrNan(float x) { return (Bits(x) & 0x7F800000u) == 0x7F800000u; }

class DrawRunner {
public:
    DrawRunner(const DrawState &state, const RasterState &rs, const DrawInput &input, Scene &scene)
        : state_(state),
          rs_(rs),
          input_(input),
          scene_(scene),
          kept_(nullptr),
          num_slots_(1 + rs.num_inputs) {
        vertex_size_ = 16 * num_slots_;
        max_vertices_ = kVbufBytes / vertex_size_;
        if (max_vertices_ >= static_cast<int>(kUndefinedVertexId)) {
            max_vertices_ = static_cast<int>(kUndefinedVertexId) - 1;
        }
        tmps_.resize(kMaxClippedVertices + 1);
    }

    void SetKept(const DrawState *kept) { kept_ = kept; }

    // Everything inside draw_vbo, under FTZ and DAZ.
    void RunSegments();
    // The final draw_flush, under IEEE.
    void FinalFlush() { FlushVertices(true); }

private:
    void RunSegment(const std::uint32_t *fetch, int fetch_count, const std::uint16_t *elts,
                    int draw_count);
    void Fetch(const std::uint32_t *fetch, int count);
    void PostVs(int count);
    void Emit(const std::uint16_t *elts, int count);
    void PipelineTriangle(PVertex *v0, PVertex *v1, PVertex *v2);
    void ClipTriangle(PVertex *v0, PVertex *v1, PVertex *v2, std::uint32_t clipmask);
    void Interp(PVertex *dst, float t, const PVertex *out, const PVertex *in);
    void CullTriangle(PVertex *v0, PVertex *v1, PVertex *v2);
    void VbufTriangle(PVertex *v0, PVertex *v1, PVertex *v2);
    std::uint16_t EmitVertex(PVertex *v);
    void FlushVertices(bool ieee);
    void SetupHw(const float *v0, const float *v1, const float *v2, bool ieee);

    const DrawState &state_;
    const RasterState &rs_;
    const DrawInput &input_;
    Scene &scene_;
    const DrawState *kept_;
    int num_slots_;

    std::vector<PVertex> verts_;  // the current segment's vertices
    std::vector<float> attribs_[kMaxAttributes];
    std::vector<float> clip_, inputs_;
    std::vector<PVertex> tmps_;
    bool in_pipeline_ = false;

    // The vbuf stage.
    int vertex_size_, max_vertices_;
    bool vbuf_allocated_ = false;
    bool vbuf_first_ = true;
    std::vector<float> vbuf_vertices_;
    int vbuf_nr_vertices_ = 0;
    std::vector<std::uint16_t> vbuf_indices_;

    // The emit path's hardware vertices.
    std::vector<float> hw_;
};

void DrawRunner::Fetch(const std::uint32_t *fetch, int count) {
    const std::vector<AttributeDecl> &attrs = state_.program->Attributes();
    for (const AttributeDecl &a : attrs) {
        const AttributeSource &src = input_.attribs[a.location];
        std::vector<float> &out = attribs_[a.location];
        out.resize(static_cast<std::size_t>(count) * 4);
        for (int i = 0; i < count; ++i) {
            float *v = &out[static_cast<std::size_t>(i) * 4];
            if (!src.enabled) {
                std::memcpy(v, src.generic, 16);
                continue;
            }
            // draw_llvm's fetch: the element is read when stride * index, in 32 bits, is below
            // buffer_size - (format size - 1) - offset; otherwise it reads as zero.
            const std::uint32_t bsize = static_cast<std::uint32_t>(src.size) * 4u;
            std::uint32_t adj = 0;
            const std::uint64_t size = src.buffer_size;
            if (size >= bsize - 1 && size - (bsize - 1) >= src.offset && size <= 0xFFFFFFFFu) {
                adj = static_cast<std::uint32_t>(size - (bsize - 1) - src.offset);
            }
            const std::uint32_t off = static_cast<std::uint32_t>(src.stride) * fetch[i];
            float f[4] = {0, 0, 0, 0};
            if (off < adj && src.data) {
                std::memcpy(f, src.data + src.offset + off, bsize);
            }
            v[0] = f[0];
            v[1] = src.size > 1 ? f[1] : 0.0f;
            v[2] = src.size > 2 ? f[2] : 0.0f;
            v[3] = src.size > 3 ? f[3] : 1.0f;
        }
    }
    const float *ptrs[kMaxAttributes] = {};
    for (const AttributeDecl &a : attrs) {
        ptrs[a.location] = attribs_[a.location].data();
    }
    clip_.resize(static_cast<std::size_t>(count) * 4);
    inputs_.resize(static_cast<std::size_t>(count) * 4 * rs_.num_inputs + 4);
    state_.program->RunVertices(state_, count, ptrs, clip_.data(), inputs_.data());
}

// The vertex function's epilogue: clip mask with unordered compares, 1/w, viewport with fma.
void DrawRunner::PostVs(int count) {
    verts_.resize(count);
    for (int i = 0; i < count; ++i) {
        PVertex &v = verts_[i];
        const float *c = &clip_[static_cast<std::size_t>(i) * 4];
        const float x = c[0], y = c[1], z = c[2], w = c[3];
        std::uint32_t m = 0;
        if (ftz::Ugt(x, w)) m |= 1;
        if (ftz::Ugt(0.0f, ftz::Add(x, w))) m |= 2;
        if (ftz::Ugt(y, w)) m |= 4;
        if (ftz::Ugt(0.0f, ftz::Add(y, w))) m |= 8;
        if (ftz::Ugt(0.0f, ftz::Add(z, w))) m |= 16;
        if (ftz::Ugt(z, w)) m |= 32;
        v.clipmask = m;
        v.vertex_id = kUndefinedVertexId;
        std::memcpy(v.clip, c, 16);
        const float oow = ftz::Div(1.0f, w);
        v.data[0][0] = ftz::Fma(ftz::Mul(x, oow), rs_.scale[0], rs_.translate[0]);
        v.data[0][1] = ftz::Fma(ftz::Mul(y, oow), rs_.scale[1], rs_.translate[1]);
        v.data[0][2] = ftz::Fma(ftz::Mul(z, oow), rs_.scale[2], rs_.translate[2]);
        v.data[0][3] = oow;
        for (int s = 0; s < rs_.num_inputs; ++s) {
            std::memcpy(v.data[1 + s],
                        &inputs_[(static_cast<std::size_t>(i) * rs_.num_inputs + s) * 4], 16);
        }
    }
}

void DrawRunner::SetupHw(const float *v0, const float *v1, const float *v2, bool ieee) {
    SetupTriangle(rs_, kept_, scene_, ieee, reinterpret_cast<const float(*)[4]>(v0),
                  reinterpret_cast<const float(*)[4]>(v1), reinterpret_cast<const float(*)[4]>(v2));
}

// draw_pt_emit: the pending pipeline triangles first, then an exact copy of the vertices and
// lp_setup_draw_elements in index order.
void DrawRunner::Emit(const std::uint16_t *elts, int count) {
    FlushVertices(false);
    const int n = static_cast<int>(verts_.size());
    hw_.resize(static_cast<std::size_t>(n) * 4 * num_slots_);
    for (int i = 0; i < n; ++i) {
        std::memcpy(&hw_[static_cast<std::size_t>(i) * 4 * num_slots_], verts_[i].data,
                    16 * num_slots_);
    }
    const std::size_t stride = static_cast<std::size_t>(4) * num_slots_;
    for (int i = 2; i < count; i += 3) {
        SetupHw(&hw_[elts[i - 2] * stride], &hw_[elts[i - 1] * stride], &hw_[elts[i] * stride],
                false);
    }
}

void DrawRunner::FlushVertices(bool ieee) {
    if (vbuf_allocated_) {
        if (!vbuf_indices_.empty()) {
            const std::size_t stride = static_cast<std::size_t>(4) * num_slots_;
            for (std::size_t i = 2; i < vbuf_indices_.size(); i += 3) {
                SetupHw(&vbuf_vertices_[vbuf_indices_[i - 2] * stride],
                        &vbuf_vertices_[vbuf_indices_[i - 1] * stride],
                        &vbuf_vertices_[vbuf_indices_[i] * stride], ieee);
            }
            vbuf_indices_.clear();
        }
        if (vbuf_nr_vertices_) {
            for (PVertex &t : tmps_) {
                t.vertex_id = kUndefinedVertexId;
            }
            if (in_pipeline_) {
                for (PVertex &v : verts_) {
                    v.vertex_id = kUndefinedVertexId;
                }
            }
        }
        vbuf_allocated_ = false;
        vbuf_nr_vertices_ = 0;
    }
    vbuf_first_ = true;
}

std::uint16_t DrawRunner::EmitVertex(PVertex *v) {
    if (v->vertex_id == kUndefinedVertexId) {
        const std::size_t stride = static_cast<std::size_t>(4) * num_slots_;
        vbuf_vertices_.resize((vbuf_nr_vertices_ + 1) * stride);
        std::memcpy(&vbuf_vertices_[vbuf_nr_vertices_ * stride], v->data, 16 * num_slots_);
        v->vertex_id = static_cast<std::uint32_t>(vbuf_nr_vertices_++);
    }
    return static_cast<std::uint16_t>(v->vertex_id);
}

// vbuf_first_tri and vbuf_tri. A flush resets the stage to vbuf_first_tri, so the triangle after
// a full buffer starts yet another buffer of its own.
void DrawRunner::VbufTriangle(PVertex *v0, PVertex *v1, PVertex *v2) {
    if (vbuf_first_) {
        FlushVertices(false);
        vbuf_allocated_ = true;
        vbuf_nr_vertices_ = 0;
        vbuf_first_ = false;
    }
    if (vbuf_nr_vertices_ + 3 > max_vertices_ ||
        static_cast<int>(vbuf_indices_.size()) + 3 > kVbufIndices) {
        FlushVertices(false);
        vbuf_allocated_ = true;
        vbuf_nr_vertices_ = 0;
    }
    vbuf_indices_.push_back(EmitVertex(v0));
    vbuf_indices_.push_back(EmitVertex(v1));
    vbuf_indices_.push_back(EmitVertex(v2));
}

// draw_pipe_cull.c.
void DrawRunner::CullTriangle(PVertex *v0, PVertex *v1, PVertex *v2) {
    if (rs_.cull_face == 0) {
        VbufTriangle(v0, v1, v2);
        return;
    }
    const float ex = ftz::Sub(v0->data[0][0], v2->data[0][0]);
    const float ey = ftz::Sub(v0->data[0][1], v2->data[0][1]);
    const float fx = ftz::Sub(v1->data[0][0], v2->data[0][0]);
    const float fy = ftz::Sub(v1->data[0][1], v2->data[0][1]);
    const float det = ftz::Sub(ftz::Mul(ex, fy), ftz::Mul(ey, fx));
    if (!ftz::Oeq(det, 0.0f)) {
        const bool ccw = ftz::Olt(det, 0.0f);
        const std::uint32_t face = (ccw == rs_.front_ccw) ? 1u : 2u;
        if ((face & rs_.cull_face) == 0) {
            VbufTriangle(v0, v1, v2);
        }
    } else if ((2u & rs_.cull_face) == 0) {
        VbufTriangle(v0, v1, v2);
    }
}

// interp of draw_pipe_clip.c: LINTERP(t, out, in) = out + t * (in - out), the C window transform
// without fma.
void DrawRunner::Interp(PVertex *dst, float t, const PVertex *out, const PVertex *in) {
    dst->clipmask = 0;
    dst->vertex_id = kUndefinedVertexId;
    for (int k = 0; k < 4; ++k) {
        dst->clip[k] = ftz::Add(out->clip[k], ftz::Mul(t, ftz::Sub(in->clip[k], out->clip[k])));
    }
    const float *pos = dst->clip;
    const float oow = ftz::Div(1.0f, pos[3]);
    for (int k = 0; k < 3; ++k) {
        dst->data[0][k] = ftz::Add(ftz::Mul(ftz::Mul(pos[k], oow), rs_.scale[k]), rs_.translate[k]);
    }
    dst->data[0][3] = oow;
    bool any_linear = false;
    for (int s = 0; s < rs_.num_inputs; ++s) {
        if (rs_.interp[s] == Interp::kPerspective) {
            for (int k = 0; k < 4; ++k) {
                dst->data[1 + s][k] =
                    ftz::Add(out->data[1 + s][k],
                             ftz::Mul(t, ftz::Sub(in->data[1 + s][k], out->data[1 + s][k])));
            }
        } else {
            any_linear = true;
        }
    }
    if (any_linear) {
        float tn = t;
        for (int k = 0; k < 2; ++k) {
            if (!ftz::Oeq(in->clip[k], out->clip[k])) {
                const float in_c = ftz::Div(in->clip[k], in->clip[3]);
                const float out_c = ftz::Div(out->clip[k], out->clip[3]);
                const float dst_c = ftz::Div(dst->clip[k], dst->clip[3]);
                tn = ftz::Div(ftz::Sub(dst_c, out_c), ftz::Sub(in_c, out_c));
                break;
            }
        }
        for (int s = 0; s < rs_.num_inputs; ++s) {
            if (rs_.interp[s] == Interp::kLinear) {
                for (int k = 0; k < 4; ++k) {
                    dst->data[1 + s][k] =
                        ftz::Add(out->data[1 + s][k],
                                 ftz::Mul(tn, ftz::Sub(in->data[1 + s][k], out->data[1 + s][k])));
                }
            }
        }
    }
}

// The frustum planes of draw_context.c in bit order, clip_halfz false.
constexpr float kPlanes[6][4] = {
    {-1, 0, 0, 1}, {1, 0, 0, 1}, {0, -1, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {0, 0, -1, 1},
};

inline float ClipDist(const PVertex *v, int plane) {
    const float *p = kPlanes[plane];
    const float *c = v->clip;
    return ftz::Add(
        ftz::Add(ftz::Add(ftz::Mul(c[0], p[0]), ftz::Mul(c[1], p[1])), ftz::Mul(c[2], p[2])),
        ftz::Mul(c[3], p[3]));
}

// do_clip_tri: Sutherland-Hodgman in plane order, interpolating from the nearer endpoint, then a
// fan (L[i-1], L[i], L[0]).
void DrawRunner::ClipTriangle(PVertex *v0, PVertex *v1, PVertex *v2, std::uint32_t clipmask) {
    PVertex *a[kMaxClippedVertices];
    PVertex *b[kMaxClippedVertices];
    PVertex **inlist = a;
    PVertex **outlist = b;
    int tmpnr = 0;
    int n = 3;
    inlist[0] = v0;
    inlist[1] = v1;
    inlist[2] = v2;
    while (clipmask && n >= 3) {
        int plane = 0;
        while (!(clipmask & (1u << plane))) {
            ++plane;
        }
        PVertex *vert_prev = inlist[0];
        float dp_prev = ClipDist(vert_prev, plane);
        clipmask &= ~(1u << plane);
        if (IsInfOrNan(dp_prev)) {
            return;
        }
        if (n >= kMaxClippedVertices) {
            return;
        }
        inlist[n] = inlist[0];
        int outcount = 0;
        for (int i = 1; i <= n; ++i) {
            PVertex *vert = inlist[i];
            const float dp = ClipDist(vert, plane);
            if (IsInfOrNan(dp)) {
                return;
            }
            bool different_sign;
            if (ftz::Oge(dp_prev, 0.0f)) {
                if (outcount >= kMaxClippedVertices) {
                    return;
                }
                outlist[outcount++] = vert_prev;
                different_sign = ftz::Olt(dp, 0.0f);
            } else {
                different_sign = !ftz::Olt(dp, 0.0f);
            }
            if (different_sign) {
                if (tmpnr >= kMaxClippedVertices + 1 || outcount >= kMaxClippedVertices) {
                    return;
                }
                PVertex *nv = &tmps_[tmpnr++];
                outlist[outcount++] = nv;
                const float denom = ftz::Sub(dp, dp_prev);
                if (ftz::Olt(dp, 0.0f)) {
                    if (ftz::Olt(-dp, dp_prev)) {
                        Interp(nv, ftz::Div(dp, denom), vert, vert_prev);
                    } else {
                        Interp(nv, ftz::Div(-dp_prev, denom), vert_prev, vert);
                    }
                } else {
                    if (ftz::Olt(-dp_prev, dp)) {
                        Interp(nv, ftz::Div(-dp_prev, denom), vert_prev, vert);
                    } else {
                        Interp(nv, ftz::Div(dp, denom), vert, vert_prev);
                    }
                }
            }
            vert_prev = vert;
            dp_prev = dp;
        }
        PVertex **tmp = inlist;
        inlist = outlist;
        outlist = tmp;
        n = outcount;
    }
    if (n >= 3) {
        for (int i = 2; i < n; ++i) {
            CullTriangle(inlist[i - 1], inlist[i], inlist[0]);
        }
    }
}

void DrawRunner::PipelineTriangle(PVertex *v0, PVertex *v1, PVertex *v2) {
    const std::uint32_t m = v0->clipmask | v1->clipmask | v2->clipmask;
    if (m == 0) {
        CullTriangle(v0, v1, v2);
    } else if ((v0->clipmask & v1->clipmask & v2->clipmask) == 0) {
        ClipTriangle(v0, v1, v2, m);
    }
}

// One middle-end run: fetch and shade the fetch list, then emit or the pipeline.
void DrawRunner::RunSegment(const std::uint32_t *fetch, int fetch_count, const std::uint16_t *elts,
                            int draw_count) {
    if (fetch_count <= 0) {
        return;
    }
    Fetch(fetch, fetch_count);
    PostVs(fetch_count);
    std::uint32_t clipped = 0;
    for (const PVertex &v : verts_) {
        clipped |= v.clipmask;
    }
    if (!clipped) {
        Emit(elts, draw_count);
        return;
    }
    in_pipeline_ = true;
    const std::uint32_t max_index = static_cast<std::uint32_t>(fetch_count - 1);
    for (int i = 0; i + 2 < draw_count; i += 3) {
        const std::uint32_t i0 = std::min<std::uint32_t>(elts[i], max_index);
        const std::uint32_t i1 = std::min<std::uint32_t>(elts[i + 1], max_index);
        const std::uint32_t i2 = std::min<std::uint32_t>(elts[i + 2], max_index);
        PipelineTriangle(&verts_[i0], &verts_[i1], &verts_[i2]);
    }
    in_pipeline_ = false;
}

inline int TrimTriangles(int count) { return count < 3 ? 0 : count - (count - 3) % 3; }

void DrawRunner::RunSegments() {
    const int count = TrimTriangles(input_.count);
    if (count < 3) {
        return;
    }
    std::vector<std::uint32_t> fetch;
    std::vector<std::uint16_t> elts;
    if (!input_.indices) {
        // vsplit_run_linear: segments of up to 4096 vertices, trimmed to whole triangles.
        const int seg_max = TrimTriangles(std::min(kMaxVerticesLinear, count));
        int seg_start = 0;
        while (seg_start < count) {
            const int n = count - seg_start > seg_max ? seg_max : count - seg_start;
            fetch.resize(n);
            elts.resize(n);
            for (int i = 0; i < n; ++i) {
                fetch[i] = static_cast<std::uint32_t>(input_.first + seg_start + i);
                elts[i] = static_cast<std::uint16_t>(i);
            }
            RunSegment(fetch.data(), n, elts.data(), n);
            seg_start += n;
        }
        return;
    }
    // vsplit_run_uint{8,16,32}: the whole-primitive shortcut is refused for GRF's draws (max_index
    // - min_index > count - 1 or uint32 indices longer than a segment), so segments go through the
    // 256-entry fetch cache.
    const std::uint32_t elt_max = static_cast<std::uint32_t>(
        input_.index_space / static_cast<std::size_t>(input_.index_size));
    const std::uint32_t istart = 0;
    auto get_idx = [&](std::uint32_t i) -> std::uint32_t {
        if (i >= elt_max) {
            return 0xFFFFFFFFu;
        }
        const std::uint8_t *p = input_.indices + static_cast<std::size_t>(i) * input_.index_size;
        if (input_.index_size == 4) {
            std::uint32_t v;
            std::memcpy(&v, p, 4);
            return v;
        }
        if (input_.index_size == 2) {
            std::uint16_t v;
            std::memcpy(&v, p, 2);
            return v;
        }
        return *p;
    };
    auto segment = [&](std::uint32_t start, int icount) {
        std::uint32_t fetches[kMapSize];
        std::uint16_t draws[kMapSize];
        std::memset(fetches, 0xFF, sizeof fetches);
        bool has_max_fetch = false;
        fetch.clear();
        elts.clear();
        for (int i = 0; i < icount; ++i) {
            const std::uint32_t base = start + static_cast<std::uint32_t>(i) < start
                                           ? 0xFFFFFFFFu
                                           : start + static_cast<std::uint32_t>(i);
            const std::uint32_t idx = get_idx(base);
            if (input_.index_size == 4 && idx == 0xFFFFFFFFu && !has_max_fetch) {
                fetches[idx % kMapSize] = 0;
                has_max_fetch = true;
            }
            const std::uint32_t hash = idx % kMapSize;
            if (fetches[hash] != idx) {
                fetches[hash] = idx;
                draws[hash] = static_cast<std::uint16_t>(fetch.size());
                fetch.push_back(idx);
            }
            elts.push_back(draws[hash]);
        }
        RunSegment(fetch.data(), static_cast<int>(fetch.size()), elts.data(),
                   static_cast<int>(elts.size()));
    };
    if (count <= kSegmentSize) {
        segment(istart + input_.first, count);
    } else {
        const int seg_max = TrimTriangles(std::min(kSegmentSize, count));
        int seg_start = 0;
        while (seg_start < count) {
            const int remaining = count - seg_start;
            const int n = remaining > seg_max ? seg_max : remaining;
            segment(istart + input_.first + seg_start, n);
            seg_start += n;
        }
    }
}

// Opens draw_vbo's FTZ and DAZ window around the work; the work reads everything from memory.
CUJU_GL_NOINLINE void RunSegmentsOf(DrawRunner *r) { r->RunSegments(); }
CUJU_GL_NOINLINE void FinalFlushOf(DrawRunner *r) { r->FinalFlush(); }

}  // namespace

void RunDraw(const DrawState &state, const RasterState &rs, const DrawInput &input, Scene &scene) {
    DrawRunner runner(state, rs, input, scene);
    runner.SetKept(&state);
    {
        FpScope scope(FpMode::kFtzDaz);
        RunSegmentsOf(&runner);
    }
    {
        FpScope scope(FpMode::kIeee);
        FinalFlushOf(&runner);
    }
}

}  // namespace cuju::render::gl
