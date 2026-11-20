// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The binned scene and its rasterization, ported from Mesa's llvmpipe (MIT licence): lp_scene.c,
// lp_rast.c (clears, lp_rast_shade_tile, lp_rast_shade_quads_mask), lp_rast_tri.c (the SSE
// functions lp_rast_triangle_32_3_16 and _32_3_4, build_masks_sse) and lp_rast_tri_tmp.h in its
// 64-bit and 32-bit forms. All coverage arithmetic is integer and mirrors llvmpipe's widths,
// including 32-bit wrap-around.

#include <algorithm>
#include <atomic>
#include <cstring>

#include "../fpenv.h"
#include "fragment.h"
#include "pipeline.h"

namespace cuju::render::gl {

/// ThreadPool.

ThreadPool::ThreadPool(int threads) {
    for (int i = 1; i < threads; ++i) {
        workers_.emplace_back([this] { Worker(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (std::thread &t : workers_) {
        t.join();
    }
}

bool ThreadPool::Take(Job *only, JobPtr *job, int *index) {
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        Job *j = it->get();
        if (only && j != only) continue;
        if (j->next < j->n) {
            *job = *it;
            *index = j->next++;
            if (j->next >= j->n) queue_.erase(it);
            return true;
        }
    }
    return false;
}

void ThreadPool::RunTask(const JobPtr &job, int index, std::unique_lock<std::mutex> &lock) {
    lock.unlock();
    job->fn(index);
    lock.lock();
    if (++job->done == job->n) done_cv_.notify_all();
}

void ThreadPool::Worker() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        JobPtr job;
        int index = 0;
        cv_.wait(lock, [&] { return stop_ || Take(nullptr, &job, &index); });
        if (!job) return;
        RunTask(job, index, lock);
    }
}

ThreadPool::JobPtr ThreadPool::Submit(int n, std::function<void(int)> fn) {
    auto job = std::make_shared<Job>();
    job->fn = std::move(fn);
    job->n = n;
    if (n <= 0) return job;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(job);
    }
    if (n == 1) {
        cv_.notify_one();
    } else {
        cv_.notify_all();
    }
    return job;
}

void ThreadPool::Wait(const JobPtr &job) {
    if (!job) return;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        JobPtr taken;
        int index = 0;
        if (Take(job.get(), &taken, &index)) {
            RunTask(taken, index, lock);
            continue;
        }
        if (job->done >= job->n) return;
        done_cv_.wait(lock);
    }
}

void ThreadPool::Start(int n, std::function<void(int)> fn) { started_ = Submit(n, std::move(fn)); }

void ThreadPool::Wait() {
    Wait(started_);
    started_.reset();
}

void ThreadPool::Run(int n, const std::function<void(int)> &fn) {
    if (workers_.empty()) {
        for (int i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }
    Wait(Submit(n, [&fn](int i) { fn(i); }));
}

// Scene.

void Scene::Begin(const TargetInfo &target) {
    target_ = target;
    tiles_x_ = (target.width + kTileSize - 1) / kTileSize;
    tiles_y_ = (target.height + kTileSize - 1) / kTileSize;
    const std::size_t tiles = static_cast<std::size_t>(tiles_x_) * tiles_y_;
    if (bins_.size() == tiles) {
        for (auto &b : bins_) b.clear();
    } else {
        bins_.assign(tiles, {});
    }
    triangles_binned = 0;
    triangles_.clear();
    coefs_.clear();
    states_.clear();
    clear_values_.clear();
    empty_ = true;
}

const DrawState *Scene::Keep(std::shared_ptr<const DrawState> state) {
    if (states_.empty() || states_.back() != state) {
        states_.push_back(std::move(state));
    }
    return states_.back().get();
}

std::uint32_t Scene::NewTriangle(const DrawState *state, int num_slots, bool frontfacing) {
    SceneTriangle t{};
    t.state = state;
    t.coef = static_cast<std::uint32_t>(coefs_.size());
    t.num_slots = static_cast<std::uint8_t>(num_slots);
    t.frontfacing = frontfacing;
    coefs_.resize(coefs_.size() + 12 * static_cast<std::size_t>(num_slots));
    triangles_.push_back(t);
    return static_cast<std::uint32_t>(triangles_.size() - 1);
}

void Scene::Bin(int tx, int ty, Op op, std::uint16_t arg, std::uint32_t index) {
    bins_[static_cast<std::size_t>(ty) * tiles_x_ + tx].push_back(Cmd{op, arg, index});
    empty_ = false;
}

void Scene::ClearColor(int buffer, const std::uint8_t *value, int bytes) {
    const std::uint32_t index = static_cast<std::uint32_t>(clear_values_.size());
    clear_values_.insert(clear_values_.end(), value, value + bytes);
    clear_values_.resize(clear_values_.size() + (8 - bytes));
    for (int t = 0; t < tiles_x_ * tiles_y_; ++t) {
        bins_[t].push_back(Cmd{Op::kClearColor, static_cast<std::uint16_t>(buffer), index});
    }
    empty_ = false;
}

void Scene::ClearDepth(std::uint16_t value) {
    const std::uint32_t index = static_cast<std::uint32_t>(clear_values_.size());
    clear_values_.push_back(static_cast<std::uint8_t>(value & 0xFF));
    clear_values_.push_back(static_cast<std::uint8_t>(value >> 8));
    for (int t = 0; t < tiles_x_ * tiles_y_; ++t) {
        bins_[t].push_back(Cmd{Op::kClearDepth, 0, index});
    }
    empty_ = false;
}

namespace {

// A tile being rasterized, lp_rasterizer_task.
struct Task {
    const TargetInfo *target;
    int x, y;           // tile origin
    int width, height;  // inside the framebuffer
    std::uint64_t blocks = 0;
};

inline std::int32_t W32(std::int64_t v) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(static_cast<std::uint64_t>(v)));
}
inline std::int32_t Add32(std::int32_t a, std::int32_t b) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) + static_cast<std::uint32_t>(b));
}
inline std::int32_t Mul32(std::int32_t a, std::int32_t b) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) * static_cast<std::uint32_t>(b));
}
inline std::int32_t Shl32(std::int32_t a, int s) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) << s);
}

// build_mask_linear_sse: bit 4 * row + col set when c + col * dcdx + row * dcdy is negative in
// 32-bit arithmetic.
inline unsigned MaskLinear(std::int32_t c, std::int32_t dcdx, std::int32_t dcdy) {
    unsigned mask = 0;
    for (int row = 0; row < 4; ++row) {
        const std::int32_t r = Add32(c, Mul32(row, dcdy));
        for (int col = 0; col < 4; ++col) {
            if (Add32(r, Mul32(col, dcdx)) < 0) {
                mask |= 1u << (4 * row + col);
            }
        }
    }
    return mask;
}

inline void BuildMasks(std::int32_t c, std::int32_t cdiff, std::int32_t dcdx, std::int32_t dcdy,
                       unsigned *outmask, unsigned *partmask) {
    *outmask |= MaskLinear(c, dcdx, dcdy);
    *partmask |= MaskLinear(Add32(c, cdiff), dcdx, dcdy);
}

inline int Ffs(unsigned v) {
    int i = 0;
    while (!(v & 1u)) {
        v >>= 1;
        ++i;
    }
    return i;
}

void ShadeBlock(Task &task, const SceneTriangle &tri, const float *coefs, int x, int y,
                unsigned mask) {
    if ((x % kTileSize) >= task.width || (y % kTileSize) >= task.height) {
        return;
    }
    const TargetInfo &t = *task.target;
    FragmentBlock b;
    b.x = x;
    b.y = y;
    b.coverage = mask & 0xFFFFu;
    b.frontfacing = tri.frontfacing;
    b.a0 = reinterpret_cast<const float(*)[4]>(coefs);
    b.dadx = reinterpret_cast<const float(*)[4]>(coefs + 4 * tri.num_slots);
    b.dady = reinterpret_cast<const float(*)[4]>(coefs + 8 * tri.num_slots);
    b.state = tri.state;
    for (int i = 0; i < t.num_color; ++i) {
        const Surface &s = t.color[i];
        if (s.data) {
            b.color[i] = s.data + static_cast<std::size_t>(y) * s.stride +
                         static_cast<std::size_t>(x) * BytesPerTexel(s.format);
            b.color_stride[i] = s.stride;
        }
    }
    if (t.depth.data) {
        b.depth = t.depth.data + static_cast<std::size_t>(y) * t.depth.stride +
                  static_cast<std::size_t>(x) * 2;
        b.depth_stride = t.depth.stride;
    }
    tri.state->program->ShadeBlock(b);
    ++task.blocks;
}

// lp_rast_tri_tmp.h with RASTER_64, do_block_4.
void Block4_64(Task &task, const SceneTriangle &tri, const float *coefs, const Plane *plane, int n,
               int x, int y, const std::int64_t *c) {
    unsigned mask = 0xFFFF;
    for (int j = 0; j < n; ++j) {
        mask &= ~MaskLinear(W32((c[j] - 1) >> kFixedOrder), (-plane[j].dcdx) >> kFixedOrder,
                            plane[j].dcdy >> kFixedOrder);
    }
    if (mask) {
        ShadeBlock(task, tri, coefs, x, y, mask);
    }
}

// lp_rast_tri_tmp.h without RASTER_64, do_block_4.
void Block4_32(Task &task, const SceneTriangle &tri, const float *coefs, const Plane *plane, int n,
               int x, int y, const std::int64_t *c) {
    unsigned mask = 0xFFFF;
    for (int j = 0; j < n; ++j) {
        mask &=
            ~MaskLinear(W32(c[j] - 1),
                        static_cast<std::int32_t>(0u - static_cast<std::uint32_t>(plane[j].dcdx)),
                        plane[j].dcdy);
    }
    if (mask) {
        ShadeBlock(task, tri, coefs, x, y, mask);
    }
}

void BlockFull16(Task &task, const SceneTriangle &tri, const float *coefs, int x, int y) {
    for (int iy = 0; iy < 16; iy += 4) {
        for (int ix = 0; ix < 16; ix += 4) {
            ShadeBlock(task, tri, coefs, x + ix, y + iy, 0xFFFF);
        }
    }
}

void Block16(Task &task, const SceneTriangle &tri, const float *coefs, const Plane *plane, int n,
             int x, int y, const std::int64_t *c, bool raster64) {
    unsigned outmask = 0, partmask = 0;
    for (int j = 0; j < n; ++j) {
        if (raster64) {
            std::int32_t dcdx = (-plane[j].dcdx) >> kFixedOrder;
            std::int32_t dcdy = plane[j].dcdy >> kFixedOrder;
            const std::int32_t cox = static_cast<std::int32_t>(plane[j].eo >> kFixedOrder);
            const std::int32_t ei = Shl32(Add32(Add32(dcdy, dcdx), -cox), 2);
            const std::int32_t cox_s = Shl32(cox, 2);
            const std::int32_t co = Add32(W32(c[j] >> kFixedOrder), cox_s);
            const std::int32_t cdiff =
                Add32(Add32(ei, -cox_s),
                      Add32(W32((c[j] - 1) >> kFixedOrder), -W32(c[j] >> kFixedOrder)));
            dcdx = Shl32(dcdx, 2);
            dcdy = Shl32(dcdy, 2);
            BuildMasks(co, cdiff, dcdx, dcdy, &outmask, &partmask);
        } else {
            const std::int64_t dcdx = -static_cast<std::int64_t>(plane[j].dcdx) * 4;
            const std::int64_t dcdy = static_cast<std::int64_t>(plane[j].dcdy) * 4;
            const std::int64_t cox = static_cast<std::int64_t>(plane[j].eo) * 4;
            const std::int32_t ei = W32(static_cast<std::int64_t>(plane[j].dcdy) - plane[j].dcdx -
                                        static_cast<std::int64_t>(plane[j].eo));
            const std::int64_t cio = static_cast<std::int64_t>(ei) * 4 - 1;
            const std::int32_t co = W32(c[j] + cox);
            const std::int32_t cdiff = W32(cio - cox);
            BuildMasks(co, cdiff, W32(dcdx), W32(dcdy), &outmask, &partmask);
        }
    }
    if (outmask == 0xFFFF) {
        return;
    }
    unsigned inmask = ~partmask & 0xFFFF;
    unsigned partial = partmask & ~outmask;
    while (partial) {
        const int i = Ffs(partial);
        const int ix = (i & 3) * 4;
        const int iy = (i >> 2) * 4;
        partial &= ~(1u << i);
        std::int64_t cx[kMaxPlanes];
        for (int j = 0; j < n; ++j) {
            cx[j] = c[j] - static_cast<std::int64_t>(plane[j].dcdx) * ix +
                    static_cast<std::int64_t>(plane[j].dcdy) * iy;
        }
        if (raster64) {
            Block4_64(task, tri, coefs, plane, n, x + ix, y + iy, cx);
        } else {
            Block4_32(task, tri, coefs, plane, n, x + ix, y + iy, cx);
        }
    }
    while (inmask) {
        const int i = Ffs(inmask);
        const int ix = (i & 3) * 4;
        const int iy = (i >> 2) * 4;
        inmask &= ~(1u << i);
        ShadeBlock(task, tri, coefs, x + ix, y + iy, 0xFFFF);
    }
}

// lp_rast_triangle_N and lp_rast_triangle_32_N of lp_rast_tri_tmp.h.
void TriangleTile(Task &task, const SceneTriangle &tri, const float *coefs, unsigned plane_mask,
                  bool raster64) {
    Plane plane[kMaxPlanes];
    std::int64_t c[kMaxPlanes];
    unsigned outmask = 0, partmask = 0;
    int n = 0;
    const int x = task.x, y = task.y;
    while (plane_mask) {
        const int i = Ffs(plane_mask);
        plane_mask &= ~(1u << i);
        plane[n] = tri.plane[i];
        c[n] = plane[n].c + static_cast<std::int64_t>(plane[n].dcdy) * y -
               static_cast<std::int64_t>(plane[n].dcdx) * x;
        if (raster64) {
            std::int32_t dcdx = (-plane[n].dcdx) >> kFixedOrder;
            std::int32_t dcdy = plane[n].dcdy >> kFixedOrder;
            const std::int32_t cox = static_cast<std::int32_t>(plane[n].eo >> kFixedOrder);
            const std::int32_t ei = Shl32(Add32(Add32(dcdy, dcdx), -cox), 4);
            const std::int32_t cox_s = Shl32(cox, 4);
            const std::int32_t co = Add32(W32(c[n] >> kFixedOrder), cox_s);
            const std::int32_t cdiff =
                Add32(Add32(ei, -cox_s),
                      Add32(W32((c[n] - 1) >> kFixedOrder), -W32(c[n] >> kFixedOrder)));
            dcdx = Shl32(dcdx, 4);
            dcdy = Shl32(dcdy, 4);
            BuildMasks(co, cdiff, dcdx, dcdy, &outmask, &partmask);
        } else {
            const std::int32_t dcdx =
                Shl32(static_cast<std::int32_t>(0u - static_cast<std::uint32_t>(plane[n].dcdx)), 4);
            const std::int32_t dcdy = Shl32(plane[n].dcdy, 4);
            const std::int32_t cox = Shl32(static_cast<std::int32_t>(plane[n].eo), 4);
            const std::int32_t ei = Add32(Add32(plane[n].dcdy, -plane[n].dcdx),
                                          -static_cast<std::int32_t>(plane[n].eo));
            const std::int32_t cio = Add32(Shl32(ei, 4), -1);
            const std::int32_t co = W32(c[n] + cox);
            const std::int32_t cdiff = Add32(cio, -cox);
            BuildMasks(co, cdiff, dcdx, dcdy, &outmask, &partmask);
        }
        ++n;
    }
    if (outmask == 0xFFFF) {
        return;
    }
    unsigned inmask = ~partmask & 0xFFFF;
    unsigned partial = partmask & ~outmask;
    while (partial) {
        const int i = Ffs(partial);
        const int ix = (i & 3) * 16;
        const int iy = (i >> 2) * 16;
        std::int64_t cx[kMaxPlanes];
        for (int j = 0; j < n; ++j) {
            cx[j] = c[j] - static_cast<std::int64_t>(plane[j].dcdx) * ix +
                    static_cast<std::int64_t>(plane[j].dcdy) * iy;
        }
        partial &= ~(1u << i);
        Block16(task, tri, coefs, plane, n, x + ix, y + iy, cx, raster64);
    }
    while (inmask) {
        const int i = Ffs(inmask);
        const int ix = (i & 3) * 16;
        const int iy = (i >> 2) * 16;
        inmask &= ~(1u << i);
        BlockFull16(task, tri, coefs, x + ix, y + iy);
    }
}

// The per-pixel values of lp_rast_triangle_32_3_16 and _32_3_4 for one plane at block (x, y):
// low 32 bits of c, plus (-dcdx) * x + dcdy * y, minus 1, in 32-bit arithmetic.
inline std::int32_t Origin32(const Plane &p, int x, int y) {
    std::int32_t c = W32(p.c);
    const std::int32_t ndcdx = static_cast<std::int32_t>(0u - static_cast<std::uint32_t>(p.dcdx));
    c = Add32(c, Mul32(ndcdx, x));
    c = Add32(c, Mul32(p.dcdy, y));
    return Add32(c, -1);
}

// The covered pixels of a 4x4 block at (bx, by) from per-plane origins at the block: a pixel is
// out when any plane's value is negative.
inline unsigned Outside4(const std::int32_t *c, const Plane *plane, int n) {
    unsigned out = 0;
    for (int j = 0; j < n; ++j) {
        const std::int32_t ndcdx =
            static_cast<std::int32_t>(0u - static_cast<std::uint32_t>(plane[j].dcdx));
        out |= MaskLinear(c[j], ndcdx, plane[j].dcdy);
    }
    return out;
}

// lp_rast_triangle_32_3_16 (SSE).
void Triangle32_3_16(Task &task, const SceneTriangle &tri, const float *coefs, unsigned arg) {
    const int x = static_cast<int>(arg & 0xFF) + task.x;
    const int y = static_cast<int>(arg >> 8) + task.y;
    const Plane *plane = tri.plane;
    std::int32_t c[3], rej4[3];
    for (int j = 0; j < 3; ++j) {
        const bool dcdx_neg = plane[j].dcdx < 0;
        const bool dcdy_neg = plane[j].dcdy < 0;
        const std::int32_t r =
            static_cast<std::int32_t>((dcdy_neg ? 0u : static_cast<std::uint32_t>(plane[j].dcdy)) -
                                      (dcdx_neg ? static_cast<std::uint32_t>(plane[j].dcdx) : 0u));
        c[j] = Origin32(plane[j], x, y);
        rej4[j] = Add32(Shl32(r, 2), 1);
    }
    struct Out {
        unsigned mask;
        int i, j;
    } out[16];
    int nr = 0;
    std::int32_t crow[3] = {c[0], c[1], c[2]};
    for (int i = 0; i < 4; ++i) {
        std::int32_t cx[3] = {crow[0], crow[1], crow[2]};
        for (int j = 0; j < 4; ++j) {
            bool reject = false;
            for (int k = 0; k < 3; ++k) {
                reject |= Add32(cx[k], rej4[k]) < 0;
            }
            if (!reject) {
                const unsigned mask = Outside4(cx, plane, 3);
                out[nr].i = i;
                out[nr].j = j;
                out[nr].mask = mask;
                if (mask != 0xFFFF) {
                    ++nr;
                }
            }
            for (int k = 0; k < 3; ++k) {
                cx[k] = Add32(
                    cx[k],
                    Shl32(static_cast<std::int32_t>(0u - static_cast<std::uint32_t>(plane[k].dcdx)),
                          2));
            }
        }
        for (int k = 0; k < 3; ++k) {
            crow[k] = Add32(crow[k], Shl32(plane[k].dcdy, 2));
        }
    }
    for (int i = 0; i < nr; ++i) {
        ShadeBlock(task, tri, coefs, x + 4 * out[i].j, y + 4 * out[i].i, 0xFFFF & ~out[i].mask);
    }
}

// lp_rast_triangle_32_3_4 (SSE).
void Triangle32_3_4(Task &task, const SceneTriangle &tri, const float *coefs, unsigned arg) {
    const int x = static_cast<int>(arg & 0xFF) + task.x;
    const int y = static_cast<int>(arg >> 8) + task.y;
    std::int32_t c[3];
    for (int j = 0; j < 3; ++j) {
        c[j] = Origin32(tri.plane[j], x, y);
    }
    const unsigned mask = Outside4(c, tri.plane, 3);
    if (mask != 0xFFFF) {
        ShadeBlock(task, tri, coefs, x, y, 0xFFFF & ~mask);
    }
}

// lp_rast_triangle_32_4_16, the TRI_16 of lp_rast_tri_tmp.h: values scaled by 4 in 32 bits.
void Triangle32_4_16(Task &task, const SceneTriangle &tri, const float *coefs, unsigned arg) {
    const Plane *plane = tri.plane;
    int x = static_cast<int>(arg & 0xFF);
    int y = static_cast<int>(arg >> 8);
    unsigned outmask = 0;
    static const unsigned kRow3 = 0xF000u, kRow2 = 0x0F00u, kRow1 = 0x00F0u, kRow0 = 0x000Fu;
    static const unsigned kCol0 = 0x1111u, kCol1 = 0x2222u, kCol2 = 0x4444u, kCol3 = 0x8888u;
    static const unsigned kBottom[4] = {kRow3, kRow3 | kRow2, kRow3 | kRow2 | kRow1,
                                        kRow3 | kRow2 | kRow1 | kRow0};
    static const unsigned kRight[4] = {kCol3, kCol3 | kCol2, kCol3 | kCol2 | kCol1,
                                       kCol3 | kCol2 | kCol1 | kCol0};
    if (x + 12 >= 64) {
        outmask |= kRight[((x + 12) - 64) / 4];
    }
    if (y + 12 >= 64) {
        outmask |= kBottom[((y + 12) - 64) / 4];
    }
    x += task.x;
    y += task.y;
    std::int32_t dcdx4[4], dcdy4[4];
    for (int j = 0; j < 4; ++j) {
        dcdx4[j] =
            Mul32(static_cast<std::int32_t>(0u - static_cast<std::uint32_t>(plane[j].dcdx)), 4);
        dcdy4[j] = Mul32(plane[j].dcdy, 4);
        // const int c = plane.c + plane.dcdy * y - plane.dcdx * x, with int products.
        const std::int32_t c = W32(plane[j].c + Mul32(plane[j].dcdy, y) -
                                   static_cast<std::int64_t>(Mul32(plane[j].dcdx, x)));
        const std::int32_t cox = Mul32(static_cast<std::int32_t>(plane[j].eo), 4);
        outmask |= MaskLinear(Add32(c, cox), dcdx4[j], dcdy4[j]);
    }
    if (outmask == 0xFFFF) {
        return;
    }
    unsigned partial = 0xFFFF & ~outmask;
    while (partial) {
        const int i = Ffs(partial);
        const int ix = (i & 3) * 4;
        const int iy = (i >> 2) * 4;
        const int px = x + ix;
        const int py = y + iy;
        unsigned mask = 0xFFFF;
        partial &= ~(1u << i);
        for (int j = 0; j < 4; ++j) {
            const std::int64_t v = plane[j].c - 1 -
                                   static_cast<std::int64_t>(Mul32(plane[j].dcdx, px)) +
                                   static_cast<std::int64_t>(Mul32(plane[j].dcdy, py));
            const std::int32_t cx = W32(v * 4);
            mask &= ~MaskLinear(cx, dcdx4[j], dcdy4[j]);
        }
        if (mask) {
            ShadeBlock(task, tri, coefs, px, py, mask);
        }
    }
}

void ShadeTile(Task &task, const SceneTriangle &tri, const float *coefs) {
    for (int y = 0; y < task.height; y += 4) {
        for (int x = 0; x < task.width; x += 4) {
            ShadeBlock(task, tri, coefs, task.x + x, task.y + y, 0xFFFF);
        }
    }
}

}  // namespace

namespace {

// Writes the texel value v of bpp bytes to every texel of the task's tile: one row built once,
// then copied to each row.
void FillTile(const Task &task, std::uint8_t *data, std::size_t stride, int bpp,
              const std::uint8_t *v) {
    std::uint8_t row[kTileSize * 8];
    const int bytes = task.width * bpp;
    for (int x = 0; x < task.width; ++x) {
        std::memcpy(row + x * bpp, v, bpp);
    }
    for (int y = 0; y < task.height; ++y) {
        std::memcpy(data + static_cast<std::size_t>(task.y + y) * stride +
                        static_cast<std::size_t>(task.x) * bpp,
                    row, bytes);
    }
}

}  // namespace

void Scene::RasterizeTile(int tile, std::uint64_t *blocks) {
    Task task;
    task.target = &target_;
    task.x = (tile % tiles_x_) * kTileSize;
    task.y = (tile / tiles_x_) * kTileSize;
    task.width = std::min(kTileSize, target_.width - task.x);
    task.height = std::min(kTileSize, target_.height - task.y);
    for (const Cmd &cmd : bins_[tile]) {
        switch (cmd.op) {
            case Op::kClearColor: {
                const Surface &s = target_.color[cmd.arg];
                FillTile(task, s.data, s.stride, BytesPerTexel(s.format),
                         clear_values_.data() + cmd.index);
                break;
            }
            case Op::kClearDepth: {
                const Surface &s = target_.depth;
                FillTile(task, s.data, s.stride, 2, clear_values_.data() + cmd.index);
                break;
            }
            default: {
                const SceneTriangle &tri = triangles_[cmd.index];
                const float *coefs = coefs_.data() + tri.coef;
                switch (cmd.op) {
                    case Op::kTri64:
                        TriangleTile(task, tri, coefs, cmd.arg, true);
                        break;
                    case Op::kTri32:
                        TriangleTile(task, tri, coefs, cmd.arg, false);
                        break;
                    case Op::kTri32_3_16:
                        Triangle32_3_16(task, tri, coefs, cmd.arg);
                        break;
                    case Op::kTri32_3_4:
                        Triangle32_3_4(task, tri, coefs, cmd.arg);
                        break;
                    case Op::kTri32_4_16:
                        Triangle32_4_16(task, tri, coefs, cmd.arg);
                        break;
                    case Op::kShadeTile:
                        ShadeTile(task, tri, coefs);
                        break;
                    default:
                        break;
                }
            }
        }
    }
    *blocks += task.blocks;
}

namespace {

// Rasterizes tiles inside the FTZ and DAZ environment of llvmpipe's rasterizer threads. The scope
// is opened here and the work happens in a function that reads everything from memory.
CUJU_GL_NOINLINE void RunTile(Scene *scene, int tile, std::atomic<std::uint64_t> *blocks,
                              void (Scene::*fn)(int, std::uint64_t *)) {
    std::uint64_t n = 0;
    (scene->*fn)(tile, &n);
    blocks->fetch_add(n, std::memory_order_relaxed);
}

}  // namespace

void Scene::Rasterize(ThreadPool *pool, std::uint64_t *blocks_shaded) {
    if (empty_) {
        return;
    }
    std::atomic<std::uint64_t> blocks{0};
    const int n = tiles_x_ * tiles_y_;
    const std::function<void(int)> fn = [this, &blocks](int tile) {
        FpScope scope(FpMode::kFtzDaz);
        RunTile(this, tile, &blocks, &Scene::RasterizeTile);
    };
    if (pool) {
        pool->Run(n, fn);
    } else {
        for (int i = 0; i < n; ++i) {
            fn(i);
        }
    }
    blocks_.store(blocks.load());
    FinishRasterize(nullptr, blocks_shaded);
}

void Scene::StartRasterize(ThreadPool *pool) {
    blocks_.store(0);
    if (empty_) {
        return;
    }
    pool->Start(tiles_x_ * tiles_y_, [this](int tile) {
        FpScope scope(FpMode::kFtzDaz);
        RunTile(this, tile, &blocks_, &Scene::RasterizeTile);
    });
}

void Scene::FinishRasterize(ThreadPool *pool, std::uint64_t *blocks_shaded) {
    if (pool) {
        pool->Wait();
    }
    if (blocks_shaded) {
        *blocks_shaded += blocks_.load();
    }
    blocks_.store(0);
    for (auto &b : bins_) {
        b.clear();
    }
    triangles_.clear();
    coefs_.clear();
    states_.clear();
    clear_values_.clear();
    empty_ = true;
}

// SceneList.

void SceneList::Begin(const TargetInfo &target) {
    Reset();
    target_ = target;
    tiles_ = ((target.width + kTileSize - 1) / kTileSize) *
             ((target.height + kTileSize - 1) / kTileSize);
    begun_ = true;
}

bool SceneList::Empty() const {
    for (std::size_t i = 0; i < used_; ++i) {
        if (!parts_[i]->Empty()) return false;
    }
    return true;
}

void SceneList::Reset() {
    for (std::size_t i = 0; i < used_; ++i) {
        parts_[i]->FinishRasterize(nullptr, nullptr);
    }
    used_ = 0;
    begun_ = false;
}

Scene &SceneList::Add(bool direct) {
    if (used_ == parts_.size()) {
        parts_.push_back(std::make_unique<Scene>());
        direct_.push_back(direct);
    }
    direct_[used_] = direct;
    Scene &s = *parts_[used_++];
    s.Begin(target_);
    return s;
}

Scene &SceneList::Direct() {
    if (used_ > 0 && direct_[used_ - 1]) return *parts_[used_ - 1];
    return Add(true);
}

Scene &SceneList::NewPart() { return Add(false); }

std::uint64_t SceneList::TrianglesBinned() const {
    std::uint64_t n = 0;
    for (std::size_t i = 0; i < used_; ++i) n += parts_[i]->triangles_binned;
    return n;
}

void SceneList::RasterizeTile(int tile, std::uint64_t *blocks) {
    for (std::size_t i = 0; i < used_; ++i) {
        Scene *s = parts_[i].get();
        if (!s->Empty()) s->RasterizeTile(tile, blocks);
    }
}

namespace {

CUJU_GL_NOINLINE void RunListTile(SceneList *list, int tile, std::atomic<std::uint64_t> *blocks,
                                  void (SceneList::*fn)(int, std::uint64_t *)) {
    std::uint64_t n = 0;
    (list->*fn)(tile, &n);
    blocks->fetch_add(n, std::memory_order_relaxed);
}

}  // namespace

void SceneList::Rasterize(ThreadPool *pool, std::uint64_t *blocks_shaded) {
    if (Empty()) {
        Reset();
        return;
    }
    blocks_.store(0);
    const std::function<void(int)> fn = [this](int tile) {
        FpScope scope(FpMode::kFtzDaz);
        RunListTile(this, tile, &blocks_, &SceneList::RasterizeTile);
    };
    if (pool) {
        pool->Run(tiles_, fn);
    } else {
        for (int i = 0; i < tiles_; ++i) fn(i);
    }
    FinishRasterize(nullptr, blocks_shaded);
}

void SceneList::StartRasterize(ThreadPool *pool) {
    blocks_.store(0);
    if (Empty()) return;
    job_ = pool->Submit(tiles_, [this](int tile) {
        FpScope scope(FpMode::kFtzDaz);
        RunListTile(this, tile, &blocks_, &SceneList::RasterizeTile);
    });
}

namespace {

void Writes(const TargetInfo &t, std::vector<MemorySpan> *out) {
    const auto add = [out](const Surface &s) {
        if (s.data)
            out->push_back({s.data, s.data + static_cast<std::size_t>(s.stride) * s.height});
    };
    for (int i = 0; i < t.num_color; ++i) add(t.color[i]);
    add(t.depth);
}

bool Overlap(const std::vector<MemorySpan> &a, const std::vector<MemorySpan> &b) {
    for (const MemorySpan &x : a)
        for (const MemorySpan &y : b)
            if (x.begin < y.end && y.begin < x.end) return true;
    return false;
}

}  // namespace

void SceneList::Reads(std::vector<MemorySpan> *out) const {
    const Texture *last = nullptr;
    for (std::size_t i = 0; i < used_; ++i) {
        for (const std::shared_ptr<const DrawState> &s : parts_[i]->states_) {
            for (const Texture *t : s->textures) {
                if (!t || t == last) continue;
                last = t;
                for (const TextureLevel &l : t->levels) {
                    const std::uint8_t *d = l.texels.data();
                    if (d) out->push_back({d, d + l.texels.size()});
                }
            }
        }
    }
}

bool SceneList::Conflict(const SceneList &a, const SceneList &b) {
    std::vector<MemorySpan> aw, bw, ar, br;
    Writes(a.target_, &aw);
    Writes(b.target_, &bw);
    if (Overlap(aw, bw)) return true;
    b.Reads(&br);
    if (Overlap(aw, br)) return true;
    a.Reads(&ar);
    return Overlap(bw, ar);
}

void SceneList::FinishRasterize(ThreadPool *pool, std::uint64_t *blocks_shaded) {
    if (pool && job_) pool->Wait(job_);
    job_.reset();
    if (blocks_shaded) *blocks_shaded += blocks_.load();
    blocks_.store(0);
    Reset();
}

}  // namespace cuju::render::gl
