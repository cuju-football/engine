// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Internals of the mini GL: the port of llvmpipe's draw module (fetch, vertex shader, clip mask,
// segments, draw_pipe_clip, draw_pipe_cull, the vbuf stage), its triangle setup and binning
// (lp_setup_tri.c, lp_state_setup.c) and its rasterizer (lp_rast_tri.c, lp_rast_tri_tmp.h,
// lp_rast.c). Mesa is under the MIT licence, Copyright VMware, Inc. and others.

#ifndef CUJU_RENDER_GL_PIPELINE_H
#define CUJU_RENDER_GL_PIPELINE_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "program.h"

#if defined(_MSC_VER) && !defined(__clang__)
#define CUJU_GL_NOINLINE __declspec(noinline)
#else
#define CUJU_GL_NOINLINE __attribute__((noinline))
#endif

namespace cuju::render::gl {
constexpr int kTileSize = 64;
constexpr int kTileOrder = 6;
constexpr int kFixedOrder = 8;
constexpr int kFixedOne = 1 << kFixedOrder;
constexpr int kMaxFixedLength32 = 128;
constexpr int kMaxPlanes = 8;

struct Plane {
    std::int64_t c;
    std::int32_t dcdx;
    std::int32_t dcdy;
    std::uint32_t eo;
};

struct SceneTriangle {
    const DrawState *state;
    std::uint32_t coef;
    std::uint8_t num_slots;
    std::uint8_t nr_planes;
    bool frontfacing;
    Plane plane[kMaxPlanes];
};

enum class Op : std::uint8_t {
    kClearColor,
    kClearDepth,
    kTri64,
    kTri32,
    kTri32_3_16,
    kTri32_3_4,
    kTri32_4_16,
    kShadeTile,
};

struct Cmd {
    Op op;
    std::uint16_t arg;
    std::uint32_t index;
};

class ThreadPool {
public:
    struct Job {
        std::function<void(int)> fn;
        int n = 0, next = 0, done = 0;
    };
    using JobPtr = std::shared_ptr<Job>;

    explicit ThreadPool(int threads);
    ~ThreadPool();
    int Threads() const { return static_cast<int>(workers_.size()); }
    JobPtr Submit(int n, std::function<void(int)> fn);
    void Wait(const JobPtr &job);
    void Run(int n, const std::function<void(int)> &fn);
    void Start(int n, std::function<void(int)> fn);
    void Wait();

private:
    void Worker();
    bool Take(Job *only, JobPtr *job, int *index);
    void RunTask(const JobPtr &job, int index, std::unique_lock<std::mutex> &lock);
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_, done_cv_;
    std::deque<JobPtr> queue_;
    JobPtr started_;
    bool stop_ = false;
};

class Scene {
public:
    void Begin(const TargetInfo &target);
    bool Empty() const { return empty_; }
    const TargetInfo &Target() const { return target_; }
    int TilesX() const { return tiles_x_; }
    int TilesY() const { return tiles_y_; }

    const DrawState *Keep(std::shared_ptr<const DrawState> state);
    std::uint32_t NewTriangle(const DrawState *state, int num_slots, bool frontfacing);
    SceneTriangle &Triangle(std::uint32_t i) { return triangles_[i]; }
    float *Coefs(std::uint32_t offset) { return coefs_.data() + offset; }
    void Bin(int tx, int ty, Op op, std::uint16_t arg, std::uint32_t index);
    void ClearColor(int buffer, const std::uint8_t *value, int bytes);
    void ClearDepth(std::uint16_t value);

    void Rasterize(ThreadPool *pool, std::uint64_t *blocks_shaded);
    void StartRasterize(ThreadPool *pool);
    void FinishRasterize(ThreadPool *pool, std::uint64_t *blocks_shaded);

    std::uint64_t triangles_binned = 0;

private:
    friend class SceneList;
    void RasterizeTile(int tile, std::uint64_t *blocks);

    TargetInfo target_;
    int tiles_x_ = 0, tiles_y_ = 0;
    bool empty_ = true;
    std::vector<std::vector<Cmd>> bins_;
    std::vector<SceneTriangle> triangles_;
    std::vector<float> coefs_;
    std::vector<std::shared_ptr<const DrawState>> states_;
    std::vector<std::uint8_t> clear_values_;
    std::atomic<std::uint64_t> blocks_{0};
};

struct MemorySpan {
    const std::uint8_t *begin, *end;
};

class SceneList {
public:
    void Begin(const TargetInfo &target);
    bool Begun() const { return begun_; }
    bool Empty() const;
    void Reset();
    Scene &Direct();
    Scene &NewPart();
    std::uint64_t TrianglesBinned() const;

    void Rasterize(ThreadPool *pool, std::uint64_t *blocks_shaded);
    void StartRasterize(ThreadPool *pool);
    void FinishRasterize(ThreadPool *pool, std::uint64_t *blocks_shaded);
    static bool Conflict(const SceneList &a, const SceneList &b);

private:
    void RasterizeTile(int tile, std::uint64_t *blocks);
    void Reads(std::vector<MemorySpan> *out) const;
    Scene &Add(bool direct);

    TargetInfo target_;
    bool begun_ = false;
    int tiles_ = 0;
    std::vector<std::unique_ptr<Scene>> parts_;
    std::vector<bool> direct_;
    std::size_t used_ = 0;
    std::atomic<std::uint64_t> blocks_{0};
    ThreadPool::JobPtr job_;
};

struct RasterState {
    int fb_width = 0, fb_height = 0;
    bool bottom_edge_rule = false;
    std::uint32_t cull_face = 0;
    bool front_ccw = true;
    float scale[3] = {0, 0, 0};
    float translate[3] = {0, 0, 0};
    int num_inputs = 0;
    Interp interp[kMaxInputs] = {};
};

struct AttributeSource {
    bool enabled = false;
    const std::uint8_t *data = nullptr;
    std::size_t buffer_size = 0;
    std::size_t offset = 0;
    int stride = 0;
    int size = 4;
    float generic[4] = {0, 0, 0, 1};
};

struct DrawInput {
    AttributeSource attribs[kMaxAttributes];
    const std::uint8_t *indices = nullptr;
    std::size_t index_space = 0;
    int index_size = 4;
    std::uint32_t min_index = 0, max_index = 0xFFFFFFFFu;
    int first = 0;
    int count = 0;
};

void RunDraw(const DrawState &state, const RasterState &rs, const DrawInput &input, Scene &scene);

void SetupTriangle(const RasterState &rs, const DrawState *state, Scene &scene, bool ieee,
                   const float (*v0)[4], const float (*v1)[4], const float (*v2)[4]);

}  // namespace cuju::render::gl

#endif
