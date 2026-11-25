// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// A deterministic CPU emulation of the subset of OpenGL that GRF uses, on top of a port of
// llvmpipe's draw and rasterization pipeline (Mesa 25.2.8). Entry points mirror the GL functions
// GRF calls, with GL's enums and argument order, so the scene port and the trace replayer drive it
// call by call. Object names are the context's own: a caller replaying GL keeps its own map.
//
// Rendering is deferred into a scene of 64x64 tiles, as llvmpipe does, and the scene is
// rasterized, tile-parallel, at the points where its results become visible: a framebuffer
// binding change, a texture change, a readback or Finish. Results never depend on the thread
// count, since every tile runs its commands in submission order.

#ifndef CUJU_RENDER_GL_CONTEXT_H
#define CUJU_RENDER_GL_CONTEXT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../texture.h"
#include "program.h"

namespace cuju::render::gl {
class MipCache;
class Scene;
class SceneList;
class ThreadPool;
struct DrawInput;
struct DrawChunk;

enum : std::uint32_t {
    GL_POINTS = 0x0000,
    GL_TRIANGLES = 0x0004,
    GL_FRONT = 0x0404,
    GL_BACK = 0x0405,
    GL_FRONT_AND_BACK = 0x0408,
    GL_CULL_FACE = 0x0B44,
    GL_DEPTH_TEST = 0x0B71,
    GL_BLEND = 0x0BE2,
    GL_MULTISAMPLE = 0x809D,
    GL_TEXTURE_2D = 0x0DE1,
    GL_UNSIGNED_BYTE = 0x1401,
    GL_UNSIGNED_SHORT = 0x1403,
    GL_UNSIGNED_INT = 0x1405,
    GL_FLOAT = 0x1406,
    GL_HALF_FLOAT = 0x140B,
    GL_DEPTH_COMPONENT = 0x1902,
    GL_RGB = 0x1907,
    GL_RGBA = 0x1908,
    GL_BGRA = 0x80E1,
    GL_UNSIGNED_INT_2_10_10_10_REV = 0x8368,
    GL_RGB8 = 0x8051,
    GL_RGBA8 = 0x8058,
    GL_SRGB8 = 0x8C41,
    GL_SRGB8_ALPHA8 = 0x8C43,
    GL_RGBA16F = 0x881A,
    GL_DEPTH_COMPONENT16 = 0x81A5,
    GL_NEVER = 0x0200,
    GL_LESS = 0x0201,
    GL_EQUAL = 0x0202,
    GL_LEQUAL = 0x0203,
    GL_GREATER = 0x0204,
    GL_NOTEQUAL = 0x0205,
    GL_GEQUAL = 0x0206,
    GL_ALWAYS = 0x0207,
    GL_ZERO = 0,
    GL_ONE = 1,
    GL_SRC_ALPHA = 0x0302,
    GL_ONE_MINUS_SRC_ALPHA = 0x0303,
    GL_DEPTH_BUFFER_BIT = 0x0100,
    GL_STENCIL_BUFFER_BIT = 0x0400,
    GL_COLOR_BUFFER_BIT = 0x4000,
    GL_ARRAY_BUFFER = 0x8892,
    GL_ELEMENT_ARRAY_BUFFER = 0x8893,
    GL_FRAMEBUFFER = 0x8D40,
    GL_DRAW_FRAMEBUFFER = 0x8CA9,
    GL_READ_FRAMEBUFFER = 0x8CA8,
    GL_COLOR_ATTACHMENT0 = 0x8CE0,
    GL_DEPTH_ATTACHMENT = 0x8D00,
    GL_FRAMEBUFFER_COMPLETE = 0x8CD5,
    GL_NONE = 0,
    GL_TEXTURE0 = 0x84C0,
    GL_TEXTURE_MIN_FILTER = 0x2801,
    GL_TEXTURE_MAG_FILTER = 0x2800,
    GL_TEXTURE_WRAP_S = 0x2802,
    GL_TEXTURE_WRAP_T = 0x2803,
    GL_TEXTURE_MAX_ANISOTROPY = 0x84FE,
    GL_TEXTURE_COMPARE_MODE = 0x884C,
    GL_TEXTURE_COMPARE_FUNC = 0x884D,
};

struct ContextOptions {
    int width = 1280;
    int height = 720;
    int threads = 1;
    std::shared_ptr<MipCache> mip_cache;
};

class Context {
public:
    explicit Context(const ContextOptions &options);
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    std::uint32_t GenTexture();
    void DeleteTexture(std::uint32_t name);
    void ActiveTexture(std::uint32_t unit);
    void BindTexture(std::uint32_t target, std::uint32_t name);
    void TexParameteri(std::uint32_t target, std::uint32_t pname, std::int32_t value);
    void TexParameterf(std::uint32_t target, std::uint32_t pname, float value);
    void TexImage2D(std::uint32_t target, int level, std::uint32_t internalformat, int width,
                    int height, std::uint32_t format, std::uint32_t type, const void *data);
    void TexSubImage2D(std::uint32_t target, int level, int x, int y, int width, int height,
                       std::uint32_t format, std::uint32_t type, const void *data);
    void GenerateMipmap(std::uint32_t target);
    const Texture *GetTexture(std::uint32_t name);

    std::uint32_t GenBuffer();
    void DeleteBuffer(std::uint32_t name);
    void BindBuffer(std::uint32_t target, std::uint32_t name);
    void BufferData(std::uint32_t target, std::size_t size, const void *data, std::uint32_t usage);
    void BufferSubData(std::uint32_t target, std::size_t offset, std::size_t size,
                       const void *data);
    void *MapBufferRange(std::uint32_t target, std::size_t offset, std::size_t length,
                         std::uint32_t access);
    void UnmapBuffer(std::uint32_t target);
    std::uint32_t GenVertexArray();
    void DeleteVertexArray(std::uint32_t name);
    void BindVertexArray(std::uint32_t name);
    void EnableVertexAttribArray(std::uint32_t index);
    void DisableVertexAttribArray(std::uint32_t index);
    void VertexAttribPointer(std::uint32_t index, int size, std::uint32_t type, bool normalized,
                             int stride, std::size_t offset);

    std::uint32_t GenFramebuffer();
    void DeleteFramebuffer(std::uint32_t name);
    void BindFramebuffer(std::uint32_t target, std::uint32_t name);
    void FramebufferTexture2D(std::uint32_t target, std::uint32_t attachment,
                              std::uint32_t textarget, std::uint32_t texture, int level);
    void DrawBuffers(int n, const std::uint32_t *buffers);
    std::uint32_t CheckFramebufferStatus(std::uint32_t target);
    std::uint32_t BoundFramebuffer() const { return draw_fb_; }

    std::uint32_t CreateProgram(std::unique_ptr<Program> program);
    void DeleteProgram(std::uint32_t name);
    void UseProgram(std::uint32_t name);
    int GetUniformLocation(std::uint32_t program, const std::string &name) const;
    const Program *GetProgram(std::uint32_t name) const;
    void Uniform1i(int loc, std::int32_t v);
    void Uniform1f(int loc, float v);
    void Uniform2f(int loc, float x, float y);
    void Uniform3f(int loc, float x, float y, float z);
    void Uniform3fv(int loc, int count, const float *v);
    void UniformMatrix4fv(int loc, int count, bool transpose, const float *v);
    void UniformBits(int loc, int components, const std::uint32_t *bits);

    void Enable(std::uint32_t cap);
    void Disable(std::uint32_t cap);
    void CullFace(std::uint32_t mode);
    void FrontFace(std::uint32_t mode);
    void DepthFunc(std::uint32_t func);
    void DepthMask(bool flag);
    void BlendFunc(std::uint32_t src, std::uint32_t dst);
    void Viewport(int x, int y, int width, int height);
    void ClearColor(float r, float g, float b, float a);
    void ClearDepth(double depth);
    void Clear(std::uint32_t mask);

    void DrawArrays(std::uint32_t mode, int first, int count);
    void DrawRangeElements(std::uint32_t mode, std::uint32_t start, std::uint32_t end, int count,
                           std::uint32_t type, std::size_t offset);

    void ReadPixels(int x, int y, int width, int height, std::uint32_t format, std::uint32_t type,
                    void *out);
    std::vector<std::uint32_t> ReadDefaultFramebufferWords();
    void Finish();
    void Parallel(int n, const std::function<void(int)> &fn);
    void SetDiscard(bool discard);

    struct Stats {
        std::uint64_t triangles_binned = 0;
        std::uint64_t blocks_shaded = 0;
        std::uint64_t draws = 0;
        std::uint64_t draws_skipped = 0;
    };
    const Stats &GetStats() const { return stats_; }

private:
    struct TextureObject;
    struct BufferObject;
    struct VertexArrayObject;
    struct FramebufferObject;
    struct ProgramObject;

    TextureObject *BoundTexture();
    BufferObject *BoundBuffer(std::uint32_t target);
    TargetInfo CurrentTarget();
    std::shared_ptr<const DrawState> SnapshotState();
    void Draw(DrawInput &input);
    void FlushScene();
    void WaitRaster();
    void Sync();
    void SubmitChunk();
    void DrainDraws();

    ContextOptions options_;
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<SceneList> scene_;
    std::vector<std::unique_ptr<SceneList>> rastering_;
    std::vector<std::unique_ptr<SceneList>> spare_;
    void FinishRaster(std::size_t count);

    bool discard_ = false;
    std::unique_ptr<DrawChunk> chunk_;
    std::vector<std::shared_ptr<void>> draw_jobs_;
    Stats stats_;

    std::unordered_map<std::uint32_t, std::unique_ptr<TextureObject>> textures_;
    std::unordered_map<std::uint32_t, std::unique_ptr<BufferObject>> buffers_;
    std::unordered_map<std::uint32_t, std::unique_ptr<VertexArrayObject>> vaos_;
    std::unordered_map<std::uint32_t, std::unique_ptr<FramebufferObject>> fbos_;
    std::unordered_map<std::uint32_t, std::unique_ptr<ProgramObject>> programs_;
    std::uint32_t next_texture_ = 1, next_buffer_ = 1, next_vao_ = 1, next_fbo_ = 1,
                  next_program_ = 1;

    std::vector<std::uint32_t> window_;

    std::uint32_t active_unit_ = 0;
    std::array<std::uint32_t, kMaxTextureUnits> unit_textures_{};
    std::uint32_t array_buffer_ = 0;
    std::uint32_t vao_ = 0;
    std::unique_ptr<VertexArrayObject> default_vao_;
    std::uint32_t draw_fb_ = 0;
    std::uint32_t program_ = 0;

    bool cull_ = false;
    std::uint32_t cull_mode_ = GL_BACK;
    std::uint32_t front_face_ = 0x0901;
    bool depth_test_ = false;
    std::uint32_t depth_func_ = GL_LESS;
    bool depth_mask_ = true;
    bool blend_ = false;
    std::uint32_t blend_src_ = GL_ONE, blend_dst_ = GL_ZERO;
    int viewport_[4] = {0, 0, 0, 0};
    float clear_color_[4] = {0, 0, 0, 0};
    double clear_depth_ = 1.0;

    std::uint32_t mapped_array_ = 0, mapped_element_ = 0;
};

}

#endif
