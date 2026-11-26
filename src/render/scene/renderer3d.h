// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GRF's OpenGLRenderer3D (systems/graphics/rendering/opengl_renderer3d.cpp, Apache-2.0) and its
// Texture and VertexBuffer resources: every GL call sequence of GRF's renderer, with its caches of
// the active texture unit, the current shader and the uniform locations.

#ifndef CUJU_RENDER_SCENE_RENDERER3D_H
#define CUJU_RENDER_SCENE_RENDERER3D_H

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../assets/assets.h"
#include "gl_stream.h"
#include "grf_math.h"

namespace cuju::render::scene {
class Renderer3D;

enum class InternalFormat : std::uint8_t { kRGB8, kSRGB8, kRGBA8, kSRGBA8, kRGBA16F, kDepth16 };
enum class PixelFormat : std::uint8_t { kRGB, kRGBA, kDepth };

// GRF's Texture resource: deletes its GL texture when it dies.
class Texture {
public:
    explicit Texture(Renderer3D *renderer = nullptr) : renderer_(renderer) {}
    ~Texture();
    Texture(const Texture &) = delete;
    Texture &operator=(const Texture &) = delete;
    void SetRenderer(Renderer3D *r) { renderer_ = r; }
    int CreateTexture(InternalFormat internal, PixelFormat pixel, int width, int height, bool alpha,
                      bool repeat, bool mipmaps, bool filter, bool compare_depth = false);
    void ResizeTexture(const assets::Surface &image, InternalFormat internal, PixelFormat pixel,
                       bool alpha, bool mipmaps);
    void UpdateTexture(const assets::Surface &image, bool alpha, bool mipmaps);
    void DeleteTexture();
    int id() const { return id_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    Renderer3D *renderer_;
    int id_ = -1;
    int width_ = 0, height_ = 0;
};

struct VertexBufferId {
    int buffer = -1;
    int vertex_array = 0;
    int element_array = 0;
};

// GRF's VertexBuffer resource.
class VertexBuffer {
public:
    VertexBuffer() = default;
    ~VertexBuffer();
    VertexBuffer(const VertexBuffer &) = delete;
    VertexBuffer &operator=(const VertexBuffer &) = delete;
    void SetTriangleMesh(const std::vector<float> &vertices, int vertices_data_size,
                         const std::vector<std::uint32_t> &indices);
    void TriangleMeshWasUpdatedExternally(int vertices_data_size,
                                          const std::vector<std::uint32_t> &indices);
    VertexBufferId CreateOrUpdateVertexBuffer(Renderer3D *renderer, bool dynamic);
    std::vector<float> &vertices() { return vertices_; }
    int vertices_data_size() const { return vertices_data_size_; }
    int vao() const { return id_.vertex_array; }
    int buffer() const { return id_.buffer; }

private:
    Renderer3D *renderer_ = nullptr;
    std::vector<float> vertices_;
    std::vector<std::uint32_t> indices_;
    int vertices_data_size_ = 0;
    int vertex_count_ = 0;
    bool size_changed_ = false;
    VertexBufferId id_;
};

template <typename T>
struct Resource {
    explicit Resource(std::string ident) : ident(std::move(ident)), res(std::make_unique<T>()) {}
    std::string ident;
    std::unique_ptr<T> res;
};
template <typename T>
using ResourcePtr = std::shared_ptr<Resource<T>>;

struct RendererMaterial {
    ResourcePtr<Texture> diffuse, normal, specular, illumination;
    float shininess = 0.0f;
    float specular_amount = 0.0f;
    Vector3 self_illumination;
};

struct VertexBufferIndex {
    int start = 0;
    int size = 0;
    RendererMaterial material;
};

struct VertexBufferQueueEntry {
    const std::vector<VertexBufferIndex> *indices = nullptr;
    ResourcePtr<VertexBuffer> buffer;
    Vector3 position;
    Quaternion rotation;
};

struct LightQueueEntry {
    bool has_shadow = false;
    int shadow_texture = 0;
    Matrix4 light_projection, light_view;
    Vector3 position;
    int type = 0;
    bool shadow = false;
    Vector3 color;
    float radius = 0.0f;
};

struct Overlay2DQueueEntry {
    int texture = 0;
    int position[2] = {0, 0};
    int size[2] = {0, 0};
};

enum class RenderMode : std::uint8_t { kGeometryOnly, kDiffuse, kFull };
enum class Cull : std::uint8_t { kOff, kFront, kBack };
enum class Attachment : std::uint8_t { kNone, kBack, kDepth, kColor0, kColor1, kColor2 };

struct View {
    int x = 0, y = 0, width = 0, height = 0;
    int gbuffer = 0, gbuffer_depth = 0, gbuffer_albedo = 0, gbuffer_normal = 0, gbuffer_aux = 0;
    int accum = 0, accum_accum = 0, accum_modifier = 0;
};

class Renderer3D {
public:
    Renderer3D(GlStream &gl, int width, int height);

    GlStream &gl() { return gl_; }
    int context_width() const { return width_; }
    const std::vector<float> &ssao_kernel() const { return ssao_kernel_; }
    int context_height() const { return height_; }

    void CreateContext(const assets::Surface &noise);

    void SwapBuffers(std::vector<std::uint8_t> *screen);
    void SetMatrix(const std::string &name, const Matrix4 &m);
    void RenderOverlay2D(const std::vector<Overlay2DQueueEntry> &queue);
    void RenderOverlay2D();
    void RenderLights(const std::deque<LightQueueEntry> &lights, const Matrix4 &projection,
                      const Matrix4 &view);

    int CreateView(float x_percent, float y_percent, float width_percent, float height_percent);
    View &GetView(int id) { return views_.at(id); }
    void DeleteView(int id);

    void SetCullingMode(Cull mode);
    void SetBlending(bool on);
    void SetBlendingFunction(std::uint32_t src, std::uint32_t dst);
    void SetDepthFunction(std::uint32_t func);
    void SetDepthTesting(bool on);
    void SetDepthMask(bool on);
    void ClearBuffer(bool depth, bool color);
    Matrix4 CreatePerspectiveMatrix(float aspect, float near_cap, float far_cap);
    static Matrix4 CreateOrthoMatrix(float left, float right, float bottom, float top,
                                     float near_cap, float far_cap);

    VertexBufferId CreateVertexBuffer(const float *vertices, int vertices_data_size,
                                      const std::vector<std::uint32_t> &indices, bool dynamic);
    void UpdateVertexBuffer(const VertexBufferId &id, const float *vertices,
                            int vertices_data_size);
    void DeleteVertexBuffer(const VertexBufferId &id);
    void RenderVertexBuffer(const std::deque<VertexBufferQueueEntry> &queue, RenderMode mode);

    int CreateTexture(InternalFormat internal, PixelFormat pixel, int width, int height,
                      bool repeat, bool mipmaps, bool filter, bool compare_depth);
    void ResizeTexture(int id, const assets::Surface &source, InternalFormat internal,
                       PixelFormat pixel, bool mipmaps);
    void UpdateTexture(int id, const assets::Surface &source, bool alpha, bool mipmaps);
    void DeleteTexture(int id);
    void BindTexture(int id);
    void SetTextureUnit(int unit);

    int CreateFrameBuffer();
    void DeleteFrameBuffer(int id);
    void BindFrameBuffer(int id);
    void SetFrameBufferTexture2D(Attachment a, int texture);
    bool CheckFrameBufferStatus();
    void SetRenderTargets(const std::vector<Attachment> &targets);

    void SetFOV(float fov) { fov_ = fov; }
    void SetViewport(int x, int y, int w, int h) { gl_.Viewport(x, y, w, h); }
    void UseShader(const std::string &name);
    void SetUniformInt(const std::string &shader, const std::string &var, int value);
    void SetUniformFloat(const std::string &shader, const std::string &var, float value);
    void SetUniformFloat2(const std::string &shader, const std::string &var, float a, float b);
    void SetUniformFloat3(const std::string &shader, const std::string &var, float a, float b,
                          float c);
    void SetUniformFloat3Array(const std::string &shader, const std::string &var, int count,
                               const float *values);
    void SetUniformMatrix4(const std::string &shader, const std::string &var, const Matrix4 &m);
    void SetFramebufferGammaCorrection(bool on);

private:
    struct Shader {
        std::uint32_t program = 0, vertex = 0, fragment = 0;
    };
    void LoadShader(const std::string &name, const std::string &file);
    int UniformLocation(const std::string &shader, const std::string &var);
    void SetMaxAnisotropy();
    void DrawBufferChunk(int start, int count);
    VertexBufferId CreateSimpleVertexBuffer(const float *vertices, int bytes);

    GlStream &gl_;
    int width_, height_;
    std::vector<float> ssao_kernel_;
    float fov_ = 45;
    float anisotropy_ = 2;
    int active_unit_ = -1;
    std::map<std::string, Shader> shaders_;
    std::string current_shader_;
    std::map<std::string, int> uniform_cache_;
    std::vector<View> views_;
    int noise_texture_ = 0;
    VertexBufferId overlay_buffer_, quad_buffer_;
};

}  // namespace cuju::render::scene

#endif
