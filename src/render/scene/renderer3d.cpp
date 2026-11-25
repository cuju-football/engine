// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "renderer3d.h"

#include <cmath>

#include "random.h"

namespace cuju::render::scene {
namespace {
using namespace gl;

constexpr std::uint32_t kStaticDraw = 0x88E4;
constexpr std::uint32_t kDynamicDraw = 0x88E8;
constexpr std::uint32_t kMapWrite = 0x0002, kMapUnsynchronized = 0x0020;
constexpr std::uint32_t kRepeat = 0x2901, kClampToEdge = 0x812F;
constexpr std::uint32_t kNearest = 0x2600, kLinear = 0x2601;
constexpr std::uint32_t kNearestMipmapNearest = 0x2700, kLinearMipmapLinear = 0x2703;
constexpr std::uint32_t kCompareRefToTexture = 0x884E;
constexpr std::uint32_t kMaxAnisotropyQuery = 0x84FF;
constexpr std::uint32_t kFramebufferSrgb = 0x8DB9;
constexpr std::uint32_t kBackLeft = 0x0402;
constexpr std::uint32_t kInfoLogLength = 0x8B84;
constexpr std::uint32_t kCompileStatus = 0x8B81;

std::uint32_t InternalEnum(InternalFormat f) {
    switch (f) {
        case InternalFormat::kRGB8:
            return GL_RGB8;
        case InternalFormat::kSRGB8:
            return GL_SRGB8;
        case InternalFormat::kRGBA8:
            return GL_RGBA8;
        case InternalFormat::kSRGBA8:
            return GL_SRGB8_ALPHA8;
        case InternalFormat::kRGBA16F:
            return GL_RGBA16F;
        case InternalFormat::kDepth16:
            return GL_DEPTH_COMPONENT16;
    }
    return GL_RGB8;
}

std::uint32_t PixelEnum(PixelFormat f) {
    switch (f) {
        case PixelFormat::kRGB:
            return GL_RGB;
        case PixelFormat::kRGBA:
            return GL_RGBA;
        case PixelFormat::kDepth:
            return GL_DEPTH_COMPONENT;
    }
    return GL_RGB;
}

std::uint32_t AttachmentEnum(Attachment a) {
    switch (a) {
        case Attachment::kNone:
            return GL_NONE;
        case Attachment::kBack:
            return kBackLeft;
        case Attachment::kDepth:
            return GL_DEPTH_ATTACHMENT;
        case Attachment::kColor0:
            return GL_COLOR_ATTACHMENT0;
        case Attachment::kColor1:
            return GL_COLOR_ATTACHMENT0 + 1;
        case Attachment::kColor2:
            return GL_COLOR_ATTACHMENT0 + 2;
    }
    return GL_NONE;
}

void GeneratePoissonKernel(GrfRng &rng, float *kernel, unsigned kernel_size) {
    const unsigned candidate_size = 32;
    std::vector<Vector3> samples(kernel_size), candidates(candidate_size);
    for (unsigned i = 0; i < kernel_size; i++) {
        for (unsigned c = 0; c < candidate_size; c++) {
            float s = rng.Range(0.0f, 2.0f * kPi);
            float t = rng.Range(-1.0f, 1.0f);
            float m = std::sqrt(1.0f - t * t);
            candidates[c] = Vector3(SinFloat(s) * m, CosFloat(s) * m, std::fabs(t));
        }
        int furthest = 0;
        float furthest_distance = 0.0f;
        for (unsigned c = 0; c < candidate_size; c++) {
            float closest = 1.0f;
            for (unsigned x = 0; x < i; x++) {
                float distance = GetDistance(candidates[c], samples[x]);
                if (distance < closest) closest = distance;
            }
            if (closest > furthest_distance) {
                furthest_distance = closest;
                furthest = static_cast<int>(c);
            }
        }
        samples[i] = candidates[furthest];
    }
    for (unsigned i = 0; i < kernel_size; i++) {
        samples[i] = samples[i].GetNormalized(Vector3(0)) * rng.Range(0.14f, 1.0f);
        samples[i] = samples[i] * 0.75f + Vector3(0, 0, 1) * 0.25f;
        kernel[i * 3 + 0] = samples[i].coordinates[0];
        kernel[i * 3 + 1] = samples[i].coordinates[1];
        kernel[i * 3 + 2] = samples[i].coordinates[2];
    }
}

ProgramKind KindOf(const std::string &name) {
    if (name == "simple") return ProgramKind::kSimple;
    if (name == "lighting") return ProgramKind::kLighting;
    if (name == "ambient") return ProgramKind::kAmbient;
    if (name == "zphase") return ProgramKind::kZphase;
    if (name == "postprocess") return ProgramKind::kPostprocess;
    return ProgramKind::kOverlay;
}

}

Texture::~Texture() { DeleteTexture(); }

void Texture::DeleteTexture() {
    if (id_ != -1) {
        renderer_->DeleteTexture(id_);
        id_ = -1;
    }
}

int Texture::CreateTexture(InternalFormat internal, PixelFormat pixel, int width, int height, bool,
                           bool repeat, bool mipmaps, bool filter, bool compare_depth) {
    id_ = renderer_->CreateTexture(internal, pixel, width, height, repeat, mipmaps, filter,
                                   compare_depth);
    width_ = width;
    height_ = height;
    return id_;
}

void Texture::ResizeTexture(const assets::Surface &image, InternalFormat internal,
                            PixelFormat pixel, bool, bool mipmaps) {
    renderer_->ResizeTexture(id_, image, internal, pixel, mipmaps);
}

void Texture::UpdateTexture(const assets::Surface &image, bool, bool mipmaps) {
    renderer_->UpdateTexture(id_, image, image.alpha, mipmaps);
}

VertexBuffer::~VertexBuffer() {
    if (id_.buffer != -1) renderer_->DeleteVertexBuffer(id_);
}

void VertexBuffer::SetTriangleMesh(const std::vector<float> &vertices, int vertices_data_size,
                                   const std::vector<std::uint32_t> &indices) {
    vertices_ = vertices;
    TriangleMeshWasUpdatedExternally(vertices_data_size, indices);
}

void VertexBuffer::TriangleMeshWasUpdatedExternally(int vertices_data_size,
                                                    const std::vector<std::uint32_t> &indices) {
    if (!indices.empty()) {
        if (indices.size() != indices_.size()) size_changed_ = true;
        indices_ = indices;
    }
    vertices_data_size_ = vertices_data_size;
    int count = vertices_data_size / 5 / 3;
    if (count != vertex_count_) size_changed_ = true;
    vertex_count_ = count;
}

VertexBufferId VertexBuffer::CreateOrUpdateVertexBuffer(Renderer3D *renderer, bool dynamic) {
    renderer_ = renderer;
    if (id_.buffer != -1 && size_changed_) {
        renderer->DeleteVertexBuffer(id_);
        id_.buffer = -1;
    }
    if (id_.buffer == -1) {
        id_ =
            renderer->CreateVertexBuffer(vertices_.data(), vertices_data_size_, indices_, dynamic);
    } else {
        renderer->UpdateVertexBuffer(id_, vertices_.data(), vertices_data_size_);
    }
    size_changed_ = false;
    return id_;
}

Renderer3D::Renderer3D(GlStream &gl, int width, int height)
    : gl_(gl), width_(width), height_(height) {}

void Renderer3D::CreateContext(const assets::Surface &noise) {
    anisotropy_ = 2;
    gl_.GetFloatv(kMaxAnisotropyQuery, &anisotropy_);
    gl_.Disable(GL_MULTISAMPLE);
    gl_.CullFace(GL_BACK);
    LoadShader("simple", "media/shaders/simple");
    LoadShader("lighting", "media/shaders/lighting");
    LoadShader("ambient", "media/shaders/ambient");
    LoadShader("zphase", "media/shaders/zphase");
    LoadShader("postprocess", "media/shaders/postprocess");
    LoadShader("overlay", "media/shaders/overlay");
    current_shader_ = shaders_.begin()->first;
    noise_texture_ = CreateTexture(InternalFormat::kRGB8, PixelFormat::kRGB, noise.w, noise.h, true,
                                   false, false, false);
    UpdateTexture(noise_texture_, noise, false, false);
    static const float overlay[] = {0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1,
                                    0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1};
    overlay_buffer_ = CreateSimpleVertexBuffer(overlay, sizeof(overlay));
    static const float quad[] = {-1, 1, 0, 1, 1, -1, 0, 1, -1, -1, 0, 1,
                                 -1, 1, 0, 1, 1, 1,  0, 1, 1,  -1, 0, 1};
    quad_buffer_ = CreateSimpleVertexBuffer(quad, sizeof(quad));
    gl_.ClearColor(0, 0, 0, 0);
    gl_.ClearDepth(1.0);
    gl_.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    gl_.Disable(GL_MULTISAMPLE);
}

VertexBufferId Renderer3D::CreateSimpleVertexBuffer(const float *vertices, int bytes) {
    std::uint32_t vao = gl_.GenVertexArray();
    std::uint32_t vbo = gl_.GenBuffer();
    gl_.BindVertexArray(vao);
    gl_.BindBuffer(GL_ARRAY_BUFFER, vbo);
    gl_.BufferData(GL_ARRAY_BUFFER, bytes, vertices, kStaticDraw);
    gl_.VertexAttribPointer(0, 4, GL_FLOAT, false, 16, 0);
    gl_.EnableVertexAttribArray(0);
    gl_.BindBuffer(GL_ARRAY_BUFFER, 0);
    gl_.BindVertexArray(0);
    VertexBufferId id;
    id.buffer = static_cast<int>(vbo);
    id.vertex_array = static_cast<int>(vao);
    return id;
}

void Renderer3D::LoadShader(const std::string &name, const std::string &file) {
    Shader shader;
    shader.vertex = gl_.CreateShader(0x8B31);
    gl_.ShaderSource(shader.vertex, file + ".vert");
    gl_.CompileShader(shader.vertex);
    gl_.GetShaderiv(shader.vertex, kCompileStatus);
    gl_.GetShaderiv(shader.vertex, kInfoLogLength);
    gl_.GetShaderiv(shader.vertex, kCompileStatus);
    gl_.IsShader(shader.vertex);
    shader.fragment = gl_.CreateShader(0x8B30);
    gl_.ShaderSource(shader.fragment, file + ".frag");
    gl_.CompileShader(shader.fragment);
    gl_.GetShaderiv(shader.fragment, kCompileStatus);
    gl_.GetShaderiv(shader.fragment, kInfoLogLength);
    gl_.GetShaderiv(shader.fragment, kCompileStatus);
    gl_.IsShader(shader.fragment);
    shader.program = gl_.CreateProgram(KindOf(name));
    gl_.AttachShader(shader.program, shader.vertex);
    gl_.AttachShader(shader.program, shader.fragment);
    shaders_[name] = shader;
    std::uint32_t p = shader.program;
    if (name == "zphase") {
        gl_.BindAttribLocation(p, 0, "position");
        gl_.BindFragDataLocation(p, 0, "stdout");
    }
    if (name == "simple") {
        gl_.BindAttribLocation(p, 0, "position");
        gl_.BindAttribLocation(p, 1, "normal");
        gl_.BindAttribLocation(p, 2, "texcoord");
        gl_.BindAttribLocation(p, 3, "tangent");
        gl_.BindAttribLocation(p, 4, "bitangent");
        gl_.BindFragDataLocation(p, 0, "stdout0");
        gl_.BindFragDataLocation(p, 1, "stdout1");
        gl_.BindFragDataLocation(p, 2, "stdout2");
    }
    if (name == "ambient") {
        gl_.BindAttribLocation(p, 0, "position");
        gl_.BindFragDataLocation(p, 0, "stdout");
    }
    if (name == "lighting") {
        gl_.BindFragDataLocation(p, 0, "stdout0");
        gl_.BindFragDataLocation(p, 1, "stdout1");
    }
    if (name == "postprocess") {
        gl_.BindAttribLocation(p, 0, "position");
        gl_.BindFragDataLocation(p, 0, "stdout");
    }
    if (name == "overlay") {
        gl_.BindAttribLocation(p, 0, "vertex");
        gl_.BindFragDataLocation(p, 0, "stdout");
    }
    gl_.LinkProgram(p);
    gl_.UseProgram(p);
    if (name == "simple") {
        SetUniformInt("simple", "map_albedo", 0);
        SetUniformInt("simple", "map_normal", 1);
        SetUniformInt("simple", "map_specular", 2);
        SetUniformInt("simple", "map_illumination", 3);
    }
    if (name == "ambient") {
        SetUniformInt("ambient", "map_albedo", 0);
        SetUniformInt("ambient", "map_normal", 1);
        SetUniformInt("ambient", "map_depth", 2);
        SetUniformInt("ambient", "map_aux", 3);
        SetUniformInt("ambient", "map_noise", 4);
        SetUniformFloat("ambient", "contextX", 0.0f);
        SetUniformFloat("ambient", "contextY", 0.0f);
        SetUniformFloat("ambient", "contextWidth", static_cast<float>(width_));
        SetUniformFloat("ambient", "contextHeight", static_cast<float>(height_));
        float kernel[32 * 3];
        GrfRng rng(0);
        GeneratePoissonKernel(rng, kernel, 32);
        ssao_kernel_.assign(kernel, kernel + 32 * 3);
        SetUniformFloat3Array("ambient", "SSAO_kernel", 32, kernel);
    }
    if (name == "lighting") {
        SetUniformInt("lighting", "map_albedo", 0);
        SetUniformInt("lighting", "map_normal", 1);
        SetUniformInt("lighting", "map_depth", 2);
        SetUniformInt("lighting", "map_shadow", 7);
        SetUniformFloat("lighting", "contextX", 0.0f);
        SetUniformFloat("lighting", "contextY", 0.0f);
        SetUniformFloat("lighting", "contextWidth", static_cast<float>(width_));
        SetUniformFloat("lighting", "contextHeight", static_cast<float>(height_));
    }
    if (name == "postprocess") {
        SetUniformInt("postprocess", "map_accumulation", 0);
        SetUniformInt("postprocess", "map_modifier", 1);
        SetUniformInt("postprocess", "map_depth", 2);
        SetUniformFloat("postprocess", "contextX", 0.0f);
        SetUniformFloat("postprocess", "contextY", 0.0f);
        SetUniformFloat("postprocess", "contextWidth", static_cast<float>(width_));
        SetUniformFloat("postprocess", "contextHeight", static_cast<float>(height_));
    }
    SetTextureUnit(0);
    UseShader("");
}

void Renderer3D::SwapBuffers(std::vector<std::uint8_t> *screen) {
    screen->resize(static_cast<std::size_t>(width_) * height_ * 3);
    gl_.ReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, screen->data());
}

void Renderer3D::SetMatrix(const std::string &name, const Matrix4 &m) {
    SetUniformMatrix4(current_shader_, name, m);
}

void Renderer3D::RenderOverlay2D(const std::vector<Overlay2DQueueEntry> &queue) {
    if (queue.empty()) return;
    gl_.BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gl_.Enable(GL_BLEND);
    gl_.Disable(GL_DEPTH_TEST);
    SetDepthMask(false);
    gl_.Disable(GL_CULL_FACE);
    UseShader("overlay");
    SetMatrix("projection", CreateOrthoMatrix(0, static_cast<float>(width_),
                                              static_cast<float>(height_), 0, 0.1f, 10));
    gl_.BindVertexArray(overlay_buffer_.vertex_array);
    SetTextureUnit(0);
    for (const Overlay2DQueueEntry &q : queue) {
        Matrix4 model = Matrix4::Identity();
        model.SetTranslation(
            Vector3(static_cast<float>(q.position[0]), static_cast<float>(q.position[1]), -1.0f));
        model.SetScale(Vector3(static_cast<float>(q.size[0]), static_cast<float>(q.size[1]), 1.0f));
        SetMatrix("model", model);
        BindTexture(q.texture);
        gl_.DrawArrays(GL_TRIANGLES, 0, 6);
    }
    gl_.BindVertexArray(0);
    gl_.Enable(GL_DEPTH_TEST);
    gl_.Enable(GL_CULL_FACE);
    SetDepthMask(true);
    gl_.Disable(GL_BLEND);
}

void Renderer3D::RenderOverlay2D() {
    SetMatrix("orthoProjectionMatrix", CreateOrthoMatrix(-1, 1, -1, 1, 0.0f, 1.0f));
    Matrix4 view = Matrix4::Identity();
    view.SetTranslation(Vector3(0, 0, -0.5f));
    SetMatrix("orthoViewMatrix", view);
    SetCullingMode(Cull::kOff);
    SetTextureUnit(4);
    BindTexture(noise_texture_);
    SetTextureUnit(0);
    gl_.BindVertexArray(quad_buffer_.vertex_array);
    gl_.DrawArrays(GL_TRIANGLES, 0, 6);
    gl_.BindVertexArray(0);
    SetTextureUnit(4);
    BindTexture(0);
    SetCullingMode(Cull::kBack);
}

void Renderer3D::RenderLights(const std::deque<LightQueueEntry> &lights, const Matrix4 &projection,
                              const Matrix4 &view) {
    (void)projection;
    Vector3 camera = view.GetInverse().GetTranslation();
    SetUniformFloat3(current_shader_, "cameraPosition", camera.coordinates[0],
                     camera.coordinates[1], camera.coordinates[2]);
    for (const LightQueueEntry &light : lights) {
        SetUniformInt(current_shader_, "has_shadow", light.has_shadow ? 1 : 0);
        if (light.has_shadow) {
            SetTextureUnit(7);
            gl_.BindTexture(GL_TEXTURE_2D, light.shadow_texture);
            Matrix4 to_tex = Matrix4::Identity();
            to_tex.SetTranslation(Vector3(0.5f, 0.5f, 0.5f));
            to_tex.SetScale(Vector3(0.5f, 0.5f, 0.5f));
            SetUniformMatrix4(current_shader_, "lightViewProjectionMatrix",
                              to_tex * light.light_projection * light.light_view);
        }
        SetUniformFloat3(current_shader_, "lightColor", light.color.coordinates[0],
                         light.color.coordinates[1], light.color.coordinates[2]);
        SetUniformFloat(current_shader_, "lightRadius", light.radius);
        SetUniformFloat3(current_shader_, "lightPosition", light.position.coordinates[0],
                         light.position.coordinates[1], light.position.coordinates[2]);
        // GRF's lights are directional, type 0: a fullscreen quad.
        SetCullingMode(Cull::kOff);
        SetDepthFunction(GL_ALWAYS);
        SetMatrix("projectionMatrix", CreateOrthoMatrix(-1, 1, -1, 1, 0.0f, 1.0f));
        Matrix4 xview = Matrix4::Identity();
        xview.SetTranslation(Vector3(0, 0, -0.5f));
        SetMatrix("viewMatrix", xview);
        SetMatrix("modelMatrix", Matrix4::Identity());
        gl_.BindVertexArray(quad_buffer_.vertex_array);
        gl_.DrawArrays(GL_TRIANGLES, 0, 6);
        gl_.BindVertexArray(0);
        if (light.has_shadow) {
            gl_.BindTexture(GL_TEXTURE_2D, 0);
            SetTextureUnit(0);
        }
    }
}

int Renderer3D::CreateView(float x_percent, float y_percent, float width_percent,
                           float height_percent) {
    View view;
    view.x = static_cast<int>(std::floor(x_percent * 0.01 * width_));
    view.y = static_cast<int>(std::floor(y_percent * 0.01 * height_));
    view.width = static_cast<int>(std::floor(width_percent * 0.01 * width_));
    view.height = static_cast<int>(std::floor(height_percent * 0.01 * height_));
    view.gbuffer = CreateFrameBuffer();
    BindFrameBuffer(view.gbuffer);
    view.gbuffer_depth = CreateTexture(InternalFormat::kDepth16, PixelFormat::kDepth, view.width,
                                       view.height, false, false, false, false);
    SetFrameBufferTexture2D(Attachment::kDepth, view.gbuffer_depth);
    view.gbuffer_albedo = CreateTexture(InternalFormat::kRGBA16F, PixelFormat::kRGBA, view.width,
                                        view.height, false, false, false, false);
    SetFrameBufferTexture2D(Attachment::kColor0, view.gbuffer_albedo);
    view.gbuffer_normal = CreateTexture(InternalFormat::kRGBA16F, PixelFormat::kRGBA, view.width,
                                        view.height, false, false, false, false);
    SetFrameBufferTexture2D(Attachment::kColor1, view.gbuffer_normal);
    view.gbuffer_aux = CreateTexture(InternalFormat::kRGBA16F, PixelFormat::kRGBA, view.width,
                                     view.height, false, false, false, false);
    SetFrameBufferTexture2D(Attachment::kColor2, view.gbuffer_aux);
    SetRenderTargets({Attachment::kColor0, Attachment::kColor1, Attachment::kColor2});
    CheckFrameBufferStatus();
    BindFrameBuffer(0);
    SetRenderTargets({Attachment::kBack});
    view.accum = CreateFrameBuffer();
    BindFrameBuffer(view.accum);
    view.accum_accum = CreateTexture(InternalFormat::kRGBA16F, PixelFormat::kRGBA, view.width,
                                     view.height, false, false, false, false);
    SetFrameBufferTexture2D(Attachment::kColor0, view.accum_accum);
    view.accum_modifier = CreateTexture(InternalFormat::kRGBA16F, PixelFormat::kRGBA, view.width,
                                        view.height, false, false, false, false);
    SetFrameBufferTexture2D(Attachment::kColor1, view.accum_modifier);
    SetRenderTargets({Attachment::kColor0, Attachment::kColor1});
    CheckFrameBufferStatus();
    BindFrameBuffer(0);
    SetRenderTargets({Attachment::kBack});
    views_.push_back(view);
    return static_cast<int>(views_.size()) - 1;
}

void Renderer3D::DeleteView(int id) {
    View &view = views_.at(id);
    BindFrameBuffer(view.gbuffer);
    SetFrameBufferTexture2D(Attachment::kDepth, 0);
    SetFrameBufferTexture2D(Attachment::kColor0, 0);
    SetFrameBufferTexture2D(Attachment::kColor1, 0);
    SetFrameBufferTexture2D(Attachment::kColor2, 0);
    DeleteFrameBuffer(view.gbuffer);
    BindFrameBuffer(view.accum);
    SetFrameBufferTexture2D(Attachment::kColor0, 0);
    SetFrameBufferTexture2D(Attachment::kColor1, 0);
    DeleteFrameBuffer(view.accum);
    DeleteTexture(view.gbuffer_albedo);
    DeleteTexture(view.gbuffer_normal);
    DeleteTexture(view.gbuffer_depth);
    DeleteTexture(view.gbuffer_aux);
    DeleteTexture(view.accum_accum);
    DeleteTexture(view.accum_modifier);
}

void Renderer3D::SetCullingMode(Cull mode) {
    switch (mode) {
        case Cull::kOff:
            gl_.Disable(GL_CULL_FACE);
            break;
        case Cull::kFront:
            gl_.Enable(GL_CULL_FACE);
            gl_.CullFace(GL_FRONT);
            break;
        case Cull::kBack:
            gl_.Enable(GL_CULL_FACE);
            gl_.CullFace(GL_BACK);
            break;
    }
}

void Renderer3D::SetBlending(bool on) {
    if (on)
        gl_.Enable(GL_BLEND);
    else
        gl_.Disable(GL_BLEND);
}

void Renderer3D::SetBlendingFunction(std::uint32_t src, std::uint32_t dst) {
    gl_.BlendFunc(src, dst);
}

void Renderer3D::SetDepthFunction(std::uint32_t func) { gl_.DepthFunc(func); }

void Renderer3D::SetDepthTesting(bool on) {
    if (on)
        gl_.Enable(GL_DEPTH_TEST);
    else
        gl_.Disable(GL_DEPTH_TEST);
}

void Renderer3D::SetDepthMask(bool on) { gl_.DepthMask(on); }

void Renderer3D::ClearBuffer(bool depth, bool color) {
    std::uint32_t mask = 0;
    if (depth) mask |= GL_DEPTH_BUFFER_BIT;
    if (color) mask |= GL_COLOR_BUFFER_BIT;
    gl_.Clear(mask);
}

Matrix4 Renderer3D::CreatePerspectiveMatrix(float aspect, float near_cap, float far_cap) {
    float xnear = near_cap;
    if (xnear == -1) xnear = 30.0f;
    float xfar = far_cap;
    if (xfar == -1) xfar = 270.0f;
    Matrix4 m;
    m.ConstructProjection(fov_, aspect, xnear, xfar);
    return m;
}

Matrix4 Renderer3D::CreateOrthoMatrix(float left, float right, float bottom, float top,
                                      float near_cap, float far_cap) {
    float xnear = near_cap;
    if (xnear == -1) xnear = 30.0f;
    float xfar = far_cap;
    if (xfar == -1) xfar = 270.0f;
    Matrix4 m;
    m.ConstructOrtho(left, right, bottom, top, xnear, xfar);
    return m;
}

VertexBufferId Renderer3D::CreateVertexBuffer(const float *vertices, int vertices_data_size,
                                              const std::vector<std::uint32_t> &c_indices,
                                              bool dynamic) {
    std::vector<std::uint32_t> indices = c_indices;
    std::uint32_t usage = dynamic ? kDynamicDraw : kStaticDraw;
    std::uint32_t vao = gl_.GenVertexArray();
    gl_.BindVertexArray(vao);
    std::uint32_t vbo = gl_.GenBuffer();
    std::size_t bytes = static_cast<std::size_t>(vertices_data_size) * 4;
    gl_.BindBuffer(GL_ARRAY_BUFFER, vbo);
    gl_.BufferData(GL_ARRAY_BUFFER, bytes, nullptr, usage);
    gl_.BufferSubData(GL_ARRAY_BUFFER, 0, bytes, vertices);
    if (indices.empty()) {
        for (int i = 0; i < vertices_data_size / 5 / 3; i++) indices.push_back(i);
    }
    std::uint32_t ibo = gl_.GenBuffer();
    gl_.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    gl_.BufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * 4, nullptr, usage);
    gl_.BufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, indices.size() * 4, indices.data());
    for (int i = 0; i < 5; i++) {
        gl_.VertexAttribPointer(i, 3, GL_FLOAT, false, 0,
                                static_cast<std::size_t>(vertices_data_size / 5 * i) * 4);
        gl_.EnableVertexAttribArray(i);
    }
    gl_.BindVertexArray(0);
    gl_.BindBuffer(GL_ARRAY_BUFFER, 0);
    gl_.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    VertexBufferId id;
    id.buffer = static_cast<int>(vbo);
    id.vertex_array = static_cast<int>(vao);
    id.element_array = static_cast<int>(ibo);
    return id;
}

void Renderer3D::UpdateVertexBuffer(const VertexBufferId &id, const float *vertices,
                                    int vertices_data_size) {
    std::size_t bytes = static_cast<std::size_t>(vertices_data_size) * 4;
    gl_.BindVertexArray(id.vertex_array);
    gl_.BindBuffer(GL_ARRAY_BUFFER, id.buffer);
    gl_.BufferData(GL_ARRAY_BUFFER, bytes, nullptr, kDynamicDraw);
    gl_.MapCopyUnmap(GL_ARRAY_BUFFER, 0, bytes, kMapWrite | kMapUnsynchronized, vertices);
    gl_.BindVertexArray(0);
    gl_.BindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer3D::DeleteVertexBuffer(const VertexBufferId &id) {
    gl_.DeleteBuffer(id.buffer);
    gl_.DeleteBuffer(id.element_array);
    gl_.DeleteVertexArray(id.vertex_array);
}

void Renderer3D::DrawBufferChunk(int start, int count) {
    if (count > 0)
        gl_.DrawRangeElements(GL_TRIANGLES, start, start + count, count, GL_UNSIGNED_INT,
                              static_cast<std::size_t>(start) * 4);
}

void Renderer3D::RenderVertexBuffer(const std::deque<VertexBufferQueueEntry> &queue,
                                    RenderMode mode) {
    int cur_diffuse = -1, cur_normal = -1, cur_specular = -1, cur_illumination = -1;
    int cur_bound = -1;
    Matrix4 transform;
    for (const VertexBufferQueueEntry &entry : queue) {
        VertexBuffer *vb = entry.buffer->res.get();
        if (cur_bound != vb->vao()) {
            cur_bound = vb->vao();
            gl_.BindVertexArray(cur_bound);
        }
        transform.Construct(entry.position, Vector3(1, 1, 1), entry.rotation);
        SetMatrix("modelMatrix", transform);
        bool sequential = true;
        int chunk_start = 0, chunk_count = 0;
        auto it = entry.indices->begin();
        while (it != entry.indices->end()) {
            const VertexBufferIndex &vbi = *it;
            if (mode != RenderMode::kGeometryOnly) {
                int diffuse = 0, normal = 0, specular = 0, illumination = 0;
                if (vbi.material.diffuse) diffuse = vbi.material.diffuse->res->id();
                if (mode == RenderMode::kFull) {
                    if (vbi.material.normal) normal = vbi.material.normal->res->id();
                    if (vbi.material.specular) specular = vbi.material.specular->res->id();
                    if (vbi.material.illumination)
                        illumination = vbi.material.illumination->res->id();
                }
                if (diffuse != cur_diffuse || normal != cur_normal || specular != cur_specular ||
                    illumination != cur_illumination) {
                    if (sequential) {
                        DrawBufferChunk(chunk_start, chunk_count);
                        sequential = false;
                    }
                    bool has_normal = normal != 0, has_specular = specular != 0,
                         has_illumination = illumination != 0;
                    if (mode == RenderMode::kFull) {
                        if (has_normal && normal != cur_normal) {
                            SetTextureUnit(1);
                            gl_.BindTexture(GL_TEXTURE_2D, normal);
                        }
                        if (has_specular && specular != cur_specular) {
                            SetTextureUnit(2);
                            gl_.BindTexture(GL_TEXTURE_2D, specular);
                        }
                        if (has_illumination && illumination != cur_illumination) {
                            SetTextureUnit(3);
                            gl_.BindTexture(GL_TEXTURE_2D, illumination);
                        }
                    }
                    SetTextureUnit(0);
                    gl_.BindTexture(GL_TEXTURE_2D, diffuse);
                    if (mode == RenderMode::kFull) {
                        SetUniformFloat3("simple", "materialparams", vbi.material.shininess,
                                         vbi.material.specular_amount,
                                         vbi.material.self_illumination.coordinates[0]);
                        SetUniformFloat3("simple", "materialbools",
                                         static_cast<float>(has_normal ? 1 : 0),
                                         static_cast<float>(has_specular ? 1 : 0),
                                         static_cast<float>(has_illumination ? 1 : 0));
                    }
                    cur_diffuse = diffuse;
                    cur_normal = normal;
                    cur_specular = specular;
                    cur_illumination = illumination;
                }
            }
            int start = vbi.start, count = vbi.size;
            if (sequential && chunk_start + chunk_count != start) {
                DrawBufferChunk(chunk_start, chunk_count);
                sequential = false;
            }
            if (sequential) {
                chunk_count += count;
            } else {
                chunk_start = start;
                chunk_count = count;
                sequential = true;
            }
            ++it;
            if (sequential && it == entry.indices->end()) {
                DrawBufferChunk(chunk_start, chunk_count);
                sequential = false;
            }
        }
    }
    if (mode != RenderMode::kGeometryOnly) {
        if (mode == RenderMode::kFull) {
            SetTextureUnit(1);
            gl_.BindTexture(GL_TEXTURE_2D, 0);
            SetTextureUnit(2);
            gl_.BindTexture(GL_TEXTURE_2D, 0);
            SetTextureUnit(3);
            gl_.BindTexture(GL_TEXTURE_2D, 0);
        }
        SetTextureUnit(0);
        gl_.BindTexture(GL_TEXTURE_2D, 0);
    }
    gl_.BindVertexArray(0);
}

int Renderer3D::CreateTexture(InternalFormat internal, PixelFormat pixel, int width, int height,
                              bool repeat, bool mipmaps, bool filter, bool compare_depth) {
    std::uint32_t id = gl_.GenTexture();
    BindTexture(static_cast<int>(id));
    std::int32_t fmin, fmag;
    if (filter) {
        fmin = mipmaps ? kLinearMipmapLinear : kLinear;
        fmag = kLinear;
    } else {
        fmin = mipmaps ? kNearestMipmapNearest : kNearest;
        fmag = kNearest;
    }
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat ? kRepeat : kClampToEdge);
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat ? kRepeat : kClampToEdge);
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fmin);
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fmag);
    if (internal == InternalFormat::kDepth16 && compare_depth) {
        gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, kCompareRefToTexture);
        gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    }
    if (filter) SetMaxAnisotropy();
    gl_.TexImage2D(GL_TEXTURE_2D, 0, InternalEnum(internal), width, height, PixelEnum(pixel),
                   GL_UNSIGNED_BYTE, nullptr);
    BindTexture(0);
    return static_cast<int>(id);
}

void Renderer3D::ResizeTexture(int id, const assets::Surface &source, InternalFormat internal,
                               PixelFormat pixel, bool mipmaps) {
    BindTexture(id);
    gl_.TexImage2D(GL_TEXTURE_2D, 0, InternalEnum(internal), source.w, source.h, PixelEnum(pixel),
                   GL_UNSIGNED_BYTE, source.pixels.data());
    if (mipmaps) gl_.GenerateMipmap(GL_TEXTURE_2D);
    bool filter = mipmaps;
    std::int32_t fmin, fmag;
    if (filter) {
        fmin = mipmaps ? kLinearMipmapLinear : kLinear;
        fmag = kLinear;
        SetMaxAnisotropy();
    } else {
        fmin = mipmaps ? kNearestMipmapNearest : kNearest;
        fmag = kNearest;
    }
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, kClampToEdge);
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, kClampToEdge);
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fmin);
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fmag);
    if (internal == InternalFormat::kDepth16)
        gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, 0);
    BindTexture(0);
}

void Renderer3D::UpdateTexture(int id, const assets::Surface &source, bool alpha, bool mipmaps) {
    gl_.BindTexture(GL_TEXTURE_2D, id);
    gl_.TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, source.w, source.h, alpha ? GL_RGBA : GL_RGB,
                      GL_UNSIGNED_BYTE, source.pixels.data());
    if (mipmaps) gl_.GenerateMipmap(GL_TEXTURE_2D);
    SetMaxAnisotropy();
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                      mipmaps ? kLinearMipmapLinear : kLinear);
    gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, kLinear);
    BindTexture(0);
}

void Renderer3D::DeleteTexture(int id) { gl_.DeleteTexture(static_cast<std::uint32_t>(id)); }

void Renderer3D::BindTexture(int id) {
    gl_.BindTexture(GL_TEXTURE_2D, static_cast<std::uint32_t>(id));
}

void Renderer3D::SetTextureUnit(int unit) {
    if (unit != active_unit_) {
        gl_.ActiveTexture(GL_TEXTURE0 + unit);
        active_unit_ = unit;
    }
}

void Renderer3D::SetMaxAnisotropy() {
    gl_.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, anisotropy_);
}

int Renderer3D::CreateFrameBuffer() { return static_cast<int>(gl_.GenFramebuffer()); }

void Renderer3D::DeleteFrameBuffer(int id) {
    gl_.DeleteFramebuffer(static_cast<std::uint32_t>(id));
}

void Renderer3D::BindFrameBuffer(int id) { gl_.BindFramebuffer(GL_FRAMEBUFFER, id); }

void Renderer3D::SetFrameBufferTexture2D(Attachment a, int texture) {
    gl_.FramebufferTexture2D(GL_FRAMEBUFFER, AttachmentEnum(a), GL_TEXTURE_2D, texture, 0);
}

bool Renderer3D::CheckFrameBufferStatus() {
    return gl_.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
}

void Renderer3D::SetRenderTargets(const std::vector<Attachment> &targets) {
    std::vector<std::uint32_t> t;
    for (Attachment a : targets) t.push_back(AttachmentEnum(a));
    gl_.DrawBuffers(static_cast<int>(t.size()), t.data());
}

void Renderer3D::UseShader(const std::string &name) {
    auto it = shaders_.find(name);
    if (it != shaders_.end()) {
        current_shader_ = name;
        gl_.UseProgram(it->second.program);
    } else {
        gl_.UseProgram(0);
    }
}

int Renderer3D::UniformLocation(const std::string &shader, const std::string &var) {
    std::string key = shader + "_var_" + var;
    auto it = uniform_cache_.find(key);
    if (it != uniform_cache_.end()) return it->second;
    int loc = gl_.GetUniformLocation(shaders_.at(shader).program, var);
    uniform_cache_[key] = loc;
    return loc;
}

void Renderer3D::SetUniformInt(const std::string &shader, const std::string &var, int value) {
    gl_.Uniform1i(UniformLocation(shader, var), value);
}

void Renderer3D::SetUniformFloat(const std::string &shader, const std::string &var, float value) {
    gl_.Uniform1f(UniformLocation(shader, var), value);
}

void Renderer3D::SetUniformFloat2(const std::string &shader, const std::string &var, float a,
                                  float b) {
    gl_.Uniform2f(UniformLocation(shader, var), a, b);
}

void Renderer3D::SetUniformFloat3(const std::string &shader, const std::string &var, float a,
                                  float b, float c) {
    gl_.Uniform3f(UniformLocation(shader, var), a, b, c);
}

void Renderer3D::SetUniformFloat3Array(const std::string &shader, const std::string &var, int count,
                                       const float *values) {
    gl_.Uniform3fv(UniformLocation(shader, var), count, values);
}

void Renderer3D::SetUniformMatrix4(const std::string &shader, const std::string &var,
                                   const Matrix4 &m) {
    gl_.UniformMatrix4fv(UniformLocation(shader, var), 1, true, m.e);
}

void Renderer3D::SetFramebufferGammaCorrection(bool on) {
    if (on)
        gl_.Enable(kFramebufferSrgb);
    else
        gl_.Disable(kFramebufferSrgb);
}

}
