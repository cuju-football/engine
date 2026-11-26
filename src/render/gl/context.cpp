// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The mini GL context. State handling follows Mesa's st/mesa and llvmpipe (MIT licence):
// st_atom_rasterizer.c and st_atom_viewport.c for the orientation rules, main/viewport.c for the
// viewport transform, lp_clear.c and u_pack_color.h for clears, main/readpix.c and
// util/format_utils.h for the readback.

#include "context.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../mipmap.h"
#include "../numerics.h"
#include "mip_cache.h"
#include "pipeline.h"

namespace cuju::render::gl {
struct Context::TextureObject {
    Texture texture;
    std::uint32_t internal_format = 0;
    bool allocated = false;
    std::int32_t min_filter = 0x2702, mag_filter = 0x2601, wrap_s = 0x2901, wrap_t = 0x2901;
    float max_anisotropy = 1.0f;
};

struct Context::BufferObject {
    std::vector<std::uint8_t> data;
};

struct Context::VertexArrayObject {
    struct Attrib {
        bool enabled = false;
        std::uint32_t buffer = 0;
        int size = 4;
        std::uint32_t type = GL_FLOAT;
        bool normalized = false;
        int stride = 0;
        std::size_t offset = 0;
    };
    std::uint32_t element_buffer = 0;
    Attrib attribs[kMaxAttributes];
};

struct Context::FramebufferObject {
    std::uint32_t color[kMaxColorBuffers] = {};
    std::uint32_t depth = 0;
    std::vector<std::uint32_t> draw_buffers = {GL_COLOR_ATTACHMENT0};
};

struct DrawChunk {
    struct Item {
        std::shared_ptr<const DrawState> state;
        RasterState rs;
        DrawInput input;
    };
    std::vector<Item> items;
    long indices = 0;
};

namespace {
constexpr long kChunkIndices = 4096;

}

struct Context::ProgramObject {
    std::unique_ptr<Program> program;
    std::vector<std::uint32_t> uniforms;
};

namespace {
bool FormatOf(std::uint32_t internal, Format *f) {
    switch (internal) {
        case GL_RGB8:
        case GL_RGB:
            *f = Format::R8G8B8X8_UNORM;
            return true;
        case GL_SRGB8:
            *f = Format::R8G8B8X8_SRGB;
            return true;
        case GL_SRGB8_ALPHA8:
            *f = Format::R8G8B8A8_SRGB;
            return true;
        case GL_RGBA8:
        case GL_RGBA:
            *f = Format::R8G8B8A8_UNORM;
            return true;
        case GL_RGBA16F:
            *f = Format::R16G16B16A16_FLOAT;
            return true;
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT:
            *f = Format::Z16_UNORM;
            return true;
        default:
            return false;
    }
}

std::uint16_t HalfRne(float x) {
    const std::uint32_t b = Bits(x);
    const std::uint32_t sign = (b >> 16) & 0x8000u;
    const std::uint32_t a = b & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return static_cast<std::uint16_t>(sign | 0x7E00u);
    if (a >= 0x477FF000u) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (a < 0x33000001u) return static_cast<std::uint16_t>(sign);
    const int e = static_cast<int>(a >> 23);
    std::uint32_t m = (a & 0x7FFFFFu) | 0x800000u;
    if (e < 113) {
        const int shift = 126 - e;
        const std::uint32_t r = m >> shift;
        const std::uint32_t rem = m & ((1u << shift) - 1);
        const std::uint32_t half = 1u << (shift - 1);
        const std::uint32_t v = r + ((rem > half || (rem == half && (r & 1))) ? 1 : 0);
        return static_cast<std::uint16_t>(sign | v);
    }
    std::uint32_t v = ((static_cast<std::uint32_t>(e) - 112u) << 10) | ((m >> 13) & 0x3FFu);
    const std::uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) {
        ++v;
    }
    return static_cast<std::uint16_t>(sign | v);
}

std::uint32_t UnormRound(float v, float scale) {
    const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return static_cast<std::uint32_t>(std::lrint(c * scale));
}

}  // namespace

Context::Context(const ContextOptions &options) : options_(options) {
    pool_ = std::make_unique<ThreadPool>(std::max(1, options.threads));
    scene_ = std::make_unique<SceneList>();
    chunk_ = std::make_unique<DrawChunk>();
    window_.assign(static_cast<std::size_t>(options.width) * ((options.height + 3) & ~3), 0u);
    default_vao_ = std::make_unique<VertexArrayObject>();
    viewport_[2] = options.width;
    viewport_[3] = options.height;
}

Context::~Context() {
    DrainDraws();
    WaitRaster();
}

std::uint32_t Context::GenTexture() {
    const std::uint32_t n = next_texture_++;
    textures_[n] = std::make_unique<TextureObject>();
    return n;
}

void Context::DeleteTexture(std::uint32_t name) {
    Sync();
    textures_.erase(name);
    for (std::uint32_t &t : unit_textures_) {
        if (t == name) {
            t = 0;
        }
    }
}

void Context::ActiveTexture(std::uint32_t unit) { active_unit_ = unit - GL_TEXTURE0; }

void Context::BindTexture(std::uint32_t, std::uint32_t name) {
    if (name && !textures_.count(name)) {
        textures_[name] = std::make_unique<TextureObject>();
    }
    unit_textures_[active_unit_] = name;
}

Context::TextureObject *Context::BoundTexture() {
    auto it = textures_.find(unit_textures_[active_unit_]);
    return it == textures_.end() ? nullptr : it->second.get();
}

void Context::TexParameteri(std::uint32_t, std::uint32_t pname, std::int32_t value) {
    TextureObject *t = BoundTexture();
    if (!t) return;
    switch (pname) {
        case GL_TEXTURE_MIN_FILTER:
            t->min_filter = value;
            break;
        case GL_TEXTURE_MAG_FILTER:
            t->mag_filter = value;
            break;
        case GL_TEXTURE_WRAP_S:
            t->wrap_s = value;
            break;
        case GL_TEXTURE_WRAP_T:
            t->wrap_t = value;
            break;
        default:
            break;
    }
}

void Context::TexParameterf(std::uint32_t target, std::uint32_t pname, float value) {
    TextureObject *t = BoundTexture();
    if (!t) return;
    if (pname == GL_TEXTURE_MAX_ANISOTROPY) {
        t->max_anisotropy = value;
    } else {
        TexParameteri(target, pname, static_cast<std::int32_t>(value));
    }
}

namespace {
void Unpack(TextureLevel &level, Format format, int x, int y, int w, int h,
            std::uint32_t src_format, const std::uint8_t *src) {
    const int bpp = BytesPerTexel(format);
    const int sbpp = src_format == GL_RGB ? 3 : 4;
    const std::size_t sstride =
        (static_cast<std::size_t>(w) * sbpp + 3) & ~static_cast<std::size_t>(3);
    for (int r = 0; r < h; ++r) {
        const std::uint8_t *s = src + r * sstride;
        std::uint8_t *d =
            level.texels.data() + (static_cast<std::size_t>(y + r) * level.width + x) * bpp;
        for (int c = 0; c < w; ++c) {
            if (bpp == 4) {
                d[4 * c + 0] = s[sbpp * c + 0];
                d[4 * c + 1] = s[sbpp * c + 1];
                d[4 * c + 2] = s[sbpp * c + 2];
                d[4 * c + 3] = sbpp == 4 ? s[4 * c + 3] : 0xFF;
            } else {
                std::memcpy(d + bpp * c, s + bpp * c, bpp);
            }
        }
    }
}

}  // namespace

void Context::TexImage2D(std::uint32_t, int level, std::uint32_t internalformat, int width,
                         int height, std::uint32_t format, std::uint32_t, const void *data) {
    Sync();
    TextureObject *t = BoundTexture();
    if (!t) return;
    Format f;
    if (!FormatOf(internalformat, &f)) return;
    if (level == 0 || t->texture.format != f) {
        t->texture.format = f;
    }
    t->internal_format = internalformat;
    if (static_cast<int>(t->texture.levels.size()) <= level) {
        t->texture.levels.resize(level + 1);
    }
    TextureLevel &lv = t->texture.levels[level];
    lv.width = width;
    lv.height = height;
    const int rows = data ? height : (height + 3) & ~3;
    lv.texels.assign(static_cast<std::size_t>(width) * rows * BytesPerTexel(f), 0);
    if (data) {
        Unpack(lv, f, 0, 0, width, height, format, static_cast<const std::uint8_t *>(data));
    }
    t->allocated = true;
}

void Context::TexSubImage2D(std::uint32_t, int level, int x, int y, int width, int height,
                            std::uint32_t format, std::uint32_t, const void *data) {
    Sync();
    TextureObject *t = BoundTexture();
    if (!t || !data || level >= static_cast<int>(t->texture.levels.size())) return;
    Unpack(t->texture.levels[level], t->texture.format, x, y, width, height, format,
           static_cast<const std::uint8_t *>(data));
}

void Context::GenerateMipmap(std::uint32_t) {
    Sync();
    TextureObject *t = BoundTexture();
    if (!t || t->texture.levels.empty()) return;
    const TextureLevel &l0 = t->texture.levels[0];
    if (CanGenerateMipmaps(t->texture.format, l0.width, l0.height)) {
        if (options_.mip_cache) {
            options_.mip_cache->Generate(t->texture);
        } else {
            GenerateMipmaps(t->texture);
        }
    }
}

const Texture *Context::GetTexture(std::uint32_t name) {
    Sync();
    auto it = textures_.find(name);
    return it == textures_.end() ? nullptr : &it->second->texture;
}

std::uint32_t Context::GenBuffer() {
    const std::uint32_t n = next_buffer_++;
    buffers_[n] = std::make_unique<BufferObject>();
    return n;
}

void Context::DeleteBuffer(std::uint32_t name) {
    DrainDraws();
    buffers_.erase(name);
    if (array_buffer_ == name) array_buffer_ = 0;
}

void Context::BindBuffer(std::uint32_t target, std::uint32_t name) {
    if (name && !buffers_.count(name)) {
        buffers_[name] = std::make_unique<BufferObject>();
    }
    if (target == GL_ARRAY_BUFFER) {
        array_buffer_ = name;
    } else if (target == GL_ELEMENT_ARRAY_BUFFER) {
        VertexArrayObject *v = vao_ ? vaos_[vao_].get() : default_vao_.get();
        v->element_buffer = name;
    }
}

Context::BufferObject *Context::BoundBuffer(std::uint32_t target) {
    std::uint32_t name = 0;
    if (target == GL_ARRAY_BUFFER) {
        name = array_buffer_;
    } else if (target == GL_ELEMENT_ARRAY_BUFFER) {
        VertexArrayObject *v = vao_ ? vaos_[vao_].get() : default_vao_.get();
        name = v->element_buffer;
    }
    auto it = buffers_.find(name);
    return it == buffers_.end() ? nullptr : it->second.get();
}

void Context::BufferData(std::uint32_t target, std::size_t size, const void *data, std::uint32_t) {
    DrainDraws();
    BufferObject *b = BoundBuffer(target);
    if (!b) return;
    b->data.assign(size, 0);
    if (data && size) {
        std::memcpy(b->data.data(), data, size);
    }
}

void Context::BufferSubData(std::uint32_t target, std::size_t offset, std::size_t size,
                            const void *data) {
    DrainDraws();
    BufferObject *b = BoundBuffer(target);
    if (!b || !data || offset + size > b->data.size()) return;
    std::memcpy(b->data.data() + offset, data, size);
}

void *Context::MapBufferRange(std::uint32_t target, std::size_t offset, std::size_t length,
                              std::uint32_t) {
    DrainDraws();
    BufferObject *b = BoundBuffer(target);
    if (!b || offset + length > b->data.size()) return nullptr;
    return b->data.data() + offset;
}

void Context::UnmapBuffer(std::uint32_t) {}

std::uint32_t Context::GenVertexArray() {
    const std::uint32_t n = next_vao_++;
    vaos_[n] = std::make_unique<VertexArrayObject>();
    return n;
}

void Context::DeleteVertexArray(std::uint32_t name) {
    vaos_.erase(name);
    if (vao_ == name) vao_ = 0;
}

void Context::BindVertexArray(std::uint32_t name) {
    if (name && !vaos_.count(name)) {
        vaos_[name] = std::make_unique<VertexArrayObject>();
    }
    vao_ = name;
}

void Context::EnableVertexAttribArray(std::uint32_t index) {
    VertexArrayObject *v = vao_ ? vaos_[vao_].get() : default_vao_.get();
    if (index < kMaxAttributes) v->attribs[index].enabled = true;
}

void Context::DisableVertexAttribArray(std::uint32_t index) {
    VertexArrayObject *v = vao_ ? vaos_[vao_].get() : default_vao_.get();
    if (index < kMaxAttributes) v->attribs[index].enabled = false;
}

void Context::VertexAttribPointer(std::uint32_t index, int size, std::uint32_t type,
                                  bool normalized, int stride, std::size_t offset) {
    VertexArrayObject *v = vao_ ? vaos_[vao_].get() : default_vao_.get();
    if (index >= kMaxAttributes) return;
    VertexArrayObject::Attrib &a = v->attribs[index];
    a.buffer = array_buffer_;
    a.size = size;
    a.type = type;
    a.normalized = normalized;
    a.stride = stride;
    a.offset = offset;
}

std::uint32_t Context::GenFramebuffer() {
    const std::uint32_t n = next_fbo_++;
    fbos_[n] = std::make_unique<FramebufferObject>();
    return n;
}

void Context::DeleteFramebuffer(std::uint32_t name) {
    FlushScene();
    fbos_.erase(name);
    if (draw_fb_ == name) draw_fb_ = 0;
}

void Context::BindFramebuffer(std::uint32_t target, std::uint32_t name) {
    if (target == GL_READ_FRAMEBUFFER) return;
    if (name != draw_fb_) {
        FlushScene();
    }
    if (name && !fbos_.count(name)) {
        fbos_[name] = std::make_unique<FramebufferObject>();
    }
    draw_fb_ = name;
}

void Context::FramebufferTexture2D(std::uint32_t, std::uint32_t attachment, std::uint32_t,
                                   std::uint32_t texture, int) {
    FlushScene();
    auto it = fbos_.find(draw_fb_);
    if (it == fbos_.end()) return;
    if (attachment == GL_DEPTH_ATTACHMENT) {
        it->second->depth = texture;
    } else if (attachment >= GL_COLOR_ATTACHMENT0 &&
               attachment < GL_COLOR_ATTACHMENT0 + kMaxColorBuffers) {
        it->second->color[attachment - GL_COLOR_ATTACHMENT0] = texture;
    }
}

void Context::DrawBuffers(int n, const std::uint32_t *buffers) {
    auto it = fbos_.find(draw_fb_);
    if (it == fbos_.end()) return;
    FlushScene();
    it->second->draw_buffers.assign(buffers, buffers + n);
}

std::uint32_t Context::CheckFramebufferStatus(std::uint32_t) { return GL_FRAMEBUFFER_COMPLETE; }

TargetInfo Context::CurrentTarget() {
    TargetInfo t;
    if (draw_fb_ == 0) {
        t.window = true;
        t.width = options_.width;
        t.height = options_.height;
        t.num_color = 1;
        t.color[0].format = Format::B10G10R10X2_UNORM;
        t.color[0].width = t.width;
        t.color[0].height = t.height;
        t.color[0].data = reinterpret_cast<std::uint8_t *>(window_.data());
        t.color[0].stride = 4 * t.width;
        return t;
    }
    FramebufferObject &f = *fbos_[draw_fb_];
    int w = 1 << 30, h = 1 << 30;
    auto surface = [&](std::uint32_t name, Surface *s) {
        auto it = textures_.find(name);
        if (it == textures_.end() || it->second->texture.levels.empty()) return;
        TextureLevel &lv = it->second->texture.levels[0];
        s->format = it->second->texture.format;
        s->width = lv.width;
        s->height = lv.height;
        s->data = lv.texels.data();
        s->stride = lv.width * BytesPerTexel(s->format);
        w = std::min(w, lv.width);
        h = std::min(h, lv.height);
    };
    t.num_color = static_cast<int>(std::min<std::size_t>(f.draw_buffers.size(), kMaxColorBuffers));
    for (int i = 0; i < t.num_color; ++i) {
        const std::uint32_t db = f.draw_buffers[i];
        if (db >= GL_COLOR_ATTACHMENT0 && db < GL_COLOR_ATTACHMENT0 + kMaxColorBuffers) {
            surface(f.color[db - GL_COLOR_ATTACHMENT0], &t.color[i]);
        }
    }
    if (f.depth) {
        surface(f.depth, &t.depth);
    }
    t.width = w == (1 << 30) ? 0 : w;
    t.height = h == (1 << 30) ? 0 : h;
    return t;
}

std::uint32_t Context::CreateProgram(std::unique_ptr<Program> program) {
    const std::uint32_t n = next_program_++;
    auto po = std::make_unique<ProgramObject>();
    if (program) {
        po->uniforms.assign(program->StorageSize(), 0u);
    }
    po->program = std::move(program);
    programs_[n] = std::move(po);
    return n;
}

void Context::DeleteProgram(std::uint32_t name) {
    Sync();
    programs_.erase(name);
}

void Context::UseProgram(std::uint32_t name) { program_ = name; }

int Context::GetUniformLocation(std::uint32_t program, const std::string &name) const {
    auto it = programs_.find(program);
    if (it == programs_.end() || !it->second->program) return -1;
    return it->second->program->UniformLocation(name);
}

const Program *Context::GetProgram(std::uint32_t name) const {
    auto it = programs_.find(name);
    return it == programs_.end() ? nullptr : it->second->program.get();
}

void Context::UniformBits(int loc, int components, const std::uint32_t *bits) {
    auto it = programs_.find(program_);
    if (loc < 0 || it == programs_.end() || !it->second->program) return;
    const Program &p = *it->second->program;
    int done = 0;
    while (done < components && loc < p.NumLocations()) {
        const int n = std::min(p.LocationComponents(loc), components - done);
        std::memcpy(&it->second->uniforms[p.LocationOffset(loc)], bits + done, 4 * n);
        done += p.LocationComponents(loc);
        ++loc;
    }
}

void Context::Uniform1i(int loc, std::int32_t v) {
    const std::uint32_t b = static_cast<std::uint32_t>(v);
    UniformBits(loc, 1, &b);
}
void Context::Uniform1f(int loc, float v) {
    const std::uint32_t b = Bits(v);
    UniformBits(loc, 1, &b);
}
void Context::Uniform2f(int loc, float x, float y) {
    const std::uint32_t b[2] = {Bits(x), Bits(y)};
    UniformBits(loc, 2, b);
}
void Context::Uniform3f(int loc, float x, float y, float z) {
    const std::uint32_t b[3] = {Bits(x), Bits(y), Bits(z)};
    UniformBits(loc, 3, b);
}
void Context::Uniform3fv(int loc, int count, const float *v) {
    std::vector<std::uint32_t> b(static_cast<std::size_t>(count) * 3);
    std::memcpy(b.data(), v, b.size() * 4);
    UniformBits(loc, static_cast<int>(b.size()), b.data());
}
void Context::UniformMatrix4fv(int loc, int count, bool transpose, const float *v) {
    std::vector<std::uint32_t> b(static_cast<std::size_t>(count) * 16);
    for (int m = 0; m < count; ++m) {
        for (int i = 0; i < 16; ++i) {
            const int src = transpose ? (i % 4) * 4 + i / 4 : i;
            b[m * 16 + i] = Bits(v[m * 16 + src]);
        }
    }
    UniformBits(loc, static_cast<int>(b.size()), b.data());
}

void Context::Enable(std::uint32_t cap) {
    if (cap == GL_CULL_FACE) cull_ = true;
    if (cap == GL_DEPTH_TEST) depth_test_ = true;
    if (cap == GL_BLEND) blend_ = true;
}
void Context::Disable(std::uint32_t cap) {
    if (cap == GL_CULL_FACE) cull_ = false;
    if (cap == GL_DEPTH_TEST) depth_test_ = false;
    if (cap == GL_BLEND) blend_ = false;
}
void Context::CullFace(std::uint32_t mode) { cull_mode_ = mode; }
void Context::FrontFace(std::uint32_t mode) { front_face_ = mode; }
void Context::DepthFunc(std::uint32_t func) { depth_func_ = func; }
void Context::DepthMask(bool flag) { depth_mask_ = flag; }
void Context::BlendFunc(std::uint32_t src, std::uint32_t dst) {
    blend_src_ = src;
    blend_dst_ = dst;
}
void Context::Viewport(int x, int y, int width, int height) {
    viewport_[0] = x;
    viewport_[1] = y;
    viewport_[2] = width;
    viewport_[3] = height;
}
void Context::ClearColor(float r, float g, float b, float a) {
    clear_color_[0] = r;
    clear_color_[1] = g;
    clear_color_[2] = b;
    clear_color_[3] = a;
}
void Context::ClearDepth(double depth) { clear_depth_ = std::min(1.0, std::max(0.0, depth)); }

void Context::Clear(std::uint32_t mask) {
    if (discard_) return;
    if (!depth_mask_) {
        mask &= ~GL_DEPTH_BUFFER_BIT;
    }
    const TargetInfo target = CurrentTarget();
    if (!scene_->Begun()) {
        scene_->Begin(target);
    }
    SubmitChunk();
    Scene &scene = scene_->Direct();
    if (mask & GL_COLOR_BUFFER_BIT) {
        for (int i = 0; i < target.num_color; ++i) {
            const Surface &s = target.color[i];
            if (!s.data) continue;
            std::uint8_t v[8] = {};
            if (s.format == Format::R16G16B16A16_FLOAT) {
                for (int c = 0; c < 4; ++c) {
                    const std::uint16_t h = HalfRne(clear_color_[c]);
                    std::memcpy(v + 2 * c, &h, 2);
                }
            } else if (s.format == Format::B10G10R10X2_UNORM) {
                const std::uint32_t w = UnormRound(clear_color_[2], 1023.0f) |
                                        (UnormRound(clear_color_[1], 1023.0f) << 10) |
                                        (UnormRound(clear_color_[0], 1023.0f) << 20);
                std::memcpy(v, &w, 4);
            } else {
                for (int c = 0; c < 4; ++c) {
                    v[c] = static_cast<std::uint8_t>(UnormRound(clear_color_[c], 255.0f));
                }
            }
            scene.ClearColor(i, v, BytesPerTexel(s.format));
        }
    }
    if ((mask & GL_DEPTH_BUFFER_BIT) && target.depth.data) {
        const std::uint16_t z =
            clear_depth_ >= 1.0 ? 0xFFFF
                                : static_cast<std::uint16_t>(std::lrint(clear_depth_ * 65535.0));
        scene.ClearDepth(z);
    }
}

std::shared_ptr<const DrawState> Context::SnapshotState() {
    auto it = programs_.find(program_);
    auto s = std::make_shared<DrawState>();
    s->program = it->second->program.get();
    s->uniforms = it->second->uniforms;
    for (int u = 0; u < kMaxTextureUnits; ++u) {
        auto t = textures_.find(unit_textures_[u]);
        s->textures[u] = t == textures_.end() ? nullptr : &t->second->texture;
    }
    s->ops.depth_test = depth_test_;
    s->ops.depth_func = depth_func_;
    s->ops.depth_write = depth_mask_;
    s->ops.blend = blend_;
    s->ops.blend_src = blend_src_;
    s->ops.blend_dst = blend_dst_;
    s->target = CurrentTarget();
    return s;
}

void Context::DrawArrays(std::uint32_t mode, int first, int count) {
    if (mode != GL_TRIANGLES) return;
    DrawInput in;
    in.first = first;
    in.count = count;
    Draw(in);
}

void Context::DrawRangeElements(std::uint32_t mode, std::uint32_t, std::uint32_t, int count,
                                std::uint32_t type, std::size_t offset) {
    if (mode != GL_TRIANGLES) return;
    VertexArrayObject *v = vao_ ? vaos_[vao_].get() : default_vao_.get();
    auto eb = buffers_.find(v->element_buffer);
    if (eb == buffers_.end()) return;
    DrawInput in;
    in.index_size = type == GL_UNSIGNED_INT ? 4 : type == GL_UNSIGNED_SHORT ? 2 : 1;
    in.indices = eb->second->data.data();
    in.index_space = eb->second->data.size();
    in.first = static_cast<int>(offset / static_cast<std::size_t>(in.index_size));
    in.count = count;
    Draw(in);
}

void Context::Draw(DrawInput &in) {
    auto pit = programs_.find(program_);
    ++stats_.draws;
    if (discard_) {
        ++stats_.draws_skipped;
        return;
    }
    if (pit == programs_.end() || !pit->second->program) {
        ++stats_.draws_skipped;
        return;
    }
    const Program &p = *pit->second->program;
    VertexArrayObject *v = vao_ ? vaos_[vao_].get() : default_vao_.get();
    for (int l = 0; l < kMaxAttributes; ++l) {
        const VertexArrayObject::Attrib &a = v->attribs[l];
        AttributeSource &s = in.attribs[l];
        s.enabled = a.enabled;
        auto b = buffers_.find(a.buffer);
        if (a.enabled && b != buffers_.end()) {
            s.data = b->second->data.data();
            s.buffer_size = b->second->data.size();
        }
        s.offset = a.offset;
        s.size = a.size;
        s.stride = a.stride ? a.stride : 4 * a.size;
    }
    const TargetInfo target = CurrentTarget();
    if (!scene_->Begun()) {
        scene_->Begin(target);
    }
    RasterState rs;
    rs.fb_width = target.width;
    rs.fb_height = target.height;
    rs.bottom_edge_rule = target.window;
    rs.cull_face = !cull_ ? 0u : cull_mode_ == GL_FRONT ? 1u : cull_mode_ == GL_BACK ? 2u : 3u;
    rs.front_ccw = (front_face_ == 0x0901) != !target.window;
    const float half_w = 0.5f * static_cast<float>(viewport_[2]);
    const float half_h = 0.5f * static_cast<float>(viewport_[3]);
    rs.scale[0] = half_w;
    rs.translate[0] = half_w + static_cast<float>(viewport_[0]);
    rs.scale[1] = half_h;
    rs.translate[1] = half_h + static_cast<float>(viewport_[1]);
    rs.scale[2] = 0.5f;
    rs.translate[2] = 0.5f;
    if (target.window) {
        rs.scale[1] = -rs.scale[1];
        rs.translate[1] = static_cast<float>(target.height) - rs.translate[1];
    }
    rs.num_inputs = p.NumInputs();
    for (int i = 0; i < rs.num_inputs; ++i) {
        rs.interp[i] = p.InputInterp(i);
    }
    if (pool_->Threads() == 0) {
        Scene &scene = scene_->Direct();
        const DrawState *kept = scene.Keep(SnapshotState());
        RunDraw(*kept, rs, in, scene);
        return;
    }
    chunk_->items.push_back({SnapshotState(), rs, in});
    chunk_->indices += in.count;
    if (chunk_->indices >= kChunkIndices) {
        SubmitChunk();
    }
}

void Context::SubmitChunk() {
    if (chunk_->items.empty()) {
        return;
    }
    Scene *part = &scene_->NewPart();
    auto items = std::make_shared<std::vector<DrawChunk::Item>>(std::move(chunk_->items));
    chunk_->items.clear();
    chunk_->indices = 0;
    draw_jobs_.push_back(pool_->Submit(1, [part, items](int) {
        for (DrawChunk::Item &item : *items) {
            const DrawState *kept = part->Keep(std::move(item.state));
            RunDraw(*kept, item.rs, item.input, *part);
        }
    }));
}

void Context::DrainDraws() {
    SubmitChunk();
    for (const std::shared_ptr<void> &job : draw_jobs_) {
        pool_->Wait(std::static_pointer_cast<ThreadPool::Job>(job));
    }
    draw_jobs_.clear();
}

void Context::FlushScene() {
    DrainDraws();
    if (!scene_->Begun()) {
        return;
    }
    stats_.triangles_binned += scene_->TrianglesBinned();
    if (scene_->Empty()) {
        scene_->Reset();
        return;
    }
    if (pool_->Threads() == 0) {
        WaitRaster();
        scene_->Rasterize(pool_.get(), &stats_.blocks_shaded);
        return;
    }
    std::size_t wait = rastering_.size() >= 2 ? rastering_.size() - 1 : 0;
    for (std::size_t i = 0; i < rastering_.size(); ++i) {
        if (SceneList::Conflict(*rastering_[i], *scene_)) wait = std::max(wait, i + 1);
    }
    FinishRaster(wait);
    scene_->StartRasterize(pool_.get());
    rastering_.push_back(std::move(scene_));
    if (spare_.empty()) {
        scene_ = std::make_unique<SceneList>();
    } else {
        scene_ = std::move(spare_.back());
        spare_.pop_back();
    }
}

void Context::FinishRaster(std::size_t count) {
    count = std::min(count, rastering_.size());
    for (std::size_t i = 0; i < count; ++i) {
        rastering_[i]->FinishRasterize(pool_.get(), &stats_.blocks_shaded);
        spare_.push_back(std::move(rastering_[i]));
    }
    rastering_.erase(rastering_.begin(), rastering_.begin() + static_cast<std::ptrdiff_t>(count));
}

void Context::WaitRaster() { FinishRaster(rastering_.size()); }

void Context::Sync() {
    FlushScene();
    WaitRaster();
}

void Context::Finish() { Sync(); }

void Context::Parallel(int n, const std::function<void(int)> &fn) { pool_->Run(n, fn); }

void Context::SetDiscard(bool discard) {
    if (discard && !discard_) {
        FlushScene();
    }
    discard_ = discard;
}

namespace {
struct Unorm10To8 {
    std::uint8_t v[1024];
    Unorm10To8() {
        for (std::uint32_t i = 0; i < 1024; ++i) {
            v[i] = static_cast<std::uint8_t>((i * 255u + 511u) / 1023u);
        }
    }
};

const Unorm10To8 kUnorm10To8;

}  // namespace

void Context::ReadPixels(int x, int y, int width, int height, std::uint32_t format,
                         std::uint32_t type, void *out) {
    Sync();
    if (discard_ || format != GL_RGB || type != GL_UNSIGNED_BYTE) return;
    std::uint8_t *dst = static_cast<std::uint8_t *>(out);
    const std::size_t stride =
        (static_cast<std::size_t>(width) * 3 + 3) & ~static_cast<std::size_t>(3);
    const std::uint8_t *lut = kUnorm10To8.v;
    constexpr int kBand = 32;
    const int bands = (height + kBand - 1) / kBand;
    const std::function<void(int)> band = [&](int k) {
        const int end = std::min(height, (k + 1) * kBand);
        for (int r = k * kBand; r < end; ++r) {
            const int mem_row = options_.height - 1 - (y + r);
            const std::uint32_t *src =
                window_.data() + static_cast<std::size_t>(mem_row) * options_.width + x;
            std::uint8_t *d = dst + r * stride;
            for (int c = 0; c < width; ++c) {
                const std::uint32_t w = src[c];
                d[3 * c + 0] = lut[(w >> 20) & 0x3FFu];
                d[3 * c + 1] = lut[(w >> 10) & 0x3FFu];
                d[3 * c + 2] = lut[w & 0x3FFu];
            }
        }
    };
    pool_->Run(bands, band);
}

std::vector<std::uint32_t> Context::ReadDefaultFramebufferWords() {
    Sync();
    std::vector<std::uint32_t> out(static_cast<std::size_t>(options_.width) * options_.height);
    for (int r = 0; r < options_.height; ++r) {
        std::memcpy(&out[static_cast<std::size_t>(r) * options_.width],
                    &window_[static_cast<std::size_t>(options_.height - 1 - r) * options_.width],
                    4 * static_cast<std::size_t>(options_.width));
    }
    return out;
}

}  // namespace cuju::render::gl
