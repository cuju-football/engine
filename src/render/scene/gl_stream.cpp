// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include "gl_stream.h"

#include <cstring>

namespace cuju::render::scene {
namespace {
constexpr const char *kOpNames[] = {
    "glBindTexture",
    "glBlendFunc",
    "glClear",
    "glClearColor",
    "glClearDepth",
    "glCullFace",
    "glDeleteTextures",
    "glDepthFunc",
    "glDepthMask",
    "glGenVertexArrays",
    "glBindBuffer",
    "glBindVertexArray",
    "glBufferData",
    "glBufferSubData",
    "glDeleteFramebuffers",
    "glBindFramebuffer",
    "glUnmapBuffer",
    "glLinkProgram",
    "glUseProgram",
    "glCreateProgram",
    "glIsShader",
    "glGetShaderiv",
    "glCreateShader",
    "glDrawBuffers",
    "glShaderSource",
    "glCompileShader",
    "glGetUniformLocation",
    "glUniform1i",
    "glUniform3f",
    "glUniformMatrix4fv",
    "glUniform1f",
    "glUniform2f",
    "glUniform3fv",
    "glReadPixels",
    "glBindAttribLocation",
    "glBindFragDataLocation",
    "glAttachShader",
    "glActiveTexture",
    "glCheckFramebufferStatus",
    "glFramebufferTexture2D",
    "glGenFramebuffers",
    "glEnableVertexAttribArray",
    "glVertexAttribPointer",
    "glDeleteVertexArrays",
    "glDeleteBuffers",
    "glDisable",
    "glDrawArrays",
    "glEnable",
    "glGenBuffers",
    "glGenerateMipmap",
    "glGenTextures",
    "glGetFloatv",
    "glMapBufferRange",
    "glDrawRangeElements",
    "glTexImage2D",
    "glTexParameterf",
    "glTexParameteri",
    "glTexSubImage2D",
    "glViewport",
};
static_assert(sizeof(kOpNames) / sizeof(kOpNames[0]) == static_cast<int>(Op::kCount));

std::uint64_t F(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}
std::uint64_t D(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, 8);
    return b;
}
std::uint64_t I(std::int64_t v) { return static_cast<std::uint64_t>(v); }

constexpr std::uint32_t kCompileStatus = 0x8B81;

}  // namespace

const char *OpName(Op op) { return kOpNames[static_cast<int>(op)]; }

std::uint32_t NameAllocator::Alloc() {
    for (std::uint32_t i = 1; i < used_.size(); ++i) {
        if (!used_[i]) {
            used_[i] = true;
            return i;
        }
    }
    used_.push_back(true);
    return static_cast<std::uint32_t>(used_.size() - 1);
}

void NameAllocator::Free(std::uint32_t name) {
    if (name != 0 && name < used_.size()) used_[name] = false;
}

std::size_t UploadBytes(int width, int height, std::uint32_t format, std::uint32_t type) {
    int bpp = 4;
    if (format == gl::GL_RGB) bpp = 3;
    if (format == gl::GL_DEPTH_COMPONENT) bpp = type == gl::GL_UNSIGNED_SHORT ? 2 : 4;
    std::size_t row = static_cast<std::size_t>(width) * bpp;
    std::size_t stride = (row + 3) & ~static_cast<std::size_t>(3);
    if (height <= 0 || width <= 0) return 0;
    return stride * (height - 1) + row;
}

GlStream::GlStream(gl::Context *context, Recorder *recorder)
    : context_(context), recorder_(recorder) {}

GlStream::~GlStream() = default;

Call &GlStream::Record(Op op, std::initializer_list<std::uint64_t> args) {
    if (recorder_->before) recorder_->before(op);
    recorder_->calls.emplace_back();
    Call &c = recorder_->calls.back();
    c.op = op;
    c.nargs = static_cast<std::uint8_t>(args.size());
    int i = 0;
    for (std::uint64_t a : args) c.args[i++] = a;
    return c;
}

void GlStream::Item(Call *call, const void *data, std::size_t size) {
    call->has_item = true;
    const auto *p = static_cast<const std::uint8_t *>(data);
    call->item.assign(p, p + size);
}

std::uint32_t GlStream::Ctx(const std::unordered_map<std::uint32_t, std::uint32_t> &map,
                            std::uint32_t name) const {
    if (name == 0) return 0;
    auto it = map.find(name);
    return it == map.end() ? 0 : it->second;
}

void GlStream::GetFloatv(std::uint32_t pname, float *out) {
    *out = 16.0f;
    if (recorder_) Item(&Record(Op::kGetFloatv, {pname, 0}), out, 4);
}

std::uint32_t GlStream::CreateShader(std::uint32_t type) {
    std::uint32_t name = shader_objects_.Alloc();
    if (recorder_) {
        Call &c = Record(Op::kCreateShader, {type});
        c.has_ret = true;
        c.ret = name;
    }
    return name;
}

void GlStream::ShaderSource(std::uint32_t shader, const std::string &file) {
    if (recorder_) Item(&Record(Op::kShaderSource, {shader, 1, 0, 0}), file.data(), file.size());
}

void GlStream::CompileShader(std::uint32_t shader) {
    if (recorder_) Record(Op::kCompileShader, {shader});
}

int GlStream::GetShaderiv(std::uint32_t shader, std::uint32_t pname) {
    std::int32_t v = pname == kCompileStatus ? 1 : 0;
    if (recorder_) Item(&Record(Op::kGetShaderiv, {shader, pname, 0}), &v, 4);
    return v;
}

bool GlStream::IsShader(std::uint32_t shader) {
    if (recorder_) {
        Call &c = Record(Op::kIsShader, {shader});
        c.has_ret = true;
        c.ret = 1;
    }
    return true;
}

std::uint32_t GlStream::CreateProgram(gl::ProgramKind kind) {
    std::uint32_t name = shader_objects_.Alloc();
    if (recorder_) {
        Call &c = Record(Op::kCreateProgram, {});
        c.has_ret = true;
        c.ret = name;
    }
    programs_[name] = gl::MakeProgram(kind);
    if (context_) ctx_programs_[name] = context_->CreateProgram(gl::MakeProgram(kind));
    return name;
}

void GlStream::AttachShader(std::uint32_t program, std::uint32_t shader) {
    if (recorder_) Record(Op::kAttachShader, {program, shader});
}

void GlStream::BindAttribLocation(std::uint32_t program, std::uint32_t index,
                                  const std::string &name) {
    if (recorder_)
        Item(&Record(Op::kBindAttribLocation, {program, index, 0}), name.data(), name.size());
}

void GlStream::BindFragDataLocation(std::uint32_t program, std::uint32_t color,
                                    const std::string &name) {
    if (recorder_)
        Item(&Record(Op::kBindFragDataLocation, {program, color, 0}), name.data(), name.size());
}

void GlStream::LinkProgram(std::uint32_t program) {
    if (recorder_) Record(Op::kLinkProgram, {program});
}

void GlStream::UseProgram(std::uint32_t program) {
    if (recorder_) Record(Op::kUseProgram, {program});
    if (context_) context_->UseProgram(Ctx(ctx_programs_, program));
}

int GlStream::GetUniformLocation(std::uint32_t program, const std::string &name) {
    int loc = -1;
    auto it = programs_.find(program);
    if (it != programs_.end() && it->second) loc = it->second->UniformLocation(name);
    if (recorder_) {
        Call &c = Record(Op::kGetUniformLocation, {program, 0});
        c.has_ret = true;
        c.ret = I(loc);
        Item(&c, name.data(), name.size());
    }
    return loc;
}

void GlStream::Uniform1i(int loc, std::int32_t v) {
    if (recorder_) Record(Op::kUniform1i, {I(loc), I(v)});
    if (context_) context_->Uniform1i(loc, v);
}

void GlStream::Uniform1f(int loc, float v) {
    if (recorder_) Record(Op::kUniform1f, {I(loc), F(v)});
    if (context_) context_->Uniform1f(loc, v);
}

void GlStream::Uniform2f(int loc, float x, float y) {
    if (recorder_) Record(Op::kUniform2f, {I(loc), F(x), F(y)});
    if (context_) context_->Uniform2f(loc, x, y);
}

void GlStream::Uniform3f(int loc, float x, float y, float z) {
    if (recorder_) Record(Op::kUniform3f, {I(loc), F(x), F(y), F(z)});
    if (context_) context_->Uniform3f(loc, x, y, z);
}

void GlStream::Uniform3fv(int loc, int count, const float *v) {
    if (recorder_)
        Item(&Record(Op::kUniform3fv, {I(loc), I(count), 0}), v,
             12 * static_cast<std::size_t>(count));
    if (context_) context_->Uniform3fv(loc, count, v);
}

void GlStream::UniformMatrix4fv(int loc, int count, bool transpose, const float *v) {
    if (recorder_)
        Item(&Record(Op::kUniformMatrix4fv, {I(loc), I(count), transpose ? 1u : 0u, 0}), v,
             64 * static_cast<std::size_t>(count));
    if (context_) context_->UniformMatrix4fv(loc, count, transpose, v);
}

std::uint32_t GlStream::GenTexture() {
    std::uint32_t name = textures_.Alloc();
    if (recorder_) Item(&Record(Op::kGenTextures, {1, 0}), &name, 4);
    if (context_) ctx_textures_[name] = context_->GenTexture();
    return name;
}

void GlStream::DeleteTexture(std::uint32_t name) {
    if (recorder_) Item(&Record(Op::kDeleteTextures, {1, 0}), &name, 4);
    if (context_) {
        context_->DeleteTexture(Ctx(ctx_textures_, name));
        ctx_textures_.erase(name);
    }
    textures_.Free(name);
}

void GlStream::ActiveTexture(std::uint32_t unit) {
    if (recorder_) Record(Op::kActiveTexture, {unit});
    if (context_) context_->ActiveTexture(unit);
}

void GlStream::BindTexture(std::uint32_t target, std::uint32_t name) {
    if (recorder_) Record(Op::kBindTexture, {target, name});
    if (context_) context_->BindTexture(target, Ctx(ctx_textures_, name));
}

void GlStream::TexParameteri(std::uint32_t target, std::uint32_t pname, std::int32_t value) {
    if (recorder_) Record(Op::kTexParameteri, {target, pname, I(value)});
    if (context_) context_->TexParameteri(target, pname, value);
}

void GlStream::TexParameterf(std::uint32_t target, std::uint32_t pname, float value) {
    if (recorder_) Record(Op::kTexParameterf, {target, pname, F(value)});
    if (context_) context_->TexParameterf(target, pname, value);
}

void GlStream::TexImage2D(std::uint32_t target, int level, std::uint32_t internalformat, int width,
                          int height, std::uint32_t format, std::uint32_t type, const void *data) {
    if (recorder_) {
        Call &c = Record(Op::kTexImage2D, {target, I(level), I(internalformat), I(width), I(height),
                                           0, format, type, 0});
        if (data) Item(&c, data, UploadBytes(width, height, format, type));
    }
    if (context_)
        context_->TexImage2D(target, level, internalformat, width, height, format, type, data);
}

void GlStream::TexSubImage2D(std::uint32_t target, int level, int x, int y, int width, int height,
                             std::uint32_t format, std::uint32_t type, const void *data) {
    if (recorder_) {
        Call &c = Record(Op::kTexSubImage2D,
                         {target, I(level), I(x), I(y), I(width), I(height), format, type, 0});
        if (data) Item(&c, data, UploadBytes(width, height, format, type));
    }
    if (context_) context_->TexSubImage2D(target, level, x, y, width, height, format, type, data);
}

void GlStream::GenerateMipmap(std::uint32_t target) {
    if (recorder_) Record(Op::kGenerateMipmap, {target});
    if (context_) context_->GenerateMipmap(target);
}

std::uint32_t GlStream::GenBuffer() {
    std::uint32_t name = buffers_.Alloc();
    if (recorder_) Item(&Record(Op::kGenBuffers, {1, 0}), &name, 4);
    if (context_) ctx_buffers_[name] = context_->GenBuffer();
    return name;
}

void GlStream::DeleteBuffer(std::uint32_t name) {
    if (recorder_) Item(&Record(Op::kDeleteBuffers, {1, 0}), &name, 4);
    if (context_) {
        context_->DeleteBuffer(Ctx(ctx_buffers_, name));
        ctx_buffers_.erase(name);
    }
    buffers_.Free(name);
}

void GlStream::BindBuffer(std::uint32_t target, std::uint32_t name) {
    if (recorder_) Record(Op::kBindBuffer, {target, name});
    if (context_) context_->BindBuffer(target, Ctx(ctx_buffers_, name));
}

void GlStream::BufferData(std::uint32_t target, std::size_t size, const void *data,
                          std::uint32_t usage) {
    if (recorder_) {
        Call &c = Record(Op::kBufferData, {target, size, 0, usage});
        if (data) Item(&c, data, size);
    }
    if (context_) context_->BufferData(target, size, data, usage);
}

void GlStream::BufferSubData(std::uint32_t target, std::size_t offset, std::size_t size,
                             const void *data) {
    if (recorder_) Item(&Record(Op::kBufferSubData, {target, offset, size, 0}), data, size);
    if (context_) context_->BufferSubData(target, offset, size, data);
}

void GlStream::MapCopyUnmap(std::uint32_t target, std::size_t offset, std::size_t length,
                            std::uint32_t access, const void *data) {
    if (recorder_) {
        Call &m = Record(Op::kMapBufferRange, {target, offset, length, access});
        m.has_ret = true;
        Item(&Record(Op::kUnmapBuffer, {target}), data, length);
    }
    if (context_) {
        void *p = context_->MapBufferRange(target, offset, length, access);
        std::memcpy(p, data, length);
        context_->UnmapBuffer(target);
    }
}

std::uint32_t GlStream::GenVertexArray() {
    std::uint32_t name = vaos_.Alloc();
    if (recorder_) Item(&Record(Op::kGenVertexArrays, {1, 0}), &name, 4);
    if (context_) ctx_vaos_[name] = context_->GenVertexArray();
    return name;
}

void GlStream::DeleteVertexArray(std::uint32_t name) {
    if (recorder_) Item(&Record(Op::kDeleteVertexArrays, {1, 0}), &name, 4);
    if (context_) {
        context_->DeleteVertexArray(Ctx(ctx_vaos_, name));
        ctx_vaos_.erase(name);
    }
    vaos_.Free(name);
}

void GlStream::BindVertexArray(std::uint32_t name) {
    if (recorder_) Record(Op::kBindVertexArray, {name});
    if (context_) context_->BindVertexArray(Ctx(ctx_vaos_, name));
}

void GlStream::EnableVertexAttribArray(std::uint32_t index) {
    if (recorder_) Record(Op::kEnableVertexAttribArray, {index});
    if (context_) context_->EnableVertexAttribArray(index);
}

void GlStream::VertexAttribPointer(std::uint32_t index, int size, std::uint32_t type,
                                   bool normalized, int stride, std::size_t offset) {
    if (recorder_)
        Record(Op::kVertexAttribPointer,
               {index, I(size), type, normalized ? 1u : 0u, I(stride), offset});
    if (context_) context_->VertexAttribPointer(index, size, type, normalized, stride, offset);
}

std::uint32_t GlStream::GenFramebuffer() {
    std::uint32_t name = fbos_.Alloc();
    if (recorder_) Item(&Record(Op::kGenFramebuffers, {1, 0}), &name, 4);
    if (context_) ctx_fbos_[name] = context_->GenFramebuffer();
    return name;
}

void GlStream::DeleteFramebuffer(std::uint32_t name) {
    if (recorder_) Item(&Record(Op::kDeleteFramebuffers, {1, 0}), &name, 4);
    if (context_) {
        context_->DeleteFramebuffer(Ctx(ctx_fbos_, name));
        ctx_fbos_.erase(name);
    }
    fbos_.Free(name);
}

void GlStream::BindFramebuffer(std::uint32_t target, std::uint32_t name) {
    if (recorder_) Record(Op::kBindFramebuffer, {target, name});
    if (context_) context_->BindFramebuffer(target, Ctx(ctx_fbos_, name));
}

void GlStream::FramebufferTexture2D(std::uint32_t target, std::uint32_t attachment,
                                    std::uint32_t textarget, std::uint32_t texture, int level) {
    if (recorder_)
        Record(Op::kFramebufferTexture2D, {target, attachment, textarget, texture, I(level)});
    if (context_)
        context_->FramebufferTexture2D(target, attachment, textarget, Ctx(ctx_textures_, texture),
                                       level);
}

void GlStream::DrawBuffers(int n, const std::uint32_t *buffers) {
    if (recorder_)
        Item(&Record(Op::kDrawBuffers, {I(n), 0}), buffers, 4 * static_cast<std::size_t>(n));
    if (context_) context_->DrawBuffers(n, buffers);
}

std::uint32_t GlStream::CheckFramebufferStatus(std::uint32_t target) {
    std::uint32_t status = gl::GL_FRAMEBUFFER_COMPLETE;
    if (context_) status = context_->CheckFramebufferStatus(target);
    if (recorder_) {
        Call &c = Record(Op::kCheckFramebufferStatus, {target});
        c.has_ret = true;
        c.ret = status;
    }
    return status;
}

void GlStream::Enable(std::uint32_t cap) {
    if (recorder_) Record(Op::kEnable, {cap});
    if (context_) context_->Enable(cap);
}

void GlStream::Disable(std::uint32_t cap) {
    if (recorder_) Record(Op::kDisable, {cap});
    if (context_) context_->Disable(cap);
}

void GlStream::CullFace(std::uint32_t mode) {
    if (recorder_) Record(Op::kCullFace, {mode});
    if (context_) context_->CullFace(mode);
}

void GlStream::DepthFunc(std::uint32_t func) {
    if (recorder_) Record(Op::kDepthFunc, {func});
    if (context_) context_->DepthFunc(func);
}

void GlStream::DepthMask(bool flag) {
    if (recorder_) Record(Op::kDepthMask, {flag ? 1u : 0u});
    if (context_) context_->DepthMask(flag);
}

void GlStream::BlendFunc(std::uint32_t src, std::uint32_t dst) {
    if (recorder_) Record(Op::kBlendFunc, {src, dst});
    if (context_) context_->BlendFunc(src, dst);
}

void GlStream::Viewport(int x, int y, int width, int height) {
    if (recorder_) Record(Op::kViewport, {I(x), I(y), I(width), I(height)});
    if (context_) context_->Viewport(x, y, width, height);
}

void GlStream::ClearColor(float r, float g, float b, float a) {
    if (recorder_) Record(Op::kClearColor, {F(r), F(g), F(b), F(a)});
    if (context_) context_->ClearColor(r, g, b, a);
}

void GlStream::ClearDepth(double depth) {
    if (recorder_) Record(Op::kClearDepth, {D(depth)});
    if (context_) context_->ClearDepth(depth);
}

void GlStream::Clear(std::uint32_t mask) {
    if (recorder_) Record(Op::kClear, {mask});
    if (context_) context_->Clear(mask);
}

void GlStream::DrawArrays(std::uint32_t mode, int first, int count) {
    if (recorder_) Record(Op::kDrawArrays, {mode, I(first), I(count)});
    if (context_) context_->DrawArrays(mode, first, count);
}

void GlStream::DrawRangeElements(std::uint32_t mode, std::uint32_t start, std::uint32_t end,
                                 int count, std::uint32_t type, std::size_t offset) {
    if (recorder_) Record(Op::kDrawRangeElements, {mode, start, end, I(count), type, offset});
    if (context_) context_->DrawRangeElements(mode, start, end, count, type, offset);
}

void GlStream::ReadPixels(int x, int y, int width, int height, std::uint32_t format,
                          std::uint32_t type, void *out) {
    if (context_) context_->ReadPixels(x, y, width, height, format, type, out);
    if (recorder_) {
        Call &c = Record(Op::kReadPixels, {I(x), I(y), I(width), I(height), format, type, 0});
        if (context_) Item(&c, out, static_cast<std::size_t>(width) * height * 3);
    }
}

}  // namespace cuju::render::scene
