// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The GL calls of GRF's renderer as the scene port issues them. Each call names its objects as
// Mesa does, the lowest free name per namespace, so the names equal GRF's. A stream forwards the
// calls to a mini GL context, records them for comparison with GRF's traces, or both.

#ifndef CUJU_RENDER_SCENE_GL_STREAM_H
#define CUJU_RENDER_SCENE_GL_STREAM_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../gl/context.h"
#include "../gl/program.h"

namespace cuju::render::scene {
enum class Op : std::uint8_t {
    kBindTexture,
    kBlendFunc,
    kClear,
    kClearColor,
    kClearDepth,
    kCullFace,
    kDeleteTextures,
    kDepthFunc,
    kDepthMask,
    kGenVertexArrays,
    kBindBuffer,
    kBindVertexArray,
    kBufferData,
    kBufferSubData,
    kDeleteFramebuffers,
    kBindFramebuffer,
    kUnmapBuffer,
    kLinkProgram,
    kUseProgram,
    kCreateProgram,
    kIsShader,
    kGetShaderiv,
    kCreateShader,
    kDrawBuffers,
    kShaderSource,
    kCompileShader,
    kGetUniformLocation,
    kUniform1i,
    kUniform3f,
    kUniformMatrix4fv,
    kUniform1f,
    kUniform2f,
    kUniform3fv,
    kReadPixels,
    kBindAttribLocation,
    kBindFragDataLocation,
    kAttachShader,
    kActiveTexture,
    kCheckFramebufferStatus,
    kFramebufferTexture2D,
    kGenFramebuffers,
    kEnableVertexAttribArray,
    kVertexAttribPointer,
    kDeleteVertexArrays,
    kDeleteBuffers,
    kDisable,
    kDrawArrays,
    kEnable,
    kGenBuffers,
    kGenerateMipmap,
    kGenTextures,
    kGetFloatv,
    kMapBufferRange,
    kDrawRangeElements,
    kTexImage2D,
    kTexParameterf,
    kTexParameteri,
    kTexSubImage2D,
    kViewport,
    kCount
};

const char *OpName(Op op);

struct Call {
    Op op;
    std::uint8_t nargs = 0;
    bool has_ret = false;
    std::uint64_t args[10] = {};
    std::uint64_t ret = 0;
    std::vector<std::uint8_t> item;
    bool has_item = false;
};

class Recorder {
public:
    std::vector<Call> calls;
    std::function<void(Op)> before;
    void Clear() { calls.clear(); }
};

class NameAllocator {
public:
    std::uint32_t Alloc();
    void Free(std::uint32_t name);
    bool Used(std::uint32_t name) const { return name < used_.size() && used_[name]; }

private:
    std::vector<bool> used_{true};
};

class GlStream {
public:
    GlStream(gl::Context *context, Recorder *recorder);
    ~GlStream();

    gl::Context *context() const { return context_; }
    void SetRecorder(Recorder *recorder) { recorder_ = recorder; }
    std::uint32_t ContextTexture(std::uint32_t name) const { return Ctx(ctx_textures_, name); }

    void GetFloatv(std::uint32_t pname, float *out);

    std::uint32_t CreateShader(std::uint32_t type);
    void ShaderSource(std::uint32_t shader, const std::string &file);
    void CompileShader(std::uint32_t shader);
    int GetShaderiv(std::uint32_t shader, std::uint32_t pname);
    bool IsShader(std::uint32_t shader);
    std::uint32_t CreateProgram(gl::ProgramKind kind);
    void AttachShader(std::uint32_t program, std::uint32_t shader);
    void BindAttribLocation(std::uint32_t program, std::uint32_t index, const std::string &name);
    void BindFragDataLocation(std::uint32_t program, std::uint32_t color, const std::string &name);
    void LinkProgram(std::uint32_t program);
    void UseProgram(std::uint32_t program);
    int GetUniformLocation(std::uint32_t program, const std::string &name);
    void Uniform1i(int loc, std::int32_t v);
    void Uniform1f(int loc, float v);
    void Uniform2f(int loc, float x, float y);
    void Uniform3f(int loc, float x, float y, float z);
    void Uniform3fv(int loc, int count, const float *v);
    void UniformMatrix4fv(int loc, int count, bool transpose, const float *v);

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

    std::uint32_t GenBuffer();
    void DeleteBuffer(std::uint32_t name);
    void BindBuffer(std::uint32_t target, std::uint32_t name);
    void BufferData(std::uint32_t target, std::size_t size, const void *data, std::uint32_t usage);
    void BufferSubData(std::uint32_t target, std::size_t offset, std::size_t size,
                       const void *data);
    void MapCopyUnmap(std::uint32_t target, std::size_t offset, std::size_t length,
                      std::uint32_t access, const void *data);
    std::uint32_t GenVertexArray();
    void DeleteVertexArray(std::uint32_t name);
    void BindVertexArray(std::uint32_t name);
    void EnableVertexAttribArray(std::uint32_t index);
    void VertexAttribPointer(std::uint32_t index, int size, std::uint32_t type, bool normalized,
                             int stride, std::size_t offset);

    std::uint32_t GenFramebuffer();
    void DeleteFramebuffer(std::uint32_t name);
    void BindFramebuffer(std::uint32_t target, std::uint32_t name);
    void FramebufferTexture2D(std::uint32_t target, std::uint32_t attachment,
                              std::uint32_t textarget, std::uint32_t texture, int level);
    void DrawBuffers(int n, const std::uint32_t *buffers);
    std::uint32_t CheckFramebufferStatus(std::uint32_t target);

    void Enable(std::uint32_t cap);
    void Disable(std::uint32_t cap);
    void CullFace(std::uint32_t mode);
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

private:
    Call &Record(Op op, std::initializer_list<std::uint64_t> args);
    void Item(Call *call, const void *data, std::size_t size);
    std::uint32_t Ctx(const std::unordered_map<std::uint32_t, std::uint32_t> &map,
                      std::uint32_t name) const;

    gl::Context *context_;
    Recorder *recorder_;
    NameAllocator textures_, buffers_, vaos_, fbos_, shader_objects_;
    std::unordered_map<std::uint32_t, std::uint32_t> ctx_textures_, ctx_buffers_, ctx_vaos_,
        ctx_fbos_, ctx_programs_;
    std::unordered_map<std::uint32_t, std::unique_ptr<gl::Program>> programs_;
    std::unordered_map<std::uint32_t, std::uint32_t> program_kind_;
};

std::size_t UploadBytes(int width, int height, std::uint32_t format, std::uint32_t type);

}

#endif
