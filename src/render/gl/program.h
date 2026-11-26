// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The program interface of the mini GL. Each of GRF's shader programs is a hand-written class with
// a vertex stage and a fragment stage, transcribed from llvmpipe's JIT code. The context hands the
// stages a DrawState, an immutable snapshot of everything a draw call reads.

#ifndef CUJU_RENDER_GL_PROGRAM_H
#define CUJU_RENDER_GL_PROGRAM_H

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../texture.h"

namespace cuju::render::gl {
struct FragmentBlock;

constexpr int kMaxColorBuffers = 4;
constexpr int kMaxInputs = 8;
constexpr int kMaxSlots = kMaxInputs + 1;
constexpr int kMaxAttributes = 8;
constexpr int kMaxTextureUnits = 16;

enum UniformType : std::uint32_t {
    kFloat = 0x1406,
    kVec2 = 0x8B50,
    kVec3 = 0x8B51,
    kVec4 = 0x8B52,
    kInt = 0x1404,
    kBool = 0x8B56,
    kMat4 = 0x8B5C,
    kSampler2D = 0x8B5E,
    kSampler2DShadow = 0x8B62,
};

int UniformComponents(std::uint32_t type);

struct UniformDecl {
    const char *name;
    std::uint32_t type;
    int size;
};

struct AttributeDecl {
    const char *name;
    int location;
};

enum class Interp : std::uint8_t { kPerspective, kLinear };

struct FragmentOps {
    bool depth_test = false;
    std::uint32_t depth_func = 0x0201;
    bool depth_write = true;
    bool blend = false;
    std::uint32_t blend_src = 1;
    std::uint32_t blend_dst = 0;
};

struct Surface {
    Format format = Format::R8G8B8A8_UNORM;
    int width = 0;
    int height = 0;
    std::uint8_t *data = nullptr;
    int stride = 0;
};

struct TargetInfo {
    bool window = false;
    int width = 0;
    int height = 0;
    int num_color = 0;
    Surface color[kMaxColorBuffers];
    Surface depth;
};

class Program;

struct DrawState {
    const Program *program = nullptr;
    std::vector<std::uint32_t> uniforms;
    std::array<const Texture *, kMaxTextureUnits> textures{};
    FragmentOps ops;
    TargetInfo target;

    std::uint32_t UniformBits(int loc, int c = 0) const;
    float Uniform(int loc, int c = 0) const;
    const float *UniformPtr(int loc) const;
    std::int32_t UniformInt(int loc) const { return static_cast<std::int32_t>(UniformBits(loc)); }
    const Texture *SamplerTexture(int loc) const;
};

class Program {
public:
    Program(std::vector<UniformDecl> uniforms, std::vector<AttributeDecl> attributes,
            std::vector<Interp> inputs);
    virtual ~Program() = default;

    virtual const char *Name() const = 0;

    const std::vector<UniformDecl> &Uniforms() const { return uniforms_; }
    const std::vector<AttributeDecl> &Attributes() const { return attributes_; }
    int NumInputs() const { return static_cast<int>(inputs_.size()); }
    Interp InputInterp(int i) const { return inputs_[i]; }

    int NumLocations() const { return static_cast<int>(location_offset_.size()); }
    int UniformLocation(const std::string &name) const;
    int LocationOffset(int loc) const { return location_offset_[loc]; }
    int LocationComponents(int loc) const { return location_components_[loc]; }
    int StorageSize() const { return storage_size_; }

    virtual void RunVertices(const DrawState &state, int count, const float *const *attribs,
                             float *clip, float *inputs) const = 0;

    virtual void ShadeBlock(const FragmentBlock &block) const = 0;

private:
    std::vector<UniformDecl> uniforms_;
    std::vector<AttributeDecl> attributes_;
    std::vector<Interp> inputs_;
    std::vector<int> location_offset_;
    std::vector<int> location_components_;
    std::vector<int> location_decl_;
    int storage_size_ = 0;
};

// GRF's programs, as the replayer and the scene port create them.
enum class ProgramKind : std::uint8_t {
    kZphase,
    kSimple,
    kAmbient,
    kLighting,
    kPostprocess,
    kOverlay,
};

std::unique_ptr<Program> MakeProgram(ProgramKind kind);

using ProgramFactory = std::unique_ptr<Program> (*)();
bool RegisterProgram(ProgramKind kind, ProgramFactory factory);

#define CUJU_GL_REGISTER_PROGRAM(kind, Class)                                                      \
    static const bool cuju_gl_registered_##Class =                                                 \
        ::cuju::render::gl::RegisterProgram(::cuju::render::gl::ProgramKind::kind,                 \
                                            []() -> std::unique_ptr<::cuju::render::gl::Program> { \
                                                return std::make_unique<Class>();                  \
                                            })

}  // namespace cuju::render::gl

#endif
