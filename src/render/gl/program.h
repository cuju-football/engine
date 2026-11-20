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

// Most color attachments of a framebuffer, the G-buffer has three.
constexpr int kMaxColorBuffers = 4;
// Most fragment inputs after the position, which is setup slot 0.
constexpr int kMaxInputs = 8;
constexpr int kMaxSlots = kMaxInputs + 1;
// Most vertex attribute locations.
constexpr int kMaxAttributes = 8;
// Texture units.
constexpr int kMaxTextureUnits = 16;

// GL type enums of uniforms, as the trace's PROGRAM records report them.
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

// 32-bit components of one location of a uniform type.
int UniformComponents(std::uint32_t type);

// A uniform as GL reports it. Arrays take `size` consecutive locations, one per element.
struct UniformDecl {
    const char *name;
    std::uint32_t type;
    int size;
};

// A vertex attribute: its GL location and how many components the shader reads.
struct AttributeDecl {
    const char *name;
    int location;
};

// Interpolation of a fragment input. Perspective inputs are multiplied by 1/w in setup and
// divided by the interpolated 1/w per pixel, linear ones are planes of the raw value.
enum class Interp : std::uint8_t { kPerspective, kLinear };

// Fixed-function state that the fragment stage applies itself, as llvmpipe compiles it into the
// fragment shader variant.
struct FragmentOps {
    bool depth_test = false;
    std::uint32_t depth_func = 0x0201;  // GL_LESS
    bool depth_write = true;
    bool blend = false;
    std::uint32_t blend_src = 1;  // GL_ONE
    std::uint32_t blend_dst = 0;  // GL_ZERO
};

// A render target as the fragment stage sees it.
struct Surface {
    Format format = Format::R8G8B8A8_UNORM;
    int width = 0;
    int height = 0;
    std::uint8_t *data = nullptr;  // memory row 0 first
    int stride = 0;                // bytes per row
};

// What the current framebuffer is.
struct TargetInfo {
    bool window = false;  // the default framebuffer, Y_0_TOP, memory row 0 is the top
    int width = 0;
    int height = 0;
    int num_color = 0;
    Surface color[kMaxColorBuffers];
    Surface depth;  // Z16_UNORM or none (data null)
};

class Program;

// The immutable snapshot of a draw call: program, uniform values, textures per unit, fragment
// operations and target. Triangles keep a pointer to it until they are rasterized.
struct DrawState {
    const Program *program = nullptr;
    std::vector<std::uint32_t> uniforms;  // raw bits, laid out by Program::UniformOffset
    std::array<const Texture *, kMaxTextureUnits> textures{};
    FragmentOps ops;
    TargetInfo target;

    // The raw bits and float value of component `c` of uniform location `loc`.
    std::uint32_t UniformBits(int loc, int c = 0) const;
    float Uniform(int loc, int c = 0) const;
    const float *UniformPtr(int loc) const;
    std::int32_t UniformInt(int loc) const { return static_cast<std::int32_t>(UniformBits(loc)); }
    // The texture bound to the unit that sampler uniform `loc` selects, or null.
    const Texture *SamplerTexture(int loc) const;
};

// One of GRF's programs. Locations are assigned in the order of Uniforms(), each array element
// taking its own location, exactly as Mesa assigned them for GRF's shaders (the replayer checks
// every glGetUniformLocation result against the trace).
class Program {
public:
    Program(std::vector<UniformDecl> uniforms, std::vector<AttributeDecl> attributes,
            std::vector<Interp> inputs);
    virtual ~Program() = default;

    virtual const char *Name() const = 0;

    const std::vector<UniformDecl> &Uniforms() const { return uniforms_; }
    const std::vector<AttributeDecl> &Attributes() const { return attributes_; }
    // Fragment inputs, setup slots 1 to NumInputs().
    int NumInputs() const { return static_cast<int>(inputs_.size()); }
    Interp InputInterp(int i) const { return inputs_[i]; }

    // Locations and storage.
    int NumLocations() const { return static_cast<int>(location_offset_.size()); }
    int UniformLocation(const std::string &name) const;  // -1 when absent
    int LocationOffset(int loc) const { return location_offset_[loc]; }
    int LocationComponents(int loc) const { return location_components_[loc]; }
    int StorageSize() const { return storage_size_; }

    // The vertex stage for `count` vertices, under FTZ and DAZ. attribs[l] holds count * 4 floats
    // for attribute location l (fetched as llvmpipe does, w = 1 for three-component formats), or is
    // null when the program does not read location l. Writes count * 4 floats of clip position and
    // count * NumInputs() * 4 floats of fragment inputs, vertex-major.
    virtual void RunVertices(const DrawState &state, int count, const float *const *attribs,
                             float *clip, float *inputs) const = 0;

    // The fragment stage for one 4x4 block, under FTZ and DAZ. It applies the depth test, writes
    // and blends itself through the helpers of fragment.h.
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
    kZphase,       // shadow map, fs160
    kSimple,       // G-buffer, fs157
    kAmbient,      // fs159
    kLighting,     // fs158
    kPostprocess,  // fs161
    kOverlay,      // fs162
};

// A new program of that kind, or null while its class is not written yet.
std::unique_ptr<Program> MakeProgram(ProgramKind kind);

// Each program's source file registers its class once, at namespace scope:
//   CUJU_GL_REGISTER_PROGRAM(kZphase, ZphaseProgram)
using ProgramFactory = std::unique_ptr<Program> (*)();
bool RegisterProgram(ProgramKind kind, ProgramFactory factory);

#define CUJU_GL_REGISTER_PROGRAM(kind, Class)                                                      \
    static const bool cuju_gl_registered_##Class =                                                 \
        ::cuju::render::gl::RegisterProgram(::cuju::render::gl::ProgramKind::kind,                 \
                                            []() -> std::unique_ptr<::cuju::render::gl::Program> { \
                                                return std::make_unique<Class>();                  \
                                            })

}  // namespace cuju::render::gl

#endif  // CUJU_RENDER_GL_PROGRAM_H
