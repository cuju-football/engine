// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include "program.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

#include "../numerics.h"

namespace cuju::render::gl {

int UniformComponents(std::uint32_t type) {
    switch (type) {
        case kVec2:
            return 2;
        case kVec3:
            return 3;
        case kVec4:
            return 4;
        case kMat4:
            return 16;
        default:
            return 1;
    }
}

Program::Program(std::vector<UniformDecl> uniforms, std::vector<AttributeDecl> attributes,
                 std::vector<Interp> inputs)
    : uniforms_(std::move(uniforms)),
      attributes_(std::move(attributes)),
      inputs_(std::move(inputs)) {
    for (int d = 0; d < static_cast<int>(uniforms_.size()); ++d) {
        const int n = UniformComponents(uniforms_[d].type);
        for (int e = 0; e < uniforms_[d].size; ++e) {
            location_offset_.push_back(storage_size_);
            location_components_.push_back(n);
            location_decl_.push_back(d);
            storage_size_ += n;
        }
    }
}

int Program::UniformLocation(const std::string &name) const {
    // GL accepts "a", "a[0]" and "a[k]" for array element k.
    std::string base = name;
    int element = 0;
    const std::size_t bracket = name.find('[');
    if (bracket != std::string::npos) {
        base = name.substr(0, bracket);
        element = std::atoi(name.c_str() + bracket + 1);
    }
    int loc = 0;
    for (const UniformDecl &u : uniforms_) {
        std::string n = u.name;
        const std::size_t b = n.find('[');
        if (b != std::string::npos) {
            n = n.substr(0, b);
        }
        if (n == base) {
            return element < u.size ? loc + element : -1;
        }
        loc += u.size;
    }
    return -1;
}

std::uint32_t DrawState::UniformBits(int loc, int c) const {
    return uniforms[program->LocationOffset(loc) + c];
}

float DrawState::Uniform(int loc, int c) const { return FromBits(UniformBits(loc, c)); }

const float *DrawState::UniformPtr(int loc) const {
    return reinterpret_cast<const float *>(uniforms.data() + program->LocationOffset(loc));
}

const Texture *DrawState::SamplerTexture(int loc) const {
    const std::int32_t unit = UniformInt(loc);
    if (unit < 0 || unit >= kMaxTextureUnits) {
        return nullptr;
    }
    return textures[unit];
}

namespace {

struct Registry {
    std::mutex mutex;
    ProgramFactory factories[8] = {};
};

Registry &GetRegistry() {
    static Registry registry;
    return registry;
}

}  // namespace

bool RegisterProgram(ProgramKind kind, ProgramFactory factory) {
    Registry &r = GetRegistry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.factories[static_cast<int>(kind)] = factory;
    return true;
}

std::unique_ptr<Program> MakeProgram(ProgramKind kind) {
    Registry &r = GetRegistry();
    ProgramFactory f;
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        f = r.factories[static_cast<int>(kind)];
    }
    return f ? f() : nullptr;
}

}  // namespace cuju::render::gl
