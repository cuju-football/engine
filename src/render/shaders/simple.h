// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// GRF's G-buffer program, simple.vert and simple.frag, transcribed from llvmpipe's JIT code for
// it (fs157 with its vertex function and setup_variant_3).

#ifndef CUJU_RENDER_SHADERS_SIMPLE_H
#define CUJU_RENDER_SHADERS_SIMPLE_H

#include "../gl/program.h"

namespace cuju::render::shaders {
class SimpleProgram : public gl::Program {
public:
    static constexpr int kModelMatrix = 0;
    static constexpr int kViewMatrix = 1;
    static constexpr int kProjectionMatrix = 2;
    static constexpr int kMapAlbedo = 3;
    static constexpr int kMapNormal = 4;
    static constexpr int kMapSpecular = 5;
    static constexpr int kMapIllumination = 6;
    static constexpr int kMaterialParams = 7;
    static constexpr int kMaterialBools = 8;

    SimpleProgram();
    const char *Name() const override { return "simple"; }
    void RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                     float *clip, float *inputs) const override;
    void ShadeBlock(const gl::FragmentBlock &block) const override;
};

}

#endif
