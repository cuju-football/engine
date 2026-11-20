// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// GRF's shadow map program, zphase.vert and zphase.frag, transcribed from llvmpipe's JIT code for
// it (fs160 with its vertex function and setup variant).

#ifndef CUJU_RENDER_SHADERS_ZPHASE_H
#define CUJU_RENDER_SHADERS_ZPHASE_H

#include "../gl/program.h"

namespace cuju::render::shaders {

class ZphaseProgram : public gl::Program {
public:
    // Uniform locations, as GL assigns them.
    static constexpr int kModelMatrix = 0;
    static constexpr int kViewMatrix = 1;
    static constexpr int kProjectionMatrix = 2;

    ZphaseProgram();
    const char *Name() const override { return "zphase"; }
    void RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                     float *clip, float *inputs) const override;
    void ShadeBlock(const gl::FragmentBlock &block) const override;
};

}  // namespace cuju::render::shaders

#endif  // CUJU_RENDER_SHADERS_ZPHASE_H
