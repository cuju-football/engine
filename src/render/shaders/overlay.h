// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// GRF's overlay program, overlay.vert and overlay.frag, transcribed from llvmpipe's JIT code for it
// (fs162 with its vertex function and setup variant): one RGBA8 texture, linear, clamp to edge,
// anisotropic, blended SRC_ALPHA, ONE_MINUS_SRC_ALPHA in 16-bit fixed point onto the 10-bit
// default framebuffer.

#ifndef CUJU_RENDER_SHADERS_OVERLAY_H
#define CUJU_RENDER_SHADERS_OVERLAY_H

#include "../gl/program.h"

namespace cuju::render::shaders {
class OverlayProgram : public gl::Program {
public:
    static constexpr int kModel = 0;
    static constexpr int kProjection = 1;
    static constexpr int kOurTexture = 2;

    OverlayProgram();
    const char *Name() const override { return "overlay"; }
    void RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                     float *clip, float *inputs) const override;
    void ShadeBlock(const gl::FragmentBlock &block) const override;
};

}

#endif
