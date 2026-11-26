// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// GRF's postprocess program, postprocess.vert and postprocess.frag, transcribed from llvmpipe's JIT
// code for it (fs161 with its vertex function): edge blur, SSAO blur, fog, the sky fill, grading
// and the store to the 10-bit default framebuffer.

#ifndef CUJU_RENDER_SHADERS_POSTPROCESS_H
#define CUJU_RENDER_SHADERS_POSTPROCESS_H

#include "../gl/program.h"

namespace cuju::render::shaders {
class PostprocessProgram : public gl::Program {
public:
    static constexpr int kOrthoViewMatrix = 0;
    static constexpr int kOrthoProjectionMatrix = 1;
    static constexpr int kMapAccumulation = 2;
    static constexpr int kMapModifier = 3;
    static constexpr int kMapDepth = 4;
    static constexpr int kContextWidth = 5;
    static constexpr int kContextHeight = 6;
    static constexpr int kContextX = 7;
    static constexpr int kContextY = 8;
    static constexpr int kCameraClip = 9;
    static constexpr int kFogScale = 10;

    PostprocessProgram();
    const char *Name() const override { return "postprocess"; }
    void RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                     float *clip, float *inputs) const override;
    void ShadeBlock(const gl::FragmentBlock &block) const override;
};

}  // namespace cuju::render::shaders

#endif
