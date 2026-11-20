// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// GRF's ambient pass, ambient.vert and ambient.frag: base colour, screen-space ambient occlusion
// with 32 kernel taps and the edge detection for antialiasing, transcribed from llvmpipe's JIT
// code for it (fs159 with its vertex function and setup variant).

#ifndef CUJU_RENDER_SHADERS_AMBIENT_H
#define CUJU_RENDER_SHADERS_AMBIENT_H

#include "../gl/program.h"

namespace cuju::render::shaders {

class AmbientProgram : public gl::Program {
public:
    // Uniform locations, as GL assigns them.
    static constexpr int kOrthoViewMatrix = 0;
    static constexpr int kOrthoProjectionMatrix = 1;
    static constexpr int kMapAlbedo = 2;
    static constexpr int kMapNormal = 3;
    static constexpr int kMapDepth = 4;
    static constexpr int kMapAux = 5;
    static constexpr int kMapNoise = 6;
    static constexpr int kViewMatrix = 7;
    static constexpr int kProjectionMatrix = 8;
    static constexpr int kInverseProjectionViewMatrix = 9;
    static constexpr int kContextWidth = 10;
    static constexpr int kContextHeight = 11;
    static constexpr int kContextX = 12;
    static constexpr int kContextY = 13;
    static constexpr int kCameraClip = 14;
    static constexpr int kSsaoKernel = 15;  // 32 locations
    static constexpr int kKernelSize = 32;

    AmbientProgram();
    const char *Name() const override { return "ambient"; }
    void RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                     float *clip, float *inputs) const override;
    void ShadeBlock(const gl::FragmentBlock &block) const override;
};

}  // namespace cuju::render::shaders

#endif  // CUJU_RENDER_SHADERS_AMBIENT_H
