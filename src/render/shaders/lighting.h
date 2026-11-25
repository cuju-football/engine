// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// GRF's lighting pass, lighting.vert and lighting.frag: one light added to the accumulation
// buffer, with a nine-tap shadow lookup, transcribed from llvmpipe's JIT code for it (fs158 with
// its vertex function and the position-only setup variant it shares).

#ifndef CUJU_RENDER_SHADERS_LIGHTING_H
#define CUJU_RENDER_SHADERS_LIGHTING_H

#include "../gl/program.h"

namespace cuju::render::shaders {
class LightingProgram : public gl::Program {
public:
    static constexpr int kModelMatrix = 0;
    static constexpr int kViewMatrix = 1;
    static constexpr int kProjectionMatrix = 2;
    static constexpr int kMapAlbedo = 3;
    static constexpr int kMapNormal = 4;
    static constexpr int kMapDepth = 5;
    static constexpr int kMapShadow = 6;
    static constexpr int kInverseProjectionViewMatrix = 7;
    static constexpr int kLightViewProjectionMatrix = 8;
    static constexpr int kContextWidth = 9;
    static constexpr int kContextHeight = 10;
    static constexpr int kContextX = 11;
    static constexpr int kContextY = 12;
    static constexpr int kHasShadow = 13;
    static constexpr int kCameraPosition = 14;
    static constexpr int kLightPosition = 15;
    static constexpr int kLightColor = 16;
    static constexpr int kLightRadius = 17;

    LightingProgram();
    const char *Name() const override { return "lighting"; }
    void RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                     float *clip, float *inputs) const override;
    void ShadeBlock(const gl::FragmentBlock &block) const override;
};

}

#endif
