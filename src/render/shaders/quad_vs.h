// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The vertex stage GRF's ambient and postprocess programs share, ambient.vert and
// postprocess.vert: gl_Position = orthoProjectionMatrix * orthoViewMatrix * position, a vec4
// position. llvmpipe's JIT code (fs159 and fs161 vertex functions) evaluates it as P * (V * pos),
// each column-major product ((m0r * x + m1r * y) + m2r * z) + m3r * w with separate roundings,
// under FTZ and DAZ.

#ifndef CUJU_RENDER_SHADERS_QUAD_VS_H
#define CUJU_RENDER_SHADERS_QUAD_VS_H

#include "../fpenv.h"

namespace cuju::render::shaders {
inline void OrthoQuadVertices(const float *V, const float *P, int count, const float *pos,
                              float *clip) {
    for (int i = 0; i < count; ++i) {
        const float *p = pos + 4 * i;
        float t[4];
        for (int r = 0; r < 4; ++r) {
            float s = ftz::Mul(V[0 * 4 + r], p[0]);
            s = ftz::Add(s, ftz::Mul(V[1 * 4 + r], p[1]));
            s = ftz::Add(s, ftz::Mul(V[2 * 4 + r], p[2]));
            t[r] = ftz::Add(s, ftz::Mul(V[3 * 4 + r], p[3]));
        }
        for (int r = 0; r < 4; ++r) {
            float s = ftz::Mul(P[0 * 4 + r], t[0]);
            s = ftz::Add(s, ftz::Mul(P[1 * 4 + r], t[1]));
            s = ftz::Add(s, ftz::Mul(P[2 * 4 + r], t[2]));
            clip[4 * i + r] = ftz::Add(s, ftz::Mul(P[3 * 4 + r], t[3]));
        }
    }
}

}  // namespace cuju::render::shaders

#endif
