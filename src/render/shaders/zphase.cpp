// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// Ports GRF's zphase shaders (Apache-2.0) as llvmpipe's JIT compiles them (Mesa, MIT licence).

#include "zphase.h"

#include "../fpenv.h"
#include "../gl/fragment.h"

namespace cuju::render::shaders {

ZphaseProgram::ZphaseProgram()
    : gl::Program({{"modelMatrix", gl::kMat4, 1},
                   {"viewMatrix", gl::kMat4, 1},
                   {"projectionMatrix", gl::kMat4, 1}},
                  {{"position", 0}}, {}) {}

// gl_Position = projectionMatrix * viewMatrix * modelMatrix * position, which NIR evaluates as
// (P * V) * (M * pos): the uniform-only product per call, element (j, r) the column-major
// ((P0r * Vj0 + P1r * Vj1) + P2r * Vj2) + P3r * Vj3, then M * pos with w = 1 folded away and the
// product applied to it, all separate roundings under FTZ and DAZ.
void ZphaseProgram::RunVertices(const gl::DrawState &state, int count, const float *const *attribs,
                                float *clip, float *) const {
    const float *M = state.UniformPtr(kModelMatrix);
    const float *V = state.UniformPtr(kViewMatrix);
    const float *P = state.UniformPtr(kProjectionMatrix);
    float C[16];
    for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < 4; ++r) {
            float s = ftz::Mul(P[0 * 4 + r], V[j * 4 + 0]);
            s = ftz::Add(s, ftz::Mul(P[1 * 4 + r], V[j * 4 + 1]));
            s = ftz::Add(s, ftz::Mul(P[2 * 4 + r], V[j * 4 + 2]));
            s = ftz::Add(s, ftz::Mul(P[3 * 4 + r], V[j * 4 + 3]));
            C[j * 4 + r] = s;
        }
    }
    const float *pos = attribs[0];
    for (int i = 0; i < count; ++i) {
        const float x = pos[4 * i + 0], y = pos[4 * i + 1], z = pos[4 * i + 2];
        float m[4];
        for (int r = 0; r < 4; ++r) {
            float s = ftz::Mul(M[0 * 4 + r], x);
            s = ftz::Add(s, ftz::Mul(M[1 * 4 + r], y));
            s = ftz::Add(s, ftz::Mul(M[2 * 4 + r], z));
            m[r] = ftz::Add(s, M[3 * 4 + r]);
        }
        for (int r = 0; r < 4; ++r) {
            float s = ftz::Mul(C[0 * 4 + r], m[0]);
            s = ftz::Add(s, ftz::Mul(C[1 * 4 + r], m[1]));
            s = ftz::Add(s, ftz::Mul(C[2 * 4 + r], m[2]));
            s = ftz::Add(s, ftz::Mul(C[3 * 4 + r], m[3]));
            clip[4 * i + r] = s;
        }
    }
}

// fs160: depth from the plane, D16 through the +128 trick, LESS on the stored value, masked store.
void ZphaseProgram::ShadeBlock(const gl::FragmentBlock &b) const {
    float X[16], Y[16], z[16];
    std::uint16_t d16[16];
    gl::LaneCoordinates(b, X, Y);
    gl::InterpolateDepth(b, X, Y, z);
    gl::DepthToD16(z, d16);
    const std::uint16_t lanes = gl::LaneMaskFromCoverage(b.coverage);
    const std::uint16_t pass = gl::DepthTest(b, d16, lanes);
    gl::DepthWrite(b, d16, pass);
}

CUJU_GL_REGISTER_PROGRAM(kZphase, ZphaseProgram);

}  // namespace cuju::render::shaders
