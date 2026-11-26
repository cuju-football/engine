// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// glGenerateMipmap as llvmpipe runs it: util_gen_mipmap blits each level from the previous one with
// a LINEAR, CLAMP_TO_EDGE blit through u_blitter. R8G8B8X8_UNORM and R8G8B8A8_UNORM filter on the
// AoS 8-bit path, the sRGB formats on the SoA float path, decoding each texel and encoding the
// result with three chained reciprocal square roots. The algorithms are Mesa's (MIT licence,
// Copyright VMware, Inc. and others): u_gen_mipmap.c, u_blitter.c and the gallivm sampling code.

#ifndef CUJU_RENDER_MIPMAP_H
#define CUJU_RENDER_MIPMAP_H

#include "texture.h"

namespace cuju::render {
int MipLevelCount(int width, int height);

bool CanGenerateMipmaps(Format format, int width, int height);

void GenerateMipmaps(Texture &texture);

}  // namespace cuju::render

#endif
