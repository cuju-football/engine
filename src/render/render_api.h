// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The C API of libcuju_render, the software renderer of GRF's frames. One renderer stands for one
// rendering GRF engine, a GameEnv, and its calls follow where GameEnv touches graphics. Each render
// takes a RenderState, include/cuju/render/render_state.h, passed as bytes with its size; the
// state's version must equal cuju_render_state_version(). Renderers of one process share the media
// of one path. A renderer is used by one thread at a time. Every function returns a
// cuju_render_status, and cuju_render_last_error names the last failure of the calling thread.

#ifndef CUJU_RENDER_API_H
#define CUJU_RENDER_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CUJU_RENDER_EXPORT
#else
#define CUJU_RENDER_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CUJU_RENDER_API_VERSION 2

typedef struct cuju_renderer cuju_renderer;

enum cuju_render_status {
    CUJU_RENDER_OK = 0,
    CUJU_RENDER_ERROR_ARGUMENT = 1,  // a null pointer, a size or a dimension out of range
    CUJU_RENDER_ERROR_MEDIA = 2,     // the media path lacks a file or holds a broken one
    CUJU_RENDER_ERROR_STATE = 3,     // a RenderState of another size or version, no match, or a
                                     // render at which GRF's process crashes
    CUJU_RENDER_ERROR_INTERNAL = 4   // anything else, such as memory
};

CUJU_RENDER_EXPORT uint32_t cuju_render_api_version(void);
CUJU_RENDER_EXPORT uint32_t cuju_render_state_version(void);
CUJU_RENDER_EXPORT size_t cuju_render_state_size(void);
CUJU_RENDER_EXPORT const char *cuju_render_last_error(void);

// A renderer of `width` by `height` pixels, the width a multiple of 4, drawing with `threads`
// threads, over the media at `media_path`, the folder of cuju-grf-media that holds data/ and
// fonts/.
CUJU_RENDER_EXPORT int cuju_render_create(int width, int height, int threads,
                                          const char *media_path, cuju_renderer **out);
CUJU_RENDER_EXPORT void cuju_render_destroy(cuju_renderer *renderer);

// GameEnv::reset, in GRF's order: reset_begin, stop_match, loading_page on the first reset of the
// engine, render_scene(1) when rendering, then start_match.
CUJU_RENDER_EXPORT int cuju_render_reset_begin(cuju_renderer *renderer);
CUJU_RENDER_EXPORT int cuju_render_stop_match(cuju_renderer *renderer);
CUJU_RENDER_EXPORT int cuju_render_loading_page(cuju_renderer *renderer);
CUJU_RENDER_EXPORT int cuju_render_render_scene(cuju_renderer *renderer, int swap);
// GameTask::StartMatch, with the state after ResetRenderState and game_engine_random_seed.
CUJU_RENDER_EXPORT int cuju_render_start_match(cuju_renderer *renderer, const void *state,
                                               size_t size, uint32_t seed);
// GameEnv::ProcessState of the rendering match, at each get_state and set_state.
CUJU_RENDER_EXPORT int cuju_render_state_processed(cuju_renderer *renderer);
// GRF's game_config.render, nonzero at first, which start_match and the captions read. A match
// started with it zero is GRF's match without rendering, on the pitch alone.
CUJU_RENDER_EXPORT int cuju_render_set_render(cuju_renderer *renderer, int render);
// GameEnv::render(swap): the frame's state, read back when `swap` is nonzero.
CUJU_RENDER_EXPORT int cuju_render_render(cuju_renderer *renderer, const void *state, size_t size,
                                          int swap);

// The last frame read back, width * height * 3 bytes of RGB, rows from the bottom as glReadPixels
// gives them, or from the top as GRF's observation holds them when `top_down` is nonzero.
CUJU_RENDER_EXPORT int cuju_render_frame(cuju_renderer *renderer, int top_down, uint8_t *out,
                                         size_t size);
// GRF's pixels and pixels_gray of the last frame: OpenCV's INTER_AREA resize of the top-down
// frame to `width` by `height`, RGB, and of its RGB2GRAY, one byte per pixel.
CUJU_RENDER_EXPORT int cuju_render_pixels(cuju_renderer *renderer, int width, int height,
                                          uint8_t *out, size_t size);
CUJU_RENDER_EXPORT int cuju_render_pixels_gray(cuju_renderer *renderer, int width, int height,
                                               uint8_t *out, size_t size);
// The SHA-256 of the renderer's fixed data, one `name digest` line each: the glyph atlas, the
// twelve pitch textures, the SSAO kernel, and each level of every mipmapped texture the GL context
// holds now, as glGetTexImage reads it, named mip_<GRF's texture name>_<level>. Writes
// at most `size` bytes with the terminating zero and returns the length of the whole text.
CUJU_RENDER_EXPORT size_t cuju_render_digests(cuju_renderer *renderer, char *out, size_t size);
// Whether a frame was read back since the renderer was made.
CUJU_RENDER_EXPORT int cuju_render_has_frame(cuju_renderer *renderer, int *out);

#ifdef __cplusplus
}
#endif

#endif  // CUJU_RENDER_API_H
