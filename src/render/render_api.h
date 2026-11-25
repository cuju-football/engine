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
    CUJU_RENDER_ERROR_ARGUMENT = 1,
    CUJU_RENDER_ERROR_MEDIA = 2,
    CUJU_RENDER_ERROR_STATE = 3,
    CUJU_RENDER_ERROR_INTERNAL = 4
};

CUJU_RENDER_EXPORT uint32_t cuju_render_api_version(void);
CUJU_RENDER_EXPORT uint32_t cuju_render_state_version(void);
CUJU_RENDER_EXPORT size_t cuju_render_state_size(void);
CUJU_RENDER_EXPORT const char *cuju_render_last_error(void);

CUJU_RENDER_EXPORT int cuju_render_create(int width, int height, int threads,
                                          const char *media_path, cuju_renderer **out);
CUJU_RENDER_EXPORT void cuju_render_destroy(cuju_renderer *renderer);

CUJU_RENDER_EXPORT int cuju_render_reset_begin(cuju_renderer *renderer);
CUJU_RENDER_EXPORT int cuju_render_stop_match(cuju_renderer *renderer);
CUJU_RENDER_EXPORT int cuju_render_loading_page(cuju_renderer *renderer);
CUJU_RENDER_EXPORT int cuju_render_render_scene(cuju_renderer *renderer, int swap);
CUJU_RENDER_EXPORT int cuju_render_start_match(cuju_renderer *renderer, const void *state,
                                               size_t size, uint32_t seed);
CUJU_RENDER_EXPORT int cuju_render_state_processed(cuju_renderer *renderer);
// GRF's game_config.render, nonzero at first, which start_match and the captions read.
CUJU_RENDER_EXPORT int cuju_render_set_render(cuju_renderer *renderer, int render);
CUJU_RENDER_EXPORT int cuju_render_render(cuju_renderer *renderer, const void *state, size_t size,
                                          int swap);

CUJU_RENDER_EXPORT int cuju_render_frame(cuju_renderer *renderer, int top_down, uint8_t *out,
                                         size_t size);
// GRF's pixels and pixels_gray of the last frame: OpenCV's INTER_AREA resize of the top-down frame
// to `width` by `height`, RGB, and of its RGB2GRAY, one byte per pixel.
CUJU_RENDER_EXPORT int cuju_render_pixels(cuju_renderer *renderer, int width, int height,
                                          uint8_t *out, size_t size);
CUJU_RENDER_EXPORT int cuju_render_pixels_gray(cuju_renderer *renderer, int width, int height,
                                               uint8_t *out, size_t size);
CUJU_RENDER_EXPORT size_t cuju_render_digests(cuju_renderer *renderer, char *out, size_t size);
CUJU_RENDER_EXPORT int cuju_render_has_frame(cuju_renderer *renderer, int *out);

#ifdef __cplusplus
}
#endif

#endif
