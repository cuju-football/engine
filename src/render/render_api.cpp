// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include "render_api.h"

#include <array>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "fpenv.h"
#include "frame.h"
#include "assets/text.h"
#include "pixels.h"
#include "sha256.h"

namespace {
using cuju::render::AreaResize;
using cuju::render::Renderer;
using cuju::render::RendererOptions;
using cuju::render::RenderState;
using cuju::render::SharedMedia;

thread_local std::string last_error;

int Fail(int status, const std::string &message) {
    last_error = message;
    return status;
}

std::mutex media_mutex;
std::map<std::string, std::weak_ptr<SharedMedia>> media_by_path;

std::shared_ptr<SharedMedia> MediaFor(const std::string &path, int threads) {
    std::lock_guard<std::mutex> lock(media_mutex);
    std::shared_ptr<SharedMedia> media = media_by_path[path].lock();
    if (!media) {
        media = std::make_shared<SharedMedia>(path, threads);
        media_by_path[path] = media;
    }
    return media;
}

bool CheckState(const void *state, size_t size, std::string *error) {
    if (!state) {
        *error = "the RenderState is null";
        return false;
    }
    if (size != sizeof(RenderState)) {
        *error = "a RenderState of " + std::to_string(size) + " bytes, this renderer reads " +
                 std::to_string(sizeof(RenderState));
        return false;
    }
    std::uint32_t version;
    std::memcpy(&version, state, sizeof version);
    if (version != cuju::render::RENDER_STATE_VERSION) {
        *error = "a RenderState of version " + std::to_string(version) +
                 ", this renderer reads version " +
                 std::to_string(cuju::render::RENDER_STATE_VERSION);
        return false;
    }
    return true;
}

}

struct cuju_renderer {
    std::unique_ptr<Renderer> renderer;
    std::shared_ptr<SharedMedia> media;
    RenderState state;
    bool has_frame = false;
    std::vector<std::uint8_t> top_down;
    std::map<std::array<int, 3>, std::unique_ptr<AreaResize>> resizers;
};

namespace {
template <typename F>
int Guard(cuju_renderer *r, F &&body) {
    if (!r) return Fail(CUJU_RENDER_ERROR_ARGUMENT, "the renderer is null");
    cuju::render::FpScope fp(cuju::render::FpMode::kIeee);
    try {
        int status = body();
        if (status == CUJU_RENDER_OK && !r->media->ok())
            return Fail(CUJU_RENDER_ERROR_MEDIA, r->media->error());
        return status;
    } catch (const cuju::render::scene::GrfCrash &e) {
        return Fail(CUJU_RENDER_ERROR_STATE, e.what());
    } catch (const std::bad_alloc &) {
        return Fail(CUJU_RENDER_ERROR_INTERNAL, "out of memory");
    } catch (const std::exception &e) {
        return Fail(CUJU_RENDER_ERROR_INTERNAL, e.what());
    }
}

const std::vector<std::uint8_t> &TopDown(cuju_renderer *r) {
    const std::vector<std::uint8_t> &screen = r->renderer->screen();
    const std::size_t row = static_cast<std::size_t>(r->renderer->width()) * 3;
    const int h = r->renderer->height();
    r->top_down.resize(screen.size());
    for (int y = 0; y < h; y++)
        std::memcpy(&r->top_down[static_cast<std::size_t>(y) * row],
                    &screen[static_cast<std::size_t>(h - 1 - y) * row], row);
    return r->top_down;
}

int Resize(cuju_renderer *r, int width, int height, bool gray, std::uint8_t *out, size_t size) {
    if (!out || width <= 0 || height <= 0)
        return Fail(CUJU_RENDER_ERROR_ARGUMENT, "pixels need an output and a positive size");
    const int channels = gray ? 1 : 3;
    if (size != static_cast<size_t>(width) * height * channels)
        return Fail(CUJU_RENDER_ERROR_ARGUMENT,
                    "the output's size is not width * height * " + std::to_string(channels));
    if (!r->has_frame) return Fail(CUJU_RENDER_ERROR_STATE, "no frame was read back yet");
    std::unique_ptr<AreaResize> &resize = r->resizers[{width, height, channels}];
    if (!resize)
        resize = std::make_unique<AreaResize>(r->renderer->width(), r->renderer->height(), width,
                                              height, channels);
    const std::vector<std::uint8_t> &frame = TopDown(r);
    const std::ptrdiff_t step = static_cast<std::ptrdiff_t>(r->renderer->width()) * 3;
    if (gray)
        resize->RunGray(frame.data(), step, out, width);
    else
        resize->Run(frame.data(), step, out, static_cast<std::ptrdiff_t>(width) * 3);
    return CUJU_RENDER_OK;
}

}

extern "C" {
uint32_t cuju_render_api_version(void) { return CUJU_RENDER_API_VERSION; }

uint32_t cuju_render_state_version(void) { return cuju::render::RENDER_STATE_VERSION; }

size_t cuju_render_state_size(void) { return sizeof(RenderState); }

const char *cuju_render_last_error(void) { return last_error.c_str(); }

int cuju_render_create(int width, int height, int threads, const char *media_path,
                       cuju_renderer **out) {
    if (!out || !media_path) return Fail(CUJU_RENDER_ERROR_ARGUMENT, "a null argument");
    *out = nullptr;
    // GRF's frames need a width whose rows of three bytes fill whole words of the readback.
    if (width <= 0 || height <= 0 || width % 4 || width > 16384 || height > 16384)
        return Fail(CUJU_RENDER_ERROR_ARGUMENT,
                    "the width must be a positive multiple of 4 and the height positive, both "
                    "up to 16384");
    if (threads < 1 || threads > 256)
        return Fail(CUJU_RENDER_ERROR_ARGUMENT, "threads must lie between 1 and 256");
    cuju::render::FpScope fp(cuju::render::FpMode::kIeee);
    try {
        auto r = std::make_unique<cuju_renderer>();
        r->media = MediaFor(media_path, threads);
        if (!r->media->ok()) return Fail(CUJU_RENDER_ERROR_MEDIA, r->media->error());
        RendererOptions options;
        options.width = width;
        options.height = height;
        options.threads = threads;
        options.media_path = media_path;
        r->renderer = std::make_unique<Renderer>(options, r->media);
        if (!r->media->ok()) return Fail(CUJU_RENDER_ERROR_MEDIA, r->media->error());
        *out = r.release();
        return CUJU_RENDER_OK;
    } catch (const std::bad_alloc &) {
        return Fail(CUJU_RENDER_ERROR_INTERNAL, "out of memory");
    } catch (const std::exception &e) {
        return Fail(CUJU_RENDER_ERROR_INTERNAL, e.what());
    }
}

void cuju_render_destroy(cuju_renderer *renderer) { delete renderer; }

int cuju_render_reset_begin(cuju_renderer *renderer) {
    return Guard(renderer, [&]() -> int {
        renderer->renderer->ResetBegin();
        return CUJU_RENDER_OK;
    });
}

int cuju_render_stop_match(cuju_renderer *renderer) {
    return Guard(renderer, [&]() -> int {
        renderer->renderer->StopMatch();
        return CUJU_RENDER_OK;
    });
}

int cuju_render_loading_page(cuju_renderer *renderer) {
    return Guard(renderer, [&]() -> int {
        renderer->renderer->LoadingPage();
        return CUJU_RENDER_OK;
    });
}

int cuju_render_render_scene(cuju_renderer *renderer, int swap) {
    return Guard(renderer, [&]() -> int {
        renderer->renderer->RenderScene(swap != 0);
        if (swap) renderer->has_frame = true;
        return CUJU_RENDER_OK;
    });
}

int cuju_render_start_match(cuju_renderer *renderer, const void *state, size_t size,
                            uint32_t seed) {
    return Guard(renderer, [&]() -> int {
        std::string error;
        if (!CheckState(state, size, &error)) return Fail(CUJU_RENDER_ERROR_STATE, error);
        if (renderer->renderer->has_match())
            return Fail(CUJU_RENDER_ERROR_STATE, "a match is running; stop it first");
        std::memcpy(&renderer->state, state, size);
        renderer->renderer->StartMatch(renderer->state, seed);
        return CUJU_RENDER_OK;
    });
}

int cuju_render_state_processed(cuju_renderer *renderer) {
    return Guard(renderer, [&]() -> int {
        renderer->renderer->StateProcessed();
        return CUJU_RENDER_OK;
    });
}

int cuju_render_set_render(cuju_renderer *renderer, int render) {
    return Guard(renderer, [&]() -> int {
        renderer->renderer->SetRender(render != 0);
        return CUJU_RENDER_OK;
    });
}

int cuju_render_render(cuju_renderer *renderer, const void *state, size_t size, int swap) {
    return Guard(renderer, [&]() -> int {
        std::string error;
        if (!CheckState(state, size, &error)) return Fail(CUJU_RENDER_ERROR_STATE, error);
        if (!renderer->renderer->has_match())
            return Fail(CUJU_RENDER_ERROR_STATE, "no match to render; start one first");
        std::memcpy(&renderer->state, state, size);
        renderer->renderer->Render(renderer->state, swap != 0);
        if (swap) renderer->has_frame = true;
        return CUJU_RENDER_OK;
    });
}

int cuju_render_frame(cuju_renderer *renderer, int top_down, uint8_t *out, size_t size) {
    return Guard(renderer, [&]() -> int {
        const std::vector<std::uint8_t> &screen = renderer->renderer->screen();
        const size_t want =
            static_cast<size_t>(renderer->renderer->width()) * renderer->renderer->height() * 3;
        if (!out || size != want)
            return Fail(CUJU_RENDER_ERROR_ARGUMENT, "the output's size is not width * height * 3");
        if (!renderer->has_frame || screen.size() != want)
            return Fail(CUJU_RENDER_ERROR_STATE, "no frame was read back yet");
        const std::vector<std::uint8_t> &bytes = top_down ? TopDown(renderer) : screen;
        std::memcpy(out, bytes.data(), want);
        return CUJU_RENDER_OK;
    });
}

int cuju_render_pixels(cuju_renderer *renderer, int width, int height, uint8_t *out, size_t size) {
    return Guard(renderer,
                 [&]() -> int { return Resize(renderer, width, height, false, out, size); });
}

int cuju_render_pixels_gray(cuju_renderer *renderer, int width, int height, uint8_t *out,
                            size_t size) {
    return Guard(renderer,
                 [&]() -> int { return Resize(renderer, width, height, true, out, size); });
}

size_t cuju_render_digests(cuju_renderer *renderer, char *out, size_t size) {
    if (!renderer) return 0;
    cuju::render::FpScope fp(cuju::render::FpMode::kIeee);
    std::string text;
    try {
        std::string atlas, error;
        if (cuju::render::assets::text::ReadFile(
                renderer->media->media_path() + "/fonts/caption_glyphs.bin", &atlas, &error))
            text += "glyph_atlas " + cuju::render::Sha256Hex(atlas.data(), atlas.size()) + "\n";
        const cuju::render::pitch::PitchTextures &pitch = renderer->media->Pitch();
        const char *kinds[3] = {"diffuse", "specular", "normal"};
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 3; k++) {
                const std::vector<std::uint8_t> &v = k == 0   ? pitch.diffuse[i]
                                                     : k == 1 ? pitch.specular[i]
                                                              : pitch.normal[i];
                text += std::string("pitch_") + kinds[k] + "_0" + std::to_string(i + 1) + " " +
                        cuju::render::Sha256Hex(v.data(), v.size()) + "\n";
            }
        const std::vector<float> &kernel =
            renderer->renderer->context().scene().renderer().ssao_kernel();
        text += "ssao_kernel " +
                cuju::render::Sha256Hex(kernel.data(), kernel.size() * sizeof(float)) + "\n";
        cuju::render::gl::Context &gl = renderer->renderer->gl();
        gl.Finish();
        for (std::uint32_t name = 1; name < 4096; name++) {
            std::uint32_t ours = renderer->renderer->stream().ContextTexture(name);
            const cuju::render::Texture *t = ours ? gl.GetTexture(ours) : nullptr;
            if (!t || t->levels.size() < 2) continue;
            const bool rgb = t->format == cuju::render::Format::R8G8B8X8_UNORM ||
                             t->format == cuju::render::Format::R8G8B8X8_SRGB;
            const int bpp = cuju::render::BytesPerTexel(t->format);
            const int out_bpp = rgb ? 3 : bpp;
            for (std::size_t l = 0; l < t->levels.size(); l++) {
                const cuju::render::TextureLevel &level = t->levels[l];
                const std::size_t count = static_cast<std::size_t>(level.width) * level.height;
                std::vector<std::uint8_t> bytes(count * out_bpp);
                for (std::size_t k = 0; k < count; k++)
                    std::memcpy(&bytes[k * out_bpp], &level.texels[k * bpp], out_bpp);
                text += "mip_" + std::to_string(name) + "_" + std::to_string(l) + " " +
                        cuju::render::Sha256Hex(bytes.data(), bytes.size()) + "\n";
            }
        }
    } catch (const std::exception &e) {
        last_error = e.what();
    }
    if (out && size) {
        size_t n = text.size() < size - 1 ? text.size() : size - 1;
        std::memcpy(out, text.data(), n);
        out[n] = 0;
    }
    return text.size();
}

int cuju_render_has_frame(cuju_renderer *renderer, int *out) {
    return Guard(renderer, [&]() -> int {
        if (!out) return Fail(CUJU_RENDER_ERROR_ARGUMENT, "a null output");
        *out = renderer->has_frame ? 1 : 0;
        return CUJU_RENDER_OK;
    });
}

}
