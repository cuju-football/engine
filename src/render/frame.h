// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0
//
// The C++ entry of the software renderer: one Renderer per rendering engine, which is one GRF
// GameEnv, with its own mini GL context and GRF scene, sharing the read-only media of a process.
// The calls mirror where GRF's GameEnv touches graphics, the steps of a reset, StartMatch and
// render, and each render takes a RenderState, what GRF's Match::Put reads.

#ifndef CUJU_RENDER_FRAME_H
#define CUJU_RENDER_FRAME_H

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cuju/render/render_state.h"
#include "assets/assets.h"
#include "gl/context.h"
#include "gl/mip_cache.h"
#include "overlay2d/atlas.h"
#include "pitch/pitch.h"
#include "scene/gl_stream.h"
#include "scene/match.h"
#include "scene/netting.h"
#include "scene/skinning.h"

namespace cuju::render {

struct RendererOptions {
    int width = 1280;
    int height = 720;
    int threads = 1;
    std::string media_path;  // cuju-grf-media's root, holding data/ and fonts/
    // Records every GL call from the context's creation on, for checks against GRF's traces.
    scene::Recorder *recorder = nullptr;
    // Speed: a render whose frame nobody reads waits for the next call, and when that is the next
    // render of the match, which draws every pixel again, it runs every step but rasterizes
    // nothing. Off, every render rasterizes, as for checks of each pass against GRF's.
    bool skip_unread = true;
};

// Read-only media shared by the renderers of a process: the glyph atlas, the procedural pitch,
// the full body's vertex colours, and the images, meshes and mipmap chains the renderers decoded
// or filtered, each made on first use.
class SharedMedia {
public:
    explicit SharedMedia(std::string media_path, int threads = 1);
    const std::string &media_path() const { return media_path_; }
    std::string data_dir() const { return media_path_ + "/data"; }
    const overlay2d::GlyphAtlas &atlas() const { return atlas_; }
    bool ok() const;
    std::string error() const;
    const pitch::PitchTextures &Pitch();
    const scene::VertexColorMap &VertexColors();
    // GRF's image and ASE loaders, with what they read kept for the next renderer and match.
    bool LoadSurface(const std::string &path, assets::Surface *surface);
    bool LoadAse(const std::string &path, assets::GeometryData *data);
    const std::shared_ptr<gl::MipCache> &mip_cache() const { return mip_cache_; }

private:
    void SetError(const std::string &error);
    std::string media_path_;
    int threads_;
    overlay2d::GlyphAtlas atlas_;
    mutable std::mutex error_mutex_;
    std::string error_;
    std::once_flag pitch_once_, colors_once_;
    pitch::PitchTextures pitch_;
    scene::VertexColorMap colors_;
    std::shared_ptr<gl::MipCache> mip_cache_;
    std::mutex files_mutex_;
    std::map<std::string, std::shared_ptr<const assets::Surface>> surfaces_;
    std::map<std::string, std::shared_ptr<const assets::GeometryData>> meshes_;
};

// The match's looks: GRF's two teams of GameEnv::reset's MatchData and its player database, with
// the skins, hair, kits, heights and light of the RenderState.
scene::MatchLook MatchLookOf(const RenderState &state, std::uint32_t seed);

class Renderer {
public:
    Renderer(const RendererOptions &options, std::shared_ptr<SharedMedia> media);
    ~Renderer();
    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    // GameEnv::reset: ResetBegin, StopMatch, the loading page on the first reset of an engine,
    // RenderScene(true) when rendering, then StartMatch.
    void ResetBegin();
    void StopMatch();
    void LoadingPage();
    // GameTask::StartMatch with the RenderState of the new match, after ResetRenderState, and
    // the match's game_engine_random_seed.
    void StartMatch(const RenderState &state, std::uint32_t seed);
    // GameEnv::ProcessState of the rendering match, at get_state and set_state.
    void StateProcessed();
    // GRF's `game_config.render`, on at first. Off, StartMatch builds the match as GRF builds one
    // without rendering, on the pitch alone, with no pitch textures generated, no captions drawn
    // and bodies that skip the bake, and captions keep their text.
    void SetRender(bool render);
    bool render() const { return render_; }
    // GraphicsTask::Render alone, as the reset calls it.
    const std::vector<std::uint8_t> &RenderScene(bool swap);
    // GameEnv::render: GameTask::PrepareRender with the state, then GraphicsTask::Render. With
    // `swap` the frame is read back. A render without `swap` is one whose frame nobody reads.
    const std::vector<std::uint8_t> &Render(const RenderState &state, bool swap) {
        return Render(state, swap, swap);
    }
    // The same, with `read` false for a render whose frame the caller never reads, a swap
    // included: the screen then keeps the last frame read.
    const std::vector<std::uint8_t> &Render(const RenderState &state, bool swap, bool read);
    // Runs a render that waits, in full, so that the context shows its pixels.
    void Flush();
    // Whether a render waits for the next call.
    bool deferred() const { return pending_; }

    bool has_match() { return context_->match() != nullptr; }
    // glReadPixels' bytes of the last swap: rows from the bottom, RGB.
    const std::vector<std::uint8_t> &screen() const { return screen_; }
    int width() const { return options_.width; }
    int height() const { return options_.height; }
    scene::Context &context() { return *context_; }
    scene::GlStream &stream() { return *stream_; }
    gl::Context &gl() {
        Flush();
        return *gl_;
    }

private:
    const scene::SkinnedModel &Model(int body);
    void Skin(int body, scene::GeometryData &data, bool rest, const RenderJoint *joints);
    void ApplyState(const RenderState &state);
    const std::vector<std::uint8_t> &RenderMatch(const RenderState &state, bool swap);
    // Runs the waiting render, rasterizing nothing when `discard`.
    void Resolve(bool discard);

    RendererOptions options_;
    std::shared_ptr<SharedMedia> media_;
    std::unique_ptr<gl::Context> gl_;
    std::unique_ptr<scene::GlStream> stream_;
    std::unique_ptr<scene::Context> context_;
    std::vector<std::uint8_t> screen_;

    // Per match, in the order GRF creates the bodies: the scale, the skinned model, and where the
    // body sits in the RenderState, its team and slot, or -1 and the official's index.
    std::vector<float> body_z_;
    std::vector<std::array<int, 2>> body_slot_;
    std::vector<scene::SkinnedModel> models_;
    // Whether the match's bodies were baked, which GRF does only while rendering.
    bool baked_ = true;
    bool render_ = true;
    std::vector<float> skin_buffer_;
    std::vector<std::vector<float>> skin_buffers_;
    // The goal nets, kept across matches as GRF keeps the goals mesh in its context.
    scene::GoalNetting netting_;
    bool netting_prepared_ = false;
    // What the scoreboard and the message caption show, which GRF changes outside Match::Put.
    std::int32_t shown_goals_[2] = {0, 0};
    std::string shown_message_;
    bool message_visible_ = false;
    // The render that waits for the next call.
    bool pending_ = false, pending_swap_ = false;
    std::unique_ptr<RenderState> pending_state_;
};

}  // namespace cuju::render

#endif  // CUJU_RENDER_FRAME_H
