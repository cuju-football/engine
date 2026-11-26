// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../assets/scenario_builder.h"
#include "../batch/batch_core.h"
#include "../batch/render_states.h"
#include "../batch/tables.h"
#include "engine_info.h"

namespace cuju {
namespace engine {
struct StateRefused : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
struct ScenarioRefused : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
struct UndefinedPath : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct CallRefused : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
struct OrderRefused : std::logic_error {
    using std::logic_error::logic_error;
};

// GRF: `ScenarioConfig`, with the cache of `DynamicPlayerSelection` and the controllable counts.
struct ScenarioConfig {
    state::ScenarioState scenario{};
    bool cacheComputed = false, cachedDynamicPlayerSelection = false;
    int cachedControllableLeft = -1, cachedControllableRight = -1;
    // GRF: `ScenarioConfig::ComputeCache`.
    void ComputeCache() {
        if (cacheComputed) return;
        cachedControllableLeft = cachedControllableRight = 0;
        for (int player = 0; player < scenario.leftTeamSize; player++)
            cachedControllableLeft += scenario.leftTeam[player].controllable ? 1 : 0;
        for (int player = 0; player < scenario.rightTeamSize; player++)
            cachedControllableRight += scenario.rightTeam[player].controllable ? 1 : 0;
        cachedDynamicPlayerSelection =
            !((cachedControllableLeft == scenario.leftAgents || scenario.leftAgents == 0) &&
              (cachedControllableRight == scenario.rightAgents || scenario.rightAgents == 0));
        cacheComputed = true;
    }
};

struct HostCopies {
    static constexpr bool HOST = true;
    simulation::SimulationContext context{};
    std::vector<std::shared_ptr<void>> blocks;

    template <class T>
    T *Upload(const T *data, size_t count) {
        static_assert(std::is_trivially_copyable<T>::value, "tables must be trivially copyable");
        if (count == 0) return nullptr;
        constexpr std::align_val_t ALIGNMENT{64};
        void *copy = ::operator new(count * sizeof(T), ALIGNMENT);
        memcpy(copy, static_cast<const void *>(data), count * sizeof(T));
        blocks.emplace_back(copy, [](void *block) { ::operator delete(block, ALIGNMENT); });
        return static_cast<T *>(copy);
    }
    template <class T>
    T *Upload(const std::vector<T> &values) {
        return Upload(values.data(), values.size());
    }
};

struct EngineTables {
    assets::BuilderTables builder;
    HostCopies host;
};

inline std::shared_ptr<const EngineTables> SharedEngineTables(const std::string &animations) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const EngineTables>> tables;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = tables.find(animations);
    if (found != tables.end()) return found->second;
    auto built = std::make_shared<EngineTables>();
    built->builder = assets::LoadBuilderTables(batch::ReadFileBytes(animations));
    for (const animation::Animation &entry : built->builder.store->animations)
        if (entry.frameCount > state::MAXIMUM_ANIMATION_POSITIONS)
            throw std::invalid_argument("expected animations of at most " +
                                        std::to_string(state::MAXIMUM_ANIMATION_POSITIONS) +
                                        " frames, found " + std::to_string(entry.frameCount));
    batch::UploadTables(built->host, *built->builder.store, built->builder.derived, {},
                        simulation::StaticData{});
    tables[animations] = built;
    return built;
}

struct EngineTemplate {
    batch::AssetVariant variant;
};

inline std::string TemplateKey(const state::ScenarioState &source) {
    state::ScenarioState scenario = source;
    scenario.gameEngineRandomSeed = 0;
    scenario.leftTeamDifficulty = scenario.rightTeamDifficulty = 0;
    scenario.realTime = false;
    state::Archive archive = state::Archive::Saver();
    scenario.Process(archive);
    return archive.data();
}

inline std::shared_ptr<const EngineTemplate> BuildTemplate(const state::ScenarioState &scenario,
                                                           const EngineTables &tables) {
    assets::ScenarioDescription description;
    description.useMagnet = scenario.useMagnet;
    description.offsides = scenario.offsides;
    description.realTime = scenario.realTime;
    description.leftTeamDifficulty = scenario.leftTeamDifficulty;
    description.rightTeamDifficulty = scenario.rightTeamDifficulty;
    description.deterministic = scenario.deterministic;
    description.endEpisodeOnScore = scenario.endEpisodeOnScore;
    description.endEpisodeOnPossessionChange = scenario.endEpisodeOnPossessionChange;
    description.endEpisodeOnOutOfPlay = scenario.endEpisodeOnOutOfPlay;
    description.gameDuration = scenario.gameDuration;
    description.secondHalf = scenario.secondHalf;
    description.controlAllPlayers = scenario.controlAllPlayers;
    description.gameEngineRandomSeed = scenario.gameEngineRandomSeed;
    assets::ScenarioTeams teams;
    for (int player = 0; player < scenario.leftTeamSize; player++)
        teams[0].push_back(scenario.leftTeam[player]);
    for (int player = 0; player < scenario.rightTeamSize; player++)
        teams[1].push_back(scenario.rightTeam[player]);
    assets::BuiltVariant built;
    try {
        built = assets::BuildVariant(description, teams, scenario.ballPosition,
                                     scenario.reverseTeamProcessing, tables.builder);
    } catch (const std::invalid_argument &exception) {
        throw ScenarioRefused(exception.what());
    }
    const int left = scenario.leftAgents, right = scenario.rightAgents;
    auto made = std::make_shared<EngineTemplate>();
    simulation::StateCodec codec;
    try {
        made->variant = batch::LoadVariant(
            built.staticData, built.state, built.skeletons, scenario.reverseTeamProcessing ? 1 : 0,
            0, left != 0 || right != 0, left, right, "the scenario", codec);
    } catch (const std::invalid_argument &exception) {
        throw ScenarioRefused(exception.what());
    }
    return made;
}

class TemplateCache {
public:
    std::shared_ptr<const EngineTemplate> Get(const state::ScenarioState &scenario,
                                              const EngineTables &tables) {
        const std::string key = TemplateKey(scenario);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto entry = entries_.begin(); entry != entries_.end(); ++entry)
                if (entry->first == key) {
                    entries_.splice(entries_.begin(), entries_, entry);
                    return entries_.front().second;
                }
        }
        std::shared_ptr<const EngineTemplate> built = BuildTemplate(scenario, tables);
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.emplace_front(key, built);
        while (entries_.size() > CAPACITY) entries_.pop_back();
        return built;
    }
    static TemplateCache &Instance() {
        static TemplateCache cache;
        return cache;
    }

private:
    static constexpr size_t CAPACITY = 8;
    std::mutex mutex_;
    std::list<std::pair<std::string, std::shared_ptr<const EngineTemplate>>> entries_;
};

// GRF: `ScenarioConfig::ProcessStateConstant`, the part of a scenario that `set_state` compares.
inline std::string ConstantPart(const state::ScenarioState &scenario) {
    std::string out;
    auto append = [&](const void *value, size_t size) {
        out.append(static_cast<const char *>(value), size);
    };
    append(&scenario.ballPosition, sizeof(Vector3));
    append(&scenario.leftTeamSize, 4);
    append(&scenario.rightTeamSize, 4);
    append(&scenario.leftAgents, 4);
    append(&scenario.rightAgents, 4);
    append(&scenario.useMagnet, 1);
    append(&scenario.offsides, 1);
    append(&scenario.leftTeamDifficulty, 4);
    append(&scenario.rightTeamDifficulty, 4);
    append(&scenario.deterministic, 1);
    append(&scenario.endEpisodeOnScore, 1);
    append(&scenario.endEpisodeOnPossessionChange, 1);
    append(&scenario.endEpisodeOnOutOfPlay, 1);
    append(&scenario.gameDuration, 4);
    append(&scenario.secondHalf, 4);
    append(&scenario.controlAllPlayers, 1);
    return out;
}

inline std::string ConstantDifference(const state::ScenarioState &engine,
                                      const state::ScenarioState &loaded) {
    std::vector<std::string> names;
    auto check = [&](bool same, const char *name) {
        if (!same) names.push_back(name);
    };
    check(memcmp(&engine.ballPosition, &loaded.ballPosition, sizeof(Vector3)) == 0,
          "ball_position");
    check(engine.leftTeamSize == loaded.leftTeamSize, "left_team");
    check(engine.rightTeamSize == loaded.rightTeamSize, "right_team");
    check(engine.leftAgents == loaded.leftAgents, "left_agents");
    check(engine.rightAgents == loaded.rightAgents, "right_agents");
    check(engine.useMagnet == loaded.useMagnet, "use_magnet");
    check(engine.offsides == loaded.offsides, "offsides");
    check(memcmp(&engine.leftTeamDifficulty, &loaded.leftTeamDifficulty, 4) == 0,
          "left_team_difficulty");
    check(memcmp(&engine.rightTeamDifficulty, &loaded.rightTeamDifficulty, 4) == 0,
          "right_team_difficulty");
    check(engine.deterministic == loaded.deterministic, "deterministic");
    check(engine.endEpisodeOnScore == loaded.endEpisodeOnScore, "end_episode_on_score");
    check(engine.endEpisodeOnPossessionChange == loaded.endEpisodeOnPossessionChange,
          "end_episode_on_possession_change");
    check(engine.endEpisodeOnOutOfPlay == loaded.endEpisodeOnOutOfPlay,
          "end_episode_on_out_of_play");
    check(engine.gameDuration == loaded.gameDuration, "game_duration");
    check(engine.secondHalf == loaded.secondHalf, "second_half");
    check(engine.controlAllPlayers == loaded.controlAllPlayers, "control_all_players");
    std::string text;
    for (size_t index = 0; index < names.size() && index < 3; index++)
        text += (index ? ", " : "") + names[index];
    return text;
}

class GameEngine {
public:
    explicit GameEngine(const std::string &animations) : tables_(SharedEngineTables(animations)) {}

    // GRF: `start_game`, which plays nothing until a reset.
    void Start() {
        if (started_) throw OrderRefused("start_game() runs once per engine, as GRF asserts");
        started_ = true;
    }

    // GRF: `GameEnv::reset`.
    void Reset(ScenarioConfig &config, bool animations, int ticks) {
        if (!started_) throw OrderRefused("reset() must run after start_game()");
        CheckTicks(ticks);
        state::ScenarioState &scenario = config.scenario;
        CheckAgents(scenario);
        CheckHistory(scenario.gameDuration, ticks);
        std::shared_ptr<const EngineTemplate> made =
            TemplateCache::Instance().Get(scenario, *tables_);
        const Vector3 unscaled = scenario.ballPosition;
        // GRF: `setConfig` scales x and y in double, before the copy.
        scenario.ballPosition.coordinates[0] =
            (float)((double)scenario.ballPosition.coordinates[0] * 54.4);
        scenario.ballPosition.coordinates[1] =
            (float)((double)scenario.ballPosition.coordinates[1] * -83.6);
        template_ = made;
        templateSource_ = scenario;
        templateSource_.ballPosition = unscaled;
        context_ = tables_->host.context;
        context_.staticData = &made->variant.staticData;
        context_.skeletons = made->variant.skeletons.data();
        context_.ticksPerStep = ticks;
        auto world = std::make_unique<simulation::World>(*made->variant.world);
        state::ScenarioState &played = world->serialized.scenario;
        played.leftTeamDifficulty = scenario.leftTeamDifficulty;
        played.rightTeamDifficulty = scenario.rightTeamDifficulty;
        played.realTime = scenario.realTime;
        world->refereeAnimations = animations;
        // GRF: a new match of graphics that do not render computes every cache when read.
        world->renderMode = renderMode_ ? simulation::e_Render_On : simulation::e_Render_Off;
        renderMatchPending_ = true;
        playedLeftDifficulty_ = scenario.leftTeamDifficulty;
        playedRightDifficulty_ = scenario.rightTeamDifficulty;
        playedUseMagnet_ = scenario.useMagnet;
        waitingForGameCount_ = 0;
        if (!simulation::EnvironmentResetFromTemplate(*world, context_,
                                                      scenario.gameEngineRandomSeed))
            throw UndefinedPath(simulation::UnsupportedReason());
        world_ = std::move(world);
        config_ = config;
        freeControllerCount_ = made->variant.freeControllerCount;
    }

    // GRF reads these fields of `GameEnv.config` during the match.
    void SetConfig(const ScenarioConfig &config) {
        RequireMatch();
        const state::ScenarioState &scenario = config.scenario;
        const state::ScenarioState &played = world_->serialized.scenario;
        if (scenario.leftTeamSize != played.leftTeamSize ||
            scenario.rightTeamSize != played.rightTeamSize)
            throw ScenarioRefused(
                "the teams of GameEnv.config changed size during the match, which GRF plays "
                "with formation entries that no player has");
        if (scenario.reverseTeamProcessing != played.reverseTeamProcessing)
            throw ScenarioRefused(
                "reverse_team_processing of GameEnv.config changed during the match, which "
                "GRF plays with its teams processed in one order and mirrored in the other");
        config_ = config;
    }
    const ScenarioConfig &Config() const { return config_; }

    // GRF: an action outside the action set only enables the keyboard, as idle does.
    void Action(int action, bool leftTeam, int player) {
        RequireMatch();
        CheckController(leftTeam, player);
        simulation::EnvironmentAction(*world_, action, leftTeam, player);
    }

    // GRF: any action but a direction or a sticky button is a fatal error.
    bool Sticky(int action, bool leftTeam, int player) {
        RequireMatch();
        bool sticky = false;
        for (int index = 0; index < simulation::MAXIMUM_STICKY_ACTIONS; index++)
            sticky |= simulation::StickyAction(index) == action;
        if (!sticky)
            throw CallRefused(
                "sticky_action_state takes a direction or a sticky button, found "
                "the action " +
                std::to_string(action));
        CheckController(leftTeam, player);
        return simulation::StickyActionState(*world_, action, leftTeam, player);
    }

    void Ticks(int ticks, int count) {
        RequireMatch();
        CheckTicks(ticks);
        Apply(ticks);
        for (int tick = 0; tick < count; tick++)
            if (!simulation::GameTaskProcessPhase(*world_, context_))
                throw UndefinedPath(simulation::UnsupportedReason());
        Collect();
    }
    void FinishStep() {
        RequireMatch();
        simulation::EnvironmentFinishStep(*world_);
    }
    // GRF: `GameEnv::step` without rendering.
    void Step(int ticks) {
        Ticks(ticks, ticks);
        simulation::EnvironmentFinishStep(*world_);
    }
    bool InPlay() const {
        RequireMatch();
        return world_->serialized.match.inPlay;
    }

    void Info(int ticks, EngineInfo &info) {
        RequireMatch();
        CheckTicks(ticks);
        context_.ticksPerStep = ticks;
        EnvironmentGetInfo(*world_, context_, info);
    }

    // GRF: render mode, the engine's graphics system with `game_config.render` on.
    void SetRenderMode(bool renderMode) {
        renderMode_ = renderMode;
        if (!world_) return;
        if (renderMode)
            world_->renderMode = simulation::e_Render_On;
        else if (world_->renderMode == simulation::e_Render_On)
            world_->renderMode = simulation::e_Render_Idle;
    }
    bool RenderMode() const { return renderMode_; }
    // GRF: `GameEnv::render`'s `PrepareRender`, what a render changes in the match.
    void PrepareRender() {
        RequireMatch();
        simulation::MatchPut(*world_);
    }
    const render::RenderState &ExtractRenderState() {
        render::RenderMessage message;
        render::RenderContext context = MakeRenderContext(message);
        if (renderMatchPending_) StartRenderState();
        render::ExtractRenderState(*world_, context, *renderState_);
        return *renderState_;
    }
    const render::RenderState &StartRenderState() {
        render::RenderMessage message;
        render::RenderContext context = MakeRenderContext(message);
        render::ResetRenderState(*world_, context, batch::MakeRenderDraws(*world_), *renderState_);
        renderMatchPending_ = false;
        return *renderState_;
    }

    // GRF: `start_game`'s own reset, of `ScenarioConfig::make()` with the seed 42, a match of two
    // teams of the eleven players of `TeamData`'s formation, the goalkeeper first. cuju plays no
    // match before the first reset, so only the pictures of that match are made: the looks, from
    // the player database, and the officials' skin and the light, from the draws of its creation.
    static render::RenderState InternalRenderState(bool firstOfProcess) {
        constexpr int PLAYERS = 11;
        constexpr uint32_t SEED = 42;
        render::RenderState out{};
        out.version = render::RENDER_STATE_VERSION;
        out.firstTeam = 0;
        uint8_t generator[RNG_BYTES] = {};
        const float range[2] = {0.0f, 1.0f};
        memcpy(generator + RNG_BYTES - 8, range, 8);
        RNGView{generator}.Seed(SEED);
        for (int draw = 0; draw < 2 * 2 * PLAYERS; draw++) RandomInRange(generator, 0, 1);
        render::RenderMatchDraws draws;
        draws.officialSkin = RandomInRange(generator, 1, 4);
        for (int draw = 0; draw < 3 * 3 + (firstOfProcess ? 1 : 0); draw++)
            RandomInRange(generator, 0, 1);
        draws.light[0] = RandomInRange(generator, -1.7f, 1.7f);
        draws.light[1] = RandomInRange(generator, -1.7f, 1.7f);
        draws.light[2] = RandomInRange(generator, 0, 1);
        for (int draw = 0; draw < 3; draw++)
            draws.light[3 + draw] = RandomInRange(generator, (float)-0.1, (float)0.1);
        render::RenderEpisode &episode = out.episode;
        render::SunFromDraws(draws.light, episode.sunPosition, episode.sunColor);
        episode.reverseTeamProcessing = false;
        const float zFactor = 1.0f / render::detail::DEFAULT_PLAYER_HEIGHT;
        for (int team = 0; team < 2; team++) {
            episode.teamSize[team] = PLAYERS;
            for (int slot = 0; slot < PLAYERS; slot++) {
                render::RenderHumanoidStatic &entry = episode.players[team][slot];
                entry.zMultiplier =
                    zFactor * assets::PLAYER_TABLE[slot][simulation::PLAYER_STATISTICS + 1];
                entry.skin = (int8_t)assets::PLAYER_SKIN[slot];
                render::detail::HairOfSlot(slot, entry.hairStyle, entry.hairColor);
                entry.kit = slot == 0 ? render::e_RenderKit_Goalkeeper : render::e_RenderKit_Team;
                entry.dataIndex = (int8_t)slot;
            }
        }
        for (int official = 0; official < render::RENDER_OFFICIALS; official++) {
            render::RenderHumanoidStatic &entry = episode.officials[official];
            entry.zMultiplier = zFactor * assets::OFFICIAL_HEIGHT;
            entry.skin = (int8_t)(int)roundf(draws.officialSkin);
            render::detail::HairOfSlot(-1, entry.hairStyle, entry.hairColor);
            entry.kit = render::e_RenderKit_Referee;
            entry.dataIndex = -1;
        }
        out.yellowCard.position = Vector3(0, 0, -10);
        out.yellowCard.rotation = Quaternion(0, 0, -0.0f, 1);
        out.redCard = out.yellowCard;
        return out;
    }
    render::RenderContext MakeRenderContext(render::RenderMessage &message) {
        RequireMatch();
        if (!renderState_) renderState_ = std::make_unique<render::RenderState>();
        if (animationMirrored_.empty())
            animationMirrored_ = batch::AnimationMirroredNames(tables_->builder.store->names);
        batch::MakeRenderMessage(*world_, message);
        render::RenderContext context;
        context.simulation = &context_;
        context.animationMirrored = animationMirrored_.data();
        context.message = &message;
        return context;
    }
    uint32_t StateProcessed() const {
        RequireMatch();
        return world_->stateProcessed;
    }

    int GameState() const { return gameState_; }
    void SetGameState(int value) { gameState_ = value; }
    int WaitingForGameCount() const { return waitingForGameCount_; }
    void SetWaitingForGameCount(int value) { waitingForGameCount_ = value; }

    // GRF: `GameEnv::get_state`, the pickle with its `int32` length, then the engine.
    std::string GetState(const std::string &pickle) {
        RequireMatch();
        std::string bytes;
        const int32_t size = (int32_t)pickle.size();
        bytes.append(reinterpret_cast<const char *>(&size), 4);
        bytes += pickle;
        bytes += SaveEngine(config_.scenario);
        config_.cacheComputed = false;
        // GRF: `ProcessState` marks every node to compute afresh, a save as a load.
        simulation::ProcessStateMarks(*world_);
        return bytes;
    }

    // GRF: `GameEnv::set_state`.
    std::pair<size_t, size_t> SetState(const std::string &bytes) {
        RequireMatch();
        int32_t size = 0;
        if (bytes.size() < 4) throw StateRefused("the state is shorter than its pickle's length");
        memcpy(&size, bytes.data(), 4);
        if (size < 0 || (size_t)size > bytes.size() - 4)
            throw StateRefused("the state is shorter than its pickle");
        const std::string engine = bytes.substr(4 + (size_t)size);
        auto loaded = std::make_unique<state::EnvironmentStateFull>();
        try {
            state::Archive archive = state::Archive::Loader(engine);
            state::ProcessEnvironmentScenario(archive, *loaded);
        } catch (const std::exception &exception) {
            throw StateRefused(std::string("the state holds no scenario: ") + exception.what());
        }
        if (ConstantPart(loaded->scenario) != ConstantPart(config_.scenario))
            throw StateRefused(
                "Current environment scenario != scenario in the state, as GRF reports, in " +
                ConstantDifference(config_.scenario, loaded->scenario));
        const state::ScenarioState &fresh = loaded->scenario;
        std::shared_ptr<const EngineTemplate> made = template_;
        state::ScenarioState source = templateSource_;
        if (fresh.reverseTeamProcessing != source.reverseTeamProcessing) {
            // GRF: the state's order of the teams loads into the scenario, so it plays from a
            // template of its own.
            source.reverseTeamProcessing = fresh.reverseTeamProcessing;
            made = TemplateCache::Instance().Get(source, *tables_);
        }
        auto world = std::make_unique<simulation::World>();
        simulation::StateCodec codec;
        codec.freeControllerCount = freeControllerCount_;
        codec.animationCount = (int)tables_->builder.store->animations.size();
        if (!codec.LoadFrame(engine, 0, *world))
            throw StateRefused("the state does not load into this engine");
        freeControllerCount_ = codec.freeControllerCount;
        // GRF: the referee's flag and a sent-off keeper's hand-over live outside the state, and the
        // teams keep the difficulties and the magnet of their reset.
        world->refereeAnimations = world_->refereeAnimations;
        memcpy(world->formationOverride, world_->formationOverride,
               sizeof(world->formationOverride));
        // GRF: the match keeps its graphics, its message caption and its light, and every root
        // cache is to compute afresh.
        world->renderMode = world_->renderMode;
        world->message = world_->message;
        world->draws = world_->draws;
        memcpy(world->scoreboardGoals, world_->scoreboardGoals, sizeof(world->scoreboardGoals));
        memcpy(world->humanoidHidden, world_->humanoidHidden, sizeof(world->humanoidHidden));
        world->stateProcessed = world_->stateProcessed + 1;
        // GRF: `Match::ProcessState` mirrors the ball predictions of each mental image whose flag
        // `ballPredictions_mirrored` is off once more around their load, after it turned the images
        // that the engine held into the state's frame.
        if (world->serialized.scenario.reverseTeamProcessing) {
            const int held = world_->serialized.match.mentalImages.count;
            for (int image = 0; image < world->serialized.match.mentalImages.count; image++) {
                if (image < held && !world_->flags.mentalBallMirrored[image]) continue;
                world->serialized.match.mentalImages[image].lazyFlip =
                    !world->serialized.match.mentalImages[image].lazyFlip;
                world->flags.mentalBallMirrored[image] = true;
            }
        }
        gameState_ = world->serialized.gameState;
        waitingForGameCount_ = world->serialized.waitingForGameCount;
        config_.scenario = world->serialized.scenario;
        config_.cacheComputed = false;
        template_ = made;
        templateSource_ = source;
        context_.staticData = &made->variant.staticData;
        context_.skeletons = made->variant.skeletons.data();
        world_ = std::move(world);
        return {4, (size_t)size};
    }

    // GRF: `Tracker::verify_snapshot`, run by this engine on the engine that waits.
    long long Verify(const GameEngine &waiting) {
        RequireMatch();
        waiting.RequireMatch();
        std::vector<state::LayoutEntry> mine, theirs;
        const std::string first = SaveEngine(config_.scenario, &mine);
        const std::string second = waiting.SaveEngine(waiting.config_.scenario, &theirs);
        config_.cacheComputed = false;
        simulation::ProcessStateMarks(*world_);
        const size_t count = std::min(mine.size(), theirs.size());
        for (size_t index = 0; index < count; index++) {
            const state::LayoutEntry &entry = mine[index], &other = theirs[index];
            if (entry.path != other.path || entry.size != other.size ||
                entry.position != other.position)
                return std::min(entry.position, other.position);
            if (Unvalidated(entry.path)) continue;
            if (!SameValue(entry, first.data() + entry.position, second.data() + other.position))
                return entry.position;
        }
        if (mine.size() != theirs.size() || first.size() != second.size())
            return (long long)std::min(first.size(), second.size());
        return -1;
    }

private:
    // GRF: the scenario part, saved from one configuration for both, and the fields under
    // `setValidate(false)`, the camera, the keyboards' mirror, the player colour, the linesmen and
    // each gamer's keyboard.
    static bool Unvalidated(const std::string &path) {
        auto starts = [&](const char *prefix) { return path.rfind(prefix, 0) == 0; };
        auto ends = [&](const std::string &suffix) {
            return path.size() >= suffix.size() &&
                   path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        return starts("environment.scenario") || starts("match.camera") ||
               starts("match.officials[1]") || starts("match.officials[2]") || ends(".mirror") ||
               ends(".bufferPlayerColor") || ends(".keyboardIndex");
    }
    static bool SameValue(const state::LayoutEntry &entry, const char *first, const char *second) {
        auto floats = [&](int count) {
            for (int index = 0; index < count; index++) {
                float a, b;
                memcpy(&a, first + 4 * index, 4);
                memcpy(&b, second + 4 * index, 4);
                if (a != b) return false;
            }
            return true;
        };
        const std::string type = entry.type;
        if (type == TYPE_FLOAT && entry.size == 4) return floats(1);
        if (type == TYPE_VECTOR3 && entry.size == 12) return floats(3);
        if (type == TYPE_QUATERNION && entry.size == 16) return floats(4);
        if (type == TYPE_RADIAN && entry.size == 8) {
            state::Radian a, b;
            memcpy(&a, first, 8);
            memcpy(&b, second, 8);
            return a.value() == b.value();
        }
        return memcmp(first, second, entry.size) == 0;
    }
    void RequireMatch() const {
        if (!world_)
            throw OrderRefused(
                "the engine holds no match until reset(), which must run after start_game()");
    }
    // GRF: `player` indexes the 22 keyboards from 0 for the left team and from 11 for the right.
    static void CheckController(bool leftTeam, int player) {
        const int controller = player + (leftTeam ? 0 : simulation::MAXIMUM_TEAM_CONTROLLERS);
        if (controller < 0 || controller >= 2 * simulation::MAXIMUM_TEAM_CONTROLLERS)
            throw CallRefused("expected a controller from 0 to 21, found " +
                              std::to_string(controller) + ", the player " +
                              std::to_string(player) + " of the " + (leftTeam ? "left" : "right") +
                              " team, where GRF reads past its keyboards");
    }
    static void CheckTicks(int ticks) {
        // GRF: a step of no ticks never brings the match into play.
        if (ticks < 1)
            throw CallRefused("expected physics_steps_per_frame of at least 1, found " +
                              std::to_string(ticks));
    }
    // GRF: `Match::GetState` writes a controller into a list as long as its team.
    static void CheckAgents(const state::ScenarioState &scenario) {
        if (scenario.leftAgents < 0 || scenario.rightAgents < 0)
            throw ScenarioRefused("expected agent counts of zero or more");
        if (scenario.leftAgents > scenario.leftTeamSize ||
            scenario.rightAgents > scenario.rightTeamSize)
            throw ScenarioRefused("expected at most one agent for each player of a team, found " +
                                  std::to_string(scenario.leftAgents) + " and " +
                                  std::to_string(scenario.rightAgents) + " agents for teams of " +
                                  std::to_string(scenario.leftTeamSize) + " and " +
                                  std::to_string(scenario.rightTeamSize) +
                                  ", where GRF writes past the controllers of get_info()");
    }
    // GRF: a player keeps one position per second of play between kick-offs, without a limit.
    static void CheckHistory(int gameDuration, int ticks) {
        constexpr int HISTORY =
            decltype(state::PlayerBaseState<
                     state::AIControllerState>::positionHistoryPerSecond)::CAPACITY;
        const int64_t seconds = (int64_t)gameDuration * ticks / 100 + 1;
        if (seconds > HISTORY)
            throw ScenarioRefused("expected a match of at most " + std::to_string(HISTORY) +
                                  " seconds of play, the position history of a player, found " +
                                  std::to_string(gameDuration) + " steps of " +
                                  std::to_string(ticks) + " ticks");
    }
    void Apply(int ticks) {
        CheckHistory(config_.scenario.gameDuration, ticks);
        context_.ticksPerStep = ticks;
        state::ScenarioState &played = world_->serialized.scenario;
        played = config_.scenario;
        played.leftTeamDifficulty = playedLeftDifficulty_;
        played.rightTeamDifficulty = playedRightDifficulty_;
        played.useMagnet = playedUseMagnet_;
        if (config_.cacheComputed) {
            world_->dynamicPlayerSelection = config_.cachedDynamicPlayerSelection ? 1 : 0;
        } else {
            ScenarioConfig fresh = config_;
            fresh.ComputeCache();
            world_->dynamicPlayerSelection = fresh.cachedDynamicPlayerSelection ? 3 : 2;
        }
    }
    // GRF: the first read of `DynamicPlayerSelection` fills the cache of the scenario.
    void Collect() {
        if (!config_.cacheComputed && world_->dynamicPlayerSelection < 2) config_.ComputeCache();
    }

    std::string SaveEngine(const state::ScenarioState &scenario,
                           std::vector<state::LayoutEntry> *layout = nullptr) const {
        auto copy = std::make_unique<simulation::World>(*world_);
        copy->serialized.scenario = scenario;
        copy->serialized.gameState = gameState_;
        copy->serialized.waitingForGameCount = waitingForGameCount_;
        // GRF: a flagged image saves its predictions without the turn into the state's frame.
        for (int image = 0; image < copy->serialized.match.mentalImages.count; image++)
            if (copy->flags.mentalBallMirrored[image])
                copy->serialized.match.mentalImages[image].lazyFlip =
                    !copy->serialized.match.mentalImages[image].lazyFlip;
        simulation::RestToCanonical(*copy);
        simulation::BallMaterialize(copy->serialized.match.ball);
        for (int image = 0; image < copy->serialized.match.mentalImages.count; image++)
            simulation::MentalMaterialize(copy->serialized.match.mentalImages[image]);
        state::Archive archive = state::Archive::Saver();
        if (layout) {
            archive.layout = layout;
            archive.layoutBase = &copy->serialized;
            archive.layoutStorage = false;
        }
        state::ProcessEnvironmentState(archive, copy->serialized, freeControllerCount_);
        return archive.data();
    }

    std::shared_ptr<const EngineTables> tables_;
    std::shared_ptr<const EngineTemplate> template_;
    state::ScenarioState templateSource_{};
    simulation::SimulationContext context_{};
    std::unique_ptr<simulation::World> world_;
    ScenarioConfig config_;
    float playedLeftDifficulty_ = 0, playedRightDifficulty_ = 0;
    bool playedUseMagnet_ = true;
    int gameState_ = 0, waitingForGameCount_ = 0;
    int freeControllerCount_ = 0;
    bool started_ = false;
    bool renderMode_ = false;
    std::unique_ptr<render::RenderState> renderState_;
    bool renderMatchPending_ = true;
    std::vector<uint8_t> animationMirrored_;
};

}  // namespace engine
}  // namespace cuju
