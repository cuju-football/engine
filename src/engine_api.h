// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <memory>
#include <string>

#include "cuju/engine/game_engine.h"
#include "cuju/floating_point.h"

namespace {
int EngineFail(const std::exception &exception) {
    Fail(exception);
    if (dynamic_cast<const cuju::engine::UndefinedPath *>(&exception)) return -2;
    if (dynamic_cast<const cuju::engine::StateRefused *>(&exception)) return -3;
    if (dynamic_cast<const cuju::engine::ScenarioRefused *>(&exception)) return -4;
    if (dynamic_cast<const cuju::engine::OrderRefused *>(&exception)) return -5;
    return -1;
}

cuju::engine::GameEngine &Engine(void *engine) {
    return *static_cast<cuju::engine::GameEngine *>(engine);
}

void CheckRenderStateSize(unsigned long long size) {
    if (size != sizeof(cuju::render::RenderState))
        throw std::invalid_argument(
            "expected a RenderState of " + std::to_string(sizeof(cuju::render::RenderState)) +
            " bytes, version " + std::to_string(cuju::render::RENDER_STATE_VERSION) + ", found " +
            std::to_string(size) + " bytes");
}
}  // namespace

#if !defined(_MSC_VER)
#pragma GCC visibility push(default)
#endif
extern "C" {
#define CUJU_ENGINE_VERSION 2

// GRF's `FormationEntry` as `ScenarioConfig` stores it, `y` already times -2.36.
struct cuju_engine_entry {
    float position[3], start_position[3];
    int role, lazy, controllable;
};

// GRF's `ScenarioConfig`, every field.
struct cuju_engine_scenario {
    int version;
    float ball_position[3];
    cuju_engine_entry *left_team;
    int left_size;
    cuju_engine_entry *right_team;
    int right_size;
    int left_agents, right_agents;
    int use_magnet, offsides, real_time;
    unsigned int game_engine_random_seed;
    int reverse_team_processing;
    float left_team_difficulty, right_team_difficulty;
    int deterministic, end_episode_on_score, end_episode_on_possession_change,
        end_episode_on_out_of_play;
    int game_duration, second_half, control_all_players;
    int cache_computed, cache_dynamic_player_selection, cache_controllable_left,
        cache_controllable_right;
};

struct cuju_engine_player {
    float position[3], direction[3];
    float tired_factor;
    int has_card, is_active, designated_player, role;
};

// GRF's `SharedInfo`.
struct cuju_engine_info {
    int version;
    float ball_position[3], ball_direction[3], ball_rotation[3];
    int left_size, right_size;
    cuju_engine_player left_team[11], right_team[11];
    int left_controllers[11], right_controllers[11];
    int left_goals, right_goals, game_mode, is_in_play, ball_owned_team, ball_owned_player, step;
};
}

namespace {
cuju::engine::ScenarioConfig FromScenario(const cuju_engine_scenario &source) {
    if (source.version != CUJU_ENGINE_VERSION)
        throw std::invalid_argument("expected the engine version " +
                                    std::to_string(CUJU_ENGINE_VERSION) + ", found " +
                                    std::to_string(source.version));
    cuju::engine::ScenarioConfig config;
    cuju::state::ScenarioState &scenario = config.scenario;
    for (int side = 0; side < 2; side++) {
        const int size = side ? source.right_size : source.left_size;
        if (size < 0 || size > cuju::state::MAXIMUM_TEAM_PLAYERS)
            throw cuju::engine::ScenarioRefused(std::string("the ") + (side ? "right" : "left") +
                                                " team has " + std::to_string(size) +
                                                " players, outside 0 to the " +
                                                std::to_string(cuju::state::MAXIMUM_TEAM_PLAYERS) +
                                                " of this build, where GRF crashes above 11");
        const cuju_engine_entry *entries = side ? source.right_team : source.left_team;
        cuju::state::FormationEntry *target = side ? scenario.rightTeam : scenario.leftTeam;
        for (int player = 0; player < size; player++) {
            const cuju_engine_entry &entry = entries[player];
            target[player].position =
                cuju::Vector3(entry.position[0], entry.position[1], entry.position[2]);
            target[player].startPosition = cuju::Vector3(
                entry.start_position[0], entry.start_position[1], entry.start_position[2]);
            target[player].role = entry.role;
            target[player].lazy = entry.lazy != 0;
            target[player].controllable = entry.controllable != 0;
        }
        (side ? scenario.rightTeamSize : scenario.leftTeamSize) = size;
    }
    scenario.ballPosition =
        cuju::Vector3(source.ball_position[0], source.ball_position[1], source.ball_position[2]);
    scenario.leftAgents = source.left_agents;
    scenario.rightAgents = source.right_agents;
    scenario.useMagnet = source.use_magnet != 0;
    scenario.offsides = source.offsides != 0;
    scenario.realTime = source.real_time != 0;
    scenario.gameEngineRandomSeed = source.game_engine_random_seed;
    scenario.reverseTeamProcessing = source.reverse_team_processing != 0;
    scenario.leftTeamDifficulty = source.left_team_difficulty;
    scenario.rightTeamDifficulty = source.right_team_difficulty;
    scenario.deterministic = source.deterministic != 0;
    scenario.endEpisodeOnScore = source.end_episode_on_score != 0;
    scenario.endEpisodeOnPossessionChange = source.end_episode_on_possession_change != 0;
    scenario.endEpisodeOnOutOfPlay = source.end_episode_on_out_of_play != 0;
    scenario.gameDuration = source.game_duration;
    scenario.secondHalf = source.second_half;
    scenario.controlAllPlayers = source.control_all_players != 0;
    config.cacheComputed = source.cache_computed != 0;
    config.cachedDynamicPlayerSelection = source.cache_dynamic_player_selection != 0;
    config.cachedControllableLeft = source.cache_controllable_left;
    config.cachedControllableRight = source.cache_controllable_right;
    return config;
}

void ToScenario(const cuju::engine::ScenarioConfig &config, cuju_engine_scenario &out) {
    const cuju::state::ScenarioState &scenario = config.scenario;
    for (int index = 0; index < 3; index++)
        out.ball_position[index] = scenario.ballPosition.coordinates[index];
    for (int side = 0; side < 2; side++) {
        const int size = side ? scenario.rightTeamSize : scenario.leftTeamSize;
        const cuju::state::FormationEntry *entries = side ? scenario.rightTeam : scenario.leftTeam;
        cuju_engine_entry *target = side ? out.right_team : out.left_team;
        for (int player = 0; player < size && target; player++) {
            for (int index = 0; index < 3; index++) {
                target[player].position[index] = entries[player].position.coordinates[index];
                target[player].start_position[index] =
                    entries[player].startPosition.coordinates[index];
            }
            target[player].role = entries[player].role;
            target[player].lazy = entries[player].lazy;
            target[player].controllable = entries[player].controllable;
        }
        (side ? out.right_size : out.left_size) = size;
    }
    out.left_agents = scenario.leftAgents;
    out.right_agents = scenario.rightAgents;
    out.use_magnet = scenario.useMagnet;
    out.offsides = scenario.offsides;
    out.real_time = scenario.realTime;
    out.game_engine_random_seed = scenario.gameEngineRandomSeed;
    out.reverse_team_processing = scenario.reverseTeamProcessing;
    out.left_team_difficulty = scenario.leftTeamDifficulty;
    out.right_team_difficulty = scenario.rightTeamDifficulty;
    out.deterministic = scenario.deterministic;
    out.end_episode_on_score = scenario.endEpisodeOnScore;
    out.end_episode_on_possession_change = scenario.endEpisodeOnPossessionChange;
    out.end_episode_on_out_of_play = scenario.endEpisodeOnOutOfPlay;
    out.game_duration = scenario.gameDuration;
    out.second_half = scenario.secondHalf;
    out.control_all_players = scenario.controlAllPlayers;
    out.cache_computed = config.cacheComputed;
    out.cache_dynamic_player_selection = config.cachedDynamicPlayerSelection;
    out.cache_controllable_left = config.cachedControllableLeft;
    out.cache_controllable_right = config.cachedControllableRight;
}
}  // namespace

extern "C" {
int cuju_engine_version() { return CUJU_ENGINE_VERSION; }

void *cuju_engine_create(const char *animations) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        return new cuju::engine::GameEngine(animations);
    } catch (const std::exception &exception) {
        EngineFail(exception);
        return nullptr;
    }
}

void cuju_engine_destroy(void *engine) { delete static_cast<cuju::engine::GameEngine *>(engine); }

// GRF: `start_game`.
int cuju_engine_start(void *engine) {
    try {
        Engine(engine).Start();
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

// GRF: `reset(scenario, animations)` with `ticks` physics steps per frame.
int cuju_engine_reset(void *engine, cuju_engine_scenario *scenario, int animations, int ticks) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        cuju::engine::ScenarioConfig config = FromScenario(*scenario);
        Engine(engine).Reset(config, animations != 0, ticks);
        for (int index = 0; index < 3; index++)
            scenario->ball_position[index] = config.scenario.ballPosition.coordinates[index];
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

int cuju_engine_set_config(void *engine, const cuju_engine_scenario *scenario) {
    try {
        Engine(engine).SetConfig(FromScenario(*scenario));
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

int cuju_engine_config(void *engine, cuju_engine_scenario *out) {
    try {
        ToScenario(Engine(engine).Config(), *out);
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

int cuju_engine_config_cache(void *engine, int *cache) {
    const cuju::engine::ScenarioConfig &config = Engine(engine).Config();
    cache[0] = config.cacheComputed;
    cache[1] = config.cachedDynamicPlayerSelection;
    cache[2] = config.cachedControllableLeft;
    cache[3] = config.cachedControllableRight;
    return 0;
}

// GRF: `perform_action`.
int cuju_engine_action(void *engine, int action, int left_team, int player) {
    try {
        Engine(engine).Action(action, left_team != 0, player);
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

// GRF: `sticky_action_state`, 0 or 1.
int cuju_engine_sticky(void *engine, int action, int left_team, int player) {
    try {
        return Engine(engine).Sticky(action, left_team != 0, player) ? 1 : 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

// GRF: `step` without rendering, `ticks` ticks and the end of the step.
int cuju_engine_step(void *engine, int ticks) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        Engine(engine).Step(ticks);
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

int cuju_engine_ticks(void *engine, int ticks, int count) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        Engine(engine).Ticks(ticks, count);
        return Engine(engine).InPlay() ? 1 : 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

int cuju_engine_finish_step(void *engine) {
    try {
        Engine(engine).FinishStep();
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

// GRF: `get_info` with `ticks` physics steps per frame.
int cuju_engine_info(void *engine, int ticks, cuju_engine_info *out) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        cuju::engine::EngineInfo info;
        Engine(engine).Info(ticks, info);
        auto copy = [](float *target, const cuju::Vector3 &vector) {
            for (int index = 0; index < 3; index++) target[index] = vector.coordinates[index];
        };
        copy(out->ball_position, info.ballPosition);
        copy(out->ball_direction, info.ballDirection);
        copy(out->ball_rotation, info.ballRotation);
        for (int side = 0; side < 2; side++) {
            const cuju::engine::TeamInfo &team = info.team[side];
            (side ? out->right_size : out->left_size) = team.count;
            cuju_engine_player *players = side ? out->right_team : out->left_team;
            for (int player = 0; player < team.count; player++) {
                const cuju::engine::PlayerInfo &source = team.players[player];
                copy(players[player].position, source.position);
                copy(players[player].direction, source.direction);
                players[player].tired_factor = source.tiredFactor;
                players[player].has_card = source.hasCard;
                players[player].is_active = source.isActive;
                players[player].designated_player = source.designatedPlayer;
                players[player].role = source.role;
            }
            int *controllers = side ? out->right_controllers : out->left_controllers;
            for (int index = 0; index < 11; index++)
                controllers[index] = index < cuju::simulation::MAXIMUM_TEAM_CONTROLLERS
                                         ? info.controlledPlayer[side][index]
                                         : -1;
        }
        out->left_goals = info.score[0];
        out->right_goals = info.score[1];
        out->game_mode = info.gameMode;
        out->is_in_play = info.isInPlay;
        out->ball_owned_team = info.ballOwnedTeam;
        out->ball_owned_player = info.ballOwnedPlayer;
        out->step = info.step;
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

// GRF: `GameEnv.state` and `GameEnv.waiting_for_game_count`.
int cuju_engine_game_state(void *engine) { return Engine(engine).GameState(); }
int cuju_engine_set_game_state(void *engine, int state) {
    Engine(engine).SetGameState(state);
    return 0;
}
int cuju_engine_waiting_count(void *engine) { return Engine(engine).WaitingForGameCount(); }
int cuju_engine_set_waiting_count(void *engine, int count) {
    Engine(engine).SetWaitingForGameCount(count);
    return 0;
}

// GRF: `get_state(pickle)`.
int cuju_engine_get_state(void *engine, const char *pickle, unsigned long long pickle_size,
                          const char **data, unsigned long long *size) {
    thread_local std::string bytes;
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        bytes = Engine(engine).GetState(std::string(pickle, pickle_size));
        *data = bytes.data();
        *size = bytes.size();
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

// GRF: `set_state(bytes)`.
int cuju_engine_set_state(void *engine, const char *bytes, unsigned long long size,
                          unsigned long long *pickle_offset, unsigned long long *pickle_size) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        const std::pair<size_t, size_t> pickle = Engine(engine).SetState(std::string(bytes, size));
        *pickle_offset = pickle.first;
        *pickle_size = pickle.second;
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

// GRF's tracker.
int cuju_engine_verify(void *engine, void *waiting, long long *position) {
    try {
        *position = Engine(engine).Verify(Engine(waiting));
        return *position < 0 ? 0 : 1;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

// GRF's render mode: an engine started with its graphics system while `game_config.render` is on.
int cuju_engine_set_render(void *engine, int render_mode) {
    Engine(engine).SetRenderMode(render_mode != 0);
    return 0;
}

// GRF: `PrepareRender` of `GameEnv::render`, what each render changes in the match.
int cuju_engine_prepare_render(void *engine) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        Engine(engine).PrepareRender();
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

unsigned int cuju_engine_render_state_version() { return cuju::render::RENDER_STATE_VERSION; }
unsigned long long cuju_engine_render_state_size() { return sizeof(cuju::render::RenderState); }

int cuju_engine_render_start(void *engine, void *state, unsigned long long size) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        CheckRenderStateSize(size);
        memcpy(state, &Engine(engine).StartRenderState(), sizeof(cuju::render::RenderState));
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

int cuju_engine_internal_render_state(void *state, unsigned long long size, int first) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        CheckRenderStateSize(size);
        const cuju::render::RenderState made =
            cuju::engine::GameEngine::InternalRenderState(first != 0);
        memcpy(state, &made, sizeof(cuju::render::RenderState));
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}

int cuju_engine_render_state(void *engine, void *state, unsigned long long size,
                             unsigned int *processed) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        CheckRenderStateSize(size);
        memcpy(state, &Engine(engine).ExtractRenderState(), sizeof(cuju::render::RenderState));
        if (processed) *processed = Engine(engine).StateProcessed();
        return 0;
    } catch (const std::exception &exception) {
        return EngineFail(exception);
    }
}
}
#if !defined(_MSC_VER)
#pragma GCC visibility pop
#endif
