// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#include "cuju/batch/memory_estimate.h"
#include "cuju/batch/stacked_state.h"
#include "cuju/floating_point.h"
#include "cuju/nvtx.h"

using cuju::batch::BatchEnvironmentConfiguration;

namespace {
thread_local std::string lastError;
thread_local unsigned long long *memoryEstimate = nullptr;
int Fail(const std::exception &exception) {
    lastError = exception.what();
    return -1;
}
}  // namespace

#if !defined(_MSC_VER)
#pragma GCC visibility push(default)
#endif
extern "C" {
const char *cuju_last_error() { return lastError.c_str(); }

const char *cuju_backend() { return CUJU_BACKEND_NAME; }

#define CUJU_CONFIGURATION_VERSION 3

struct cuju_configuration {
    int version;
    const char *const *assets;
    int asset_count;
    const int *assignment;
    const char *animations;
    int environment_count;
    unsigned long long seed;
    int action_set;
    int observation;
    int checkpoints;
    int numpy_legacy;
    int auto_reset;
    int threads;
    int ticks_per_step;
    int smm_width, smm_height;
    int stack;
    const long long *fixed_seeds;
    const int *agent_counts;
    // GRF's `episode_number`, the episode before the first, 0 there.
    int episode_number;
    unsigned long long first_index;
    void *(*allocate)(size_t bytes, void *context);
    void (*release)(void *pointer, void *context);
    void *allocator_context;
    int pin_threads;
    int numa;
    int page_aligned;
    // GRF's render mode of every environment, stored for the renderer, and GRF's `animations` flag
    // of `GameEnv::reset`, nonzero for the referee animations of every reset, until
    // `cuju_environment_set_render`.
    int render_mode;
    int referee_animations;
};

void *cuju_environment_create(const cuju_configuration *settings) {
    try {
        CUJU_NVTX_RANGE("create");
        cuju::DefaultFloatingPoint floatingPoint;
        if (settings->version != CUJU_CONFIGURATION_VERSION)
            throw std::invalid_argument("expected the configuration version " +
                                        std::to_string(CUJU_CONFIGURATION_VERSION) + ", found " +
                                        std::to_string(settings->version));
        if (settings->environment_count < 1)
            throw std::invalid_argument("expected at least one environment, found " +
                                        std::to_string(settings->environment_count));
        BatchEnvironmentConfiguration configuration;
        configuration.assets.assign(settings->assets, settings->assets + settings->asset_count);
        if (settings->assignment)
            configuration.assignment.assign(settings->assignment,
                                            settings->assignment + settings->environment_count);
        configuration.animations = settings->animations;
        configuration.environmentCount = settings->environment_count;
        configuration.seed = settings->seed;
        configuration.actionSet = settings->action_set;
        configuration.observation = settings->observation;
        configuration.checkpoints = settings->checkpoints != 0;
        configuration.numpyLegacy = settings->numpy_legacy != 0;
        configuration.autoReset = settings->auto_reset != 0;
        configuration.threads = settings->threads;
        configuration.ticksPerStep = settings->ticks_per_step;
        configuration.smmWidth = settings->smm_width;
        configuration.smmHeight = settings->smm_height;
        configuration.stack = settings->stack;
        if (settings->fixed_seeds)
            configuration.fixedSeeds.assign(settings->fixed_seeds,
                                            settings->fixed_seeds + settings->environment_count);
        if (settings->agent_counts)
            configuration.agentCounts.assign(settings->agent_counts,
                                             settings->agent_counts + 2 * settings->asset_count);
        configuration.episodeNumber = settings->episode_number;
        configuration.firstIndex = (int64_t)std::min(settings->first_index, 1ULL << 62);
        configuration.allocate = settings->allocate;
        configuration.release = settings->release;
        configuration.allocatorContext = settings->allocator_context;
        configuration.pinThreads = settings->pin_threads != 0;
        configuration.numa = settings->numa != 0;
        configuration.pageAligned = settings->page_aligned != 0;
        configuration.renderMode = settings->render_mode;
        configuration.refereeAnimations = settings->referee_animations != 0;
        lastError.clear();
        if (memoryEstimate) {
            using Backend =
                std::remove_reference<decltype(std::declval<BatchEnvironment &>().Device())>::type;
            *memoryEstimate = cuju::batch::EstimateMemory<Backend>(configuration);
            return nullptr;
        }
        return new BatchEnvironment(configuration);
    } catch (const std::exception &exception) {
        Fail(exception);
        return nullptr;
    }
}

int cuju_memory_estimate(const cuju_configuration *settings, unsigned long long *bytes) {
    memoryEstimate = bytes;
    cuju_environment_create(settings);
    memoryEstimate = nullptr;
    return lastError.empty() ? 0 : -1;
}

int cuju_local_memory_bytes(unsigned long long *bytes) {
    try {
        using Backend =
            std::remove_reference<decltype(std::declval<BatchEnvironment &>().Device())>::type;
        *bytes = Backend::LocalMemoryBytes();
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

unsigned long long cuju_environment_memory_bytes(void *environment) {
    return static_cast<BatchEnvironment *>(environment)->MemoryBytes();
}

void cuju_environment_destroy(void *environment) {
    delete static_cast<BatchEnvironment *>(environment);
}

int cuju_environment_count(void *environment) {
    return static_cast<BatchEnvironment *>(environment)->EnvironmentCount();
}

int cuju_environment_threads(void *environment) {
    return static_cast<BatchEnvironment *>(environment)->Threads();
}

int cuju_environment_agents(void *environment, int *left, int *right) {
    const BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
    for (int index = 0; index < batch->EnvironmentCount(); index++) {
        if (left) left[index] = batch->LeftAgents(index);
        if (right) right[index] = batch->RightAgents(index);
    }
    return batch->Agents();
}

int cuju_environment_players(void *environment, int *left, int *right) {
    const BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
    for (int index = 0; index < batch->EnvironmentCount(); index++) {
        if (left) left[index] = batch->LeftPlayers(index);
        if (right) right[index] = batch->RightPlayers(index);
    }
    return batch->Players();
}

int cuju_environment_set_difficulty(void *environment, const float *difficulty) {
    try {
        static_cast<BatchEnvironment *>(environment)->SetDifficulty(difficulty);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

void cuju_environment_difficulty(void *environment, float *difficulty) {
    const BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
    for (int index = 0; index < batch->EnvironmentCount(); index++)
        for (int side = 0; side < 2; side++)
            difficulty[index * 2 + side] = batch->Difficulty(index, side);
}

int cuju_environment_set_render(void *environment, const unsigned char *mask, int render_mode,
                                int referee_animations) {
    try {
        static_cast<BatchEnvironment *>(environment)
            ->SetRender(mask, render_mode, referee_animations != 0);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_reset(void *environment, const unsigned int *seeds, const int *reverse,
                           void *stream) {
    try {
        CUJU_NVTX_RANGE("reset");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->Reset(seeds, reverse, static_cast<BatchEnvironment::Stream>(stream));
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_reset_masked(void *environment, const unsigned char *mask, int mask_on_device,
                                  const int64_t *seeds, const int *reverse, void *stream,
                                  int synchronize) {
    try {
        CUJU_NVTX_RANGE("reset_masked");
        cuju::DefaultFloatingPoint floatingPoint;
        BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
        const BatchEnvironment::Stream batchStream = static_cast<BatchEnvironment::Stream>(stream);
        batch->ResetMasked(mask, mask_on_device != 0, seeds, reverse, batchStream);
        if (synchronize) batch->Device().Synchronize(batchStream);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_reset_done(void *environment, void *stream) {
    try {
        CUJU_NVTX_RANGE("reset_done");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->ResetDone(static_cast<BatchEnvironment::Stream>(stream));
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_step(void *environment, const int *actions, int on_device, void *stream,
                          int synchronize) {
    try {
        CUJU_NVTX_RANGE("step");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->Step(actions, on_device != 0, static_cast<BatchEnvironment::Stream>(stream),
                   synchronize != 0);
        return 0;
    } catch (const std::out_of_range &exception) {
        Fail(exception);
        return -2;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_step_async(void *environment, const int *actions, int on_device,
                                void *stream) {
    try {
        CUJU_NVTX_RANGE("step_async");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->StepAsync(actions, on_device != 0, static_cast<BatchEnvironment::Stream>(stream));
        return 0;
    } catch (const std::out_of_range &exception) {
        Fail(exception);
        return -2;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_step_wait(void *environment) {
    try {
        CUJU_NVTX_RANGE("step_wait");
        static_cast<BatchEnvironment *>(environment)->StepWait();
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_synchronize(void *environment, void *stream) {
    try {
        static_cast<BatchEnvironment *>(environment)
            ->Device()
            .Synchronize(static_cast<BatchEnvironment::Stream>(stream));
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

const char *cuju_environment_get_state(void *environment, int index, unsigned long long *size,
                                       int *episode) {
    thread_local std::string bytes;
    try {
        CUJU_NVTX_RANGE("get_state");
        cuju::DefaultFloatingPoint floatingPoint;
        cuju::simulation::EnvironmentCore core;
        cuju::batch::EnvironmentSlot slot;
        BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
        bytes = batch->GetState(index, core, slot);
        *size = bytes.size();
        episode[0] = core.previousScoreDifference;
        episode[1] = core.previousGameMode;
        episode[2] = core.previousBallOwnedTeam;
        for (int agent = 0; agent < batch->Agents(); agent++)
            episode[3 + agent] = slot.collected[agent];
        return bytes.data();
    } catch (const std::exception &exception) {
        Fail(exception);
        return nullptr;
    }
}

int cuju_environment_set_state(void *environment, int index, const char *bytes,
                               unsigned long long size, const int *episode) {
    try {
        CUJU_NVTX_RANGE("set_state");
        cuju::DefaultFloatingPoint floatingPoint;
        static_cast<BatchEnvironment *>(environment)
            ->SetState(index, std::string(bytes, size), episode[0], episode[1], episode[2],
                       episode + 3);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

const char *cuju_environment_get_states(void *environment, unsigned long long *sizes,
                                        int *episodes) {
    thread_local std::string joined;
    try {
        CUJU_NVTX_RANGE("get_states");
        cuju::DefaultFloatingPoint floatingPoint;
        BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
        std::vector<std::string> states;
        std::vector<cuju::simulation::EnvironmentCore> cores;
        std::vector<cuju::batch::EnvironmentSlot> slots;
        batch->GetStates(states, cores, slots);
        const int stride = 3 + cuju::batch::MAXIMUM_AGENTS;
        joined.clear();
        for (size_t index = 0; index < states.size(); index++) {
            sizes[index] = states[index].size();
            joined += states[index];
            int *episode = episodes + index * stride;
            episode[0] = cores[index].previousScoreDifference;
            episode[1] = cores[index].previousGameMode;
            episode[2] = cores[index].previousBallOwnedTeam;
            for (int agent = 0; agent < batch->Agents(); agent++)
                episode[3 + agent] = slots[index].collected[agent];
        }
        return joined.data();
    } catch (const std::exception &exception) {
        Fail(exception);
        return nullptr;
    }
}

int cuju_environment_set_states(void *environment, const char *bytes,
                                const unsigned long long *sizes, int count, const int *episodes) {
    try {
        CUJU_NVTX_RANGE("set_states");
        cuju::DefaultFloatingPoint floatingPoint;
        BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
        std::vector<std::string> states;
        for (int index = 0; index < count; index++) {
            states.emplace_back(bytes, sizes[index]);
            bytes += sizes[index];
        }
        batch->SetStates(0, count, states.data(), episodes, 3 + cuju::batch::MAXIMUM_AGENTS);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_roles(void *environment, int index, int *roles, int capacity) {
    try {
        int32_t written[cuju::state::MAXIMUM_PLAYERS];
        const int count = static_cast<BatchEnvironment *>(environment)->Roles(index, written);
        if (count > capacity)
            throw std::invalid_argument("expected room for " + std::to_string(count) +
                                        " roles, found " + std::to_string(capacity));
        for (int player = 0; player < count; player++) roles[player] = written[player];
        return count;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_set_stack(void *environment, int index, const char *name, const void *frames,
                               unsigned long long size) {
    try {
        cuju::batch::SetStackedFrames(*static_cast<BatchEnvironment *>(environment), index, name,
                                      frames, size);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

void *cuju_environment_buffer(void *environment, const char *name) {
    const cuju::batch::EnvironmentBuffers &buffers =
        static_cast<BatchEnvironment *>(environment)->Buffers();
    const std::string bufferName = name;
    if (bufferName == "simple115") return buffers.simple115;
    if (bufferName == "final_simple115") return buffers.finalSimple115;
    if (bufferName == "smm") return buffers.smm;
    if (bufferName == "final_smm") return buffers.finalSmm;
    if (bufferName == "reward") return buffers.reward;
    if (bufferName == "done") return buffers.done;
    if (bufferName == "score_reward") return buffers.scoreReward;
    if (bufferName == "raw") return buffers.raw;
    if (bufferName == "final_raw") return buffers.finalRaw;
    return nullptr;
}

const char *cuju_environment_raw_field(void *environment, int field, int *type, int *dimensions,
                                       unsigned long long *offset) {
    const cuju::batch::EnvironmentBuffers &buffers =
        static_cast<BatchEnvironment *>(environment)->Buffers();
    if (!buffers.raw || field < 0 || field >= cuju::batch::RAW_FIELD_COUNT) return nullptr;
    const cuju::batch::RawArray &array = buffers.rawLayout.arrays[field];
    *type = array.type;
    dimensions[0] = array.dimensions[0];
    dimensions[1] = array.dimensions[1];
    *offset = array.offset;
    return cuju::batch::RAW_FIELD_NAMES[field];
}

int cuju_environment_episode(void *environment, int index, long long *values, float *difficulty) {
    try {
        BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
        if (index < 0 || index >= batch->EnvironmentCount())
            throw std::invalid_argument("expected an environment from 0 to " +
                                        std::to_string(batch->EnvironmentCount() - 1) + ", found " +
                                        std::to_string(index));
        const cuju::batch::EnvironmentBuffers &buffers = batch->Buffers();
        cuju::batch::EnvironmentSlot slot;
        batch->Device().ToHost(&slot, buffers.slots + index, sizeof(slot));
        cuju::simulation::Observation observation;
        batch->Device().ToHost(&observation, &buffers.cores[index].observation,
                               sizeof(observation));
        const cuju::batch::Assets &bundle = batch->Bundles()[slot.bundle];
        const cuju::state::ScenarioState &scenario = bundle.variants[0].world->serialized.scenario;
        values[0] = slot.seed;
        values[1] = slot.reverse;
        values[2] = slot.episode;
        values[3] = slot.episodeOffset;
        values[4] = slot.bundle;
        values[5] = observation.step;
        values[6] = slot.fixedSeed;
        values[7] = scenario.deterministic;
        values[8] = batch->HostContext(slot.variant).ticksPerStep;
        values[9] = slot.leftAgents;
        values[10] = slot.rightAgents;
        values[11] = scenario.leftTeamSize;
        values[12] = scenario.rightTeamSize;
        for (int team = 0; team < 2; team++) {
            values[13 + team] = observation.team[team].designatedPlayer;
            values[15 + team] = slot.finalDesignated[team];
        }
        if (difficulty) {
            const cuju::state::ScenarioState &played = buffers.worlds[index].serialized.scenario;
            batch->Device().ToHost(&difficulty[0], &played.leftTeamDifficulty, sizeof(float));
            batch->Device().ToHost(&difficulty[1], &played.rightTeamDifficulty, sizeof(float));
            const cuju::state::ScenarioState &own =
                bundle.variants[slot.variant % 4].world->serialized.scenario;
            difficulty[2] = own.leftTeamDifficulty;
            difficulty[3] = own.rightTeamDifficulty;
        }
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

const char *cuju_environment_scenario(void *environment, int bundle) {
    const BatchEnvironment *batch = static_cast<BatchEnvironment *>(environment);
    if (bundle < 0 || bundle >= (int)batch->Bundles().size()) return nullptr;
    return batch->Bundles()[bundle].scenario.c_str();
}

unsigned long long cuju_environment_render_event_size() { return sizeof(cuju::batch::RenderEvent); }

int cuju_environment_enable_render_events(void *environment, int capacity) {
    try {
        static_cast<BatchEnvironment *>(environment)->EnableRenderEvents(capacity);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_render_events(void *environment, int index, void *events, int capacity,
                                   int *count) {
    try {
        *count =
            static_cast<BatchEnvironment *>(environment)
                ->RenderEventsOf(index, static_cast<cuju::batch::RenderEvent *>(events), capacity);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

int cuju_environment_render_states(void *environment, const int *environments, int count,
                                   void *states, unsigned long long size, unsigned int *processed) {
    try {
        cuju::DefaultFloatingPoint floatingPoint;
        if (count < 0 || size != (unsigned long long)count * sizeof(cuju::render::RenderState))
            throw std::invalid_argument("expected " + std::to_string(count) + " RenderStates of " +
                                        std::to_string(sizeof(cuju::render::RenderState)) +
                                        " bytes, version " +
                                        std::to_string(cuju::render::RENDER_STATE_VERSION) +
                                        ", found " + std::to_string(size) + " bytes");
        static_cast<BatchEnvironment *>(environment)
            ->RenderStates(environments, count, static_cast<cuju::render::RenderState *>(states),
                           processed);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}
}
#if !defined(_MSC_VER)
#pragma GCC visibility pop
#endif
