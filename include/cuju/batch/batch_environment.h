// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "../animation/animation_data.h"
#include "../animation/animation_derived.h"
#include "../animation/skeleton.h"
#include "batch_core.h"
#include "render_states.h"
#include "scene_states.h"
#include "state_difference.h"
#include "tables.h"

namespace cuju {
namespace batch {
template <class Backend>
class BackendBatchEnvironment {
public:
    using Stream = typename Backend::Stream;

    explicit BackendBatchEnvironment(const BatchEnvironmentConfiguration &configuration)
        : configuration_(configuration) {
        if ((configuration.observation & OBSERVATION_SIMPLE115) &&
            (configuration.observation & OBSERVATION_SIMPLE115_VERSION2))
            throw std::invalid_argument("simple115 and simple115v2 are exclusive");
        if (configuration.assets.empty()) throw std::invalid_argument("expected an asset bundle");
        // GRF: a step of no ticks never brings the match into play.
        if (configuration.ticksPerStep < 1)
            throw std::invalid_argument("expected at least one tick in a step, found " +
                                        std::to_string(configuration.ticksPerStep));
        if (!configuration.fixedSeeds.empty()) {
            if ((int)configuration.fixedSeeds.size() != configuration.environmentCount)
                throw std::invalid_argument("expected one fixed seed for each environment, found " +
                                            std::to_string(configuration.fixedSeeds.size()));
            // GRF: the scenario holds its seed as a 32-bit `unsigned int`.
            for (int64_t seed : configuration.fixedSeeds)
                if (seed < -1 || seed > 4294967295LL)
                    throw std::invalid_argument(
                        "expected a fixed seed from 0 to 4294967295, found " +
                        std::to_string(seed));
        }
        // GRF: the index and the seed share one key, so an index of 2**32 would be another seed's.
        if (configuration.firstIndex < 0 ||
            configuration.firstIndex + configuration.environmentCount > (int64_t(1) << 32))
            throw std::invalid_argument(
                "expected a first index from 0 to 2**32 minus the environment count, found " +
                std::to_string(configuration.firstIndex));
        if (configuration.episodeNumber < 0)
            throw std::invalid_argument("expected an episode number of at least 0, found " +
                                        std::to_string(configuration.episodeNumber));
        // GRF: `FrameStack` holds four observations.
        if (configuration.stack != 1 && configuration.stack != 4)
            throw std::invalid_argument("expected a stack of 1 or 4 observations, found " +
                                        std::to_string(configuration.stack));
        if (configuration.stack > 1 && (configuration.observation & OBSERVATION_RAW) &&
            !(configuration.observation & ~OBSERVATION_RAW))
            throw std::invalid_argument("the raw representation has no stack");
        if (configuration.smmWidth < 1 || configuration.smmHeight < 1)
            throw std::invalid_argument("expected an SMM of at least one cell, found " +
                                        std::to_string(configuration.smmWidth) + " by " +
                                        std::to_string(configuration.smmHeight));
        const int count = configuration.environmentCount;
        std::vector<int32_t> assignment = configuration.assignment;
        if (assignment.empty()) assignment.assign(count, 0);
        if ((int)assignment.size() != count)
            throw std::invalid_argument("expected one bundle for each of the " +
                                        std::to_string(count) + " environments, found " +
                                        std::to_string(assignment.size()));
        for (int32_t bundle : assignment)
            if (bundle < 0 || bundle >= (int)configuration.assets.size())
                throw std::invalid_argument("expected a bundle from 0 to " +
                                            std::to_string(configuration.assets.size() - 1) +
                                            ", found " + std::to_string(bundle));
        const std::vector<int32_t> &agentCounts = configuration.agentCounts;
        if (!agentCounts.empty() && agentCounts.size() != 2 * configuration.assets.size())
            throw std::invalid_argument("expected a left and a right agent count for each of the " +
                                        std::to_string(configuration.assets.size()) +
                                        " bundles, found " + std::to_string(agentCounts.size()) +
                                        " counts");
        for (size_t bundle = 0; bundle < configuration.assets.size(); bundle++) {
            const int left = agentCounts.empty() ? -1 : agentCounts[2 * bundle];
            const int right = agentCounts.empty() ? -1 : agentCounts[2 * bundle + 1];
            bundles_.push_back(LoadAssets(configuration.assets[bundle], left, right));
        }
        store_ = animation::ParseAnimationTables(ReadFileBytes(configuration.animations));
        for (const animation::Animation &animation : store_->animations)
            if (animation.frameCount > state::MAXIMUM_ANIMATION_POSITIONS)
                throw std::invalid_argument("expected animations of at most " +
                                            std::to_string(state::MAXIMUM_ANIMATION_POSITIONS) +
                                            " frames, found " +
                                            std::to_string(animation.frameCount));
        derived_ = animation::DeriveAll(*store_);
        backend_.Configure(configuration);
        auto &tables = backend_.Tables();
        UploadTables(tables, *store_, derived_, bundles_[0].variants[0].skeletons,
                     bundles_[0].variants[0].staticData);
        std::vector<simulation::SimulationContext> contexts;
        std::vector<const simulation::World *> templates;
        int agents = 0, players = 0;
        for (const Assets &bundle : bundles_) {
            if (bundle.leftAgents < 0 || bundle.rightAgents < 0 ||
                bundle.leftAgents + bundle.rightAgents > MAXIMUM_AGENTS)
                throw std::runtime_error("expected at most " + std::to_string(MAXIMUM_AGENTS) +
                                         " agents in a bundle, found " +
                                         std::to_string(bundle.leftAgents + bundle.rightAgents));
            agents = std::max(agents, bundle.leftAgents + bundle.rightAgents);
            for (const AssetVariant &variant : bundle.variants) {
                simulation::SimulationContext context = tables.context;
                context.staticData = tables.Upload(&variant.staticData, 1);
                context.skeletons = tables.Upload(variant.skeletons);
                context.ticksPerStep = configuration.ticksPerStep;
                contexts.push_back(context);
                templates.push_back(tables.Upload(variant.world.get(), 1));
                const state::ScenarioState &scenario = variant.world->serialized.scenario;
                players = std::max({players, scenario.leftTeamSize, scenario.rightTeamSize});
                // GRF: the core asserts that a scenario with `control_all_players` gives each side
                // no agent or one for every controllable player.
                if (scenario.controlAllPlayers &&
                    simulation::DynamicPlayerSelection(*variant.world)) {
                    int left = 0, right = 0;
                    for (int player = 0; player < scenario.leftTeamSize; player++)
                        left += scenario.leftTeam[player].controllable ? 1 : 0;
                    for (int player = 0; player < scenario.rightTeamSize; player++)
                        right += scenario.rightTeam[player].controllable ? 1 : 0;
                    throw std::invalid_argument(
                        "expected 0 or " + std::to_string(left) + " left agents and 0 or " +
                        std::to_string(right) +
                        " right agents, all the controllable players of a side, for a scenario "
                        "with control_all_players, found " +
                        std::to_string(scenario.leftAgents) + " and " +
                        std::to_string(scenario.rightAgents));
                }
                // GRF: a player keeps one position per second of play between kick-offs, without a
                // limit.
                constexpr int HISTORY =
                    decltype(state::PlayerBaseState<
                             state::AIControllerState>::positionHistoryPerSecond)::CAPACITY;
                const int64_t seconds =
                    (int64_t)scenario.gameDuration * configuration.ticksPerStep / 100 + 1;
                if (seconds > HISTORY)
                    throw std::invalid_argument(
                        "expected a match of at most " + std::to_string(HISTORY) +
                        " seconds of play, " + "the position history of a player, found " +
                        std::to_string(scenario.gameDuration) + " steps of " +
                        std::to_string(configuration.ticksPerStep) + " ticks");
            }
        }
        variants_.contexts = tables.Upload(contexts);
        variants_.templates = tables.Upload(templates);
        players_ = players;
        buffers_.environmentCount = count;
        buffers_.agents = agents;
        buffers_.actionSet = configuration.actionSet;
        buffers_.observation = configuration.observation;
        buffers_.checkpoints = configuration.checkpoints;
        buffers_.numpyLegacy = configuration.numpyLegacy;
        buffers_.worlds = Allocate<simulation::World>(count);
        buffers_.cores = Allocate<simulation::EnvironmentCore>(count);
        buffers_.slots = Allocate<EnvironmentSlot>(count);
        buffers_.reward = Allocate<float>((size_t)count * agents);
        buffers_.done = Allocate<uint8_t>(count);
        buffers_.scoreReward = Allocate<int32_t>(count);
        const size_t stack = (size_t)configuration.stack;
        buffers_.stack = configuration.stack;
        if (configuration.observation & (OBSERVATION_SIMPLE115 | OBSERVATION_SIMPLE115_VERSION2)) {
            buffers_.simple115 = Allocate<float>((size_t)count * agents * SIMPLE115 * stack);
            buffers_.finalSimple115 = Allocate<float>((size_t)count * agents * SIMPLE115 * stack);
        }
        buffers_.smmWidth = configuration.smmWidth;
        buffers_.smmHeight = configuration.smmHeight;
        if (configuration.observation & OBSERVATION_SMM) {
            const size_t smmBytes = (size_t)configuration.smmWidth * configuration.smmHeight *
                                    simulation::SMM_LAYERS * stack;
            const size_t marks = (size_t)count * agents * (1 + simulation::SMM_MARKS) * stack;
            buffers_.smm = Allocate<uint8_t>((size_t)count * agents * smmBytes);
            buffers_.finalSmm = Allocate<uint8_t>((size_t)count * agents * smmBytes);
            buffers_.smmMarks = Allocate<int32_t>(marks);
            buffers_.finalSmmMarks = Allocate<int32_t>(marks);
        }
        if (configuration.observation & OBSERVATION_RAW) {
            buffers_.rawLayout =
                MakeRawLayout((size_t)count * agents, players,
                              simulation::StickyActionCount(configuration.actionSet));
            buffers_.raw = Allocate<uint8_t>(buffers_.rawLayout.bytes);
            buffers_.finalRaw = Allocate<uint8_t>(buffers_.rawLayout.bytes);
        }
        if (!Backend::HOST_MEMORY) actions_ = Allocate<int32_t>((size_t)count * agents);
        if (!Backend::HOST_MEMORY) buffers_.workAreas = Allocate<WorkArea>(count);
        outputMask_ = Allocate<uint8_t>(count);
        std::vector<EnvironmentSlot> slots(count);
        std::vector<uint8_t> agentMask((size_t)count * agents);
        for (int environment = 0; environment < count; environment++) {
            EnvironmentSlot &slot = slots[environment];
            slot = EnvironmentSlot{};
            SeedEpisodeGenerator(slot.generator, configuration.seed,
                                 (uint32_t)(configuration.firstIndex + environment));
            slot.fixedSeed =
                configuration.fixedSeeds.empty() ? -1 : configuration.fixedSeeds[environment];
            slot.episodeOffset = configuration.episodeNumber;
            slot.renderMode = configuration.renderMode;
            slot.refereeAnimations = configuration.refereeAnimations ? 1 : 0;
            slot.bundle = assignment[environment];
            slot.leftAgents = bundles_[slot.bundle].leftAgents;
            slot.rightAgents = bundles_[slot.bundle].rightAgents;
            for (int agent = 0; agent < slot.leftAgents + slot.rightAgents; agent++)
                agentMask[(size_t)environment * agents + agent] = 1;
        }
        backend_.ToDevice(buffers_.slots, slots.data(), sizeof(EnvironmentSlot) * count);
        buffers_.agentMask = Allocate<uint8_t>(agentMask.size());
        backend_.ToDevice(buffers_.agentMask, agentMask.data(), agentMask.size());
        hostSlots_ = slots;
        started_.assign(count, false);
        unstarted_ = count;
        backend_.Reserve(buffers_, variants_);
    }
    BackendBatchEnvironment(const BackendBatchEnvironment &) = delete;
    BackendBatchEnvironment &operator=(const BackendBatchEnvironment &) = delete;

    int EnvironmentCount() const { return buffers_.environmentCount; }
    int Agents() const { return buffers_.agents; }
    int LeftAgents(int environment) const { return hostSlots_[environment].leftAgents; }
    int RightAgents(int environment) const { return hostSlots_[environment].rightAgents; }
    int Players() const { return players_; }
    int LeftPlayers(int environment) const { return Scenario(environment).leftTeamSize; }
    int RightPlayers(int environment) const { return Scenario(environment).rightTeamSize; }
    int Bundle(int environment) const { return hostSlots_[environment].bundle; }
    const std::vector<Assets> &Bundles() const { return bundles_; }
    const EnvironmentBuffers &Buffers() const { return buffers_; }
    Backend &Device() { return backend_; }
    int Threads() const { return backend_.Threads(); }
    size_t MemoryBytes() const { return backend_.AllocatedBytes(); }

    void Reset(const uint32_t *seeds = nullptr, const int32_t *reverse = nullptr,
               Stream stream = Stream()) {
        const int64_t *deviceSeeds = nullptr;
        const int32_t *deviceReverse = nullptr;
        std::vector<int64_t> wideSeeds;
        if (seeds) {
            wideSeeds.assign(seeds, seeds + buffers_.environmentCount);
            deviceSeeds =
                Scratch<int64_t>(wideSeeds.data(), buffers_.environmentCount, scratchSeeds_);
        }
        if (reverse) {
            CheckReverse(reverse);
            deviceReverse = Scratch<int32_t>(reverse, buffers_.environmentCount, scratchReverse_);
        }
        backend_.Reset(buffers_, variants_, nullptr, deviceSeeds, deviceReverse, stream);
        backend_.Synchronize(stream);
        started_.assign(buffers_.environmentCount, true);
        unstarted_ = 0;
    }

    void ResetMasked(const uint8_t *mask, bool maskOnDevice, const int64_t *seeds,
                     const int32_t *reverse, Stream stream = Stream()) {
        const int count = buffers_.environmentCount;
        if (seeds) CheckSeeds(seeds);
        if (reverse) CheckReverse(reverse);
        const uint8_t *deviceMask =
            mask && !maskOnDevice ? AsyncScratch<uint8_t>(mask, count, maskedMask_, stream) : mask;
        const int64_t *deviceSeeds =
            seeds ? AsyncScratch<int64_t>(seeds, count, maskedSeeds_, stream) : nullptr;
        const int32_t *deviceReverse =
            reverse ? AsyncScratch<int32_t>(reverse, count, maskedReverse_, stream) : nullptr;
        if (unstarted_ > 0) {
            std::vector<uint8_t> started(count, 1);
            if (mask && maskOnDevice && !Backend::HOST_MEMORY) {
                backend_.Synchronize(stream);
                backend_.ToHost(started.data(), mask, count);
            } else if (mask) {
                started.assign(mask, mask + count);
            }
            for (int environment = 0; environment < count; environment++)
                if (started[environment] && !started_[environment]) {
                    started_[environment] = true;
                    unstarted_--;
                }
        }
        backend_.Reset(buffers_, variants_, deviceMask, deviceSeeds, deviceReverse, stream);
    }

    void Step(const int32_t *actions, bool actionsOnDevice, Stream stream = Stream(),
              bool synchronize = true) {
        CheckStarted();
        if (!actionsOnDevice || Backend::HOST_MEMORY) CheckActions(actions);
        const int32_t *deviceActions = actions;
        if (!actionsOnDevice && !Backend::HOST_MEMORY) {
            backend_.ToDeviceAsync(actions_, actions,
                                   sizeof(int32_t) * buffers_.environmentCount * Agents(), stream);
            deviceActions = actions_;
        }
        backend_.Step(buffers_, variants_, deviceActions, configuration_.autoReset, stream);
        if (synchronize) backend_.Synchronize(stream);
    }

    void ResetDone(Stream stream = Stream()) {
        CheckStarted();
        backend_.Reset(buffers_, variants_, buffers_.done, nullptr, nullptr, stream);
    }

    void StepAsync(const int32_t *actions, bool actionsOnDevice, Stream stream = Stream()) {
        if constexpr (Backend::HOST_MEMORY) {
            CheckStarted();
            CheckActions(actions);
            const size_t count = (size_t)buffers_.environmentCount * Agents();
            if (!asyncActions_) asyncActions_ = Allocate<int32_t>(count);
            backend_.Copy(asyncActions_, actions, sizeof(int32_t) * count);
            backend_.Launch([this, buffers = buffers_, variants = variants_,
                             autoReset = configuration_.autoReset, copied = asyncActions_] {
                backend_.Step(buffers, variants, copied, autoReset, Stream());
            });
        } else {
            Step(actions, actionsOnDevice, stream, false);
        }
    }
    void StepWait() {
        if constexpr (Backend::HOST_MEMORY) backend_.Wait();
    }

    void SetRender(const uint8_t *mask, int renderMode, bool refereeAnimations) {
        const int count = buffers_.environmentCount;
        std::vector<EnvironmentSlot> slots(count);
        backend_.ToHost(slots.data(), buffers_.slots, sizeof(EnvironmentSlot) * count);
        for (int environment = 0; environment < count; environment++) {
            if (mask && !mask[environment]) continue;
            slots[environment].renderMode = renderMode;
            slots[environment].refereeAnimations = refereeAnimations ? 1 : 0;
        }
        backend_.ToDevice(buffers_.slots, slots.data(), sizeof(EnvironmentSlot) * count);
    }

    void SetDifficulty(const float *difficulty) {
        const size_t count = (size_t)buffers_.environmentCount * 2;
        for (size_t index = 0; index < count; index++)
            if (!(difficulty[index] <= 1))
                throw std::invalid_argument("expected a difficulty from 0 to 1, found " +
                                            std::to_string(difficulty[index]));
        if (!buffers_.teamDifficulty) buffers_.teamDifficulty = Allocate<float>(count);
        backend_.ToDevice(const_cast<float *>(buffers_.teamDifficulty), difficulty,
                          count * sizeof(float));
        hostDifficulty_.assign(difficulty, difficulty + count);
    }

    float Difficulty(int environment, int side) const {
        if (!hostDifficulty_.empty() && hostDifficulty_[environment * 2 + side] >= 0.0f)
            return hostDifficulty_[environment * 2 + side];
        // GRF: `episode_number` is the episode before the first, so its parity is the first's.
        const state::ScenarioState &scenario =
            Variant(Bundle(environment) * 4 + (configuration_.episodeNumber & 1))
                .world->serialized.scenario;
        return side == 0 ? scenario.leftTeamDifficulty : scenario.rightTeamDifficulty;
    }

    int Roles(int environment, int32_t *roles) {
        CheckEnvironment(environment);
        const simulation::World &world = buffers_.worlds[environment];
        int playerCount = 0;
        int playerTeamID[state::MAXIMUM_PLAYERS];
        int8_t formationOverride[state::MAXIMUM_PLAYERS];
        EnvironmentSlot slot;
        backend_.ToHost(&playerCount, &world.serialized.match.playerCount, sizeof(playerCount));
        backend_.ToHost(playerTeamID, world.playerTeamID, sizeof(playerTeamID));
        backend_.ToHost(formationOverride, world.formationOverride, sizeof(formationOverride));
        backend_.ToHost(&slot, buffers_.slots + environment, sizeof(slot));
        const simulation::StaticData &staticData = *HostContext(slot.variant).staticData;
        int count = 0;
        for (int teamID = 0; teamID < 2; teamID++)
            for (int player = 0; player < playerCount; player++) {
                if (playerTeamID[player] != teamID) continue;
                const int source = simulation::FormationSource(formationOverride[player], player);
                roles[count++] = staticData.formation[source].role;
            }
        return count;
    }

    void RenderStates(const int32_t *indices, int count, render::RenderState *out,
                      uint32_t *processed) {
        const int environments = buffers_.environmentCount;
        for (int output = 0; output < count; output++) CheckEnvironment(indices[output]);
        if (count <= 0) return;
        if (!renderStates_) {
            renderStates_ = Allocate<render::RenderState>(environments);
            renderEpisodes_ = Allocate<int32_t>(environments);
            backend_.Fill(renderEpisodes_, 0xff, sizeof(int32_t) * environments);
            MirroredTable();
        }
        if (count > renderCapacity_) {
            renderIndices_ = Allocate<int32_t>(count);
            renderOut_ = Allocate<render::RenderState>(count);
            renderProcessed_ = Allocate<uint32_t>(count);
            renderCapacity_ = count;
        }
        backend_.ToDevice(renderIndices_, indices, sizeof(int32_t) * count);
        RenderBuffers render{renderStates_, renderEpisodes_, animationMirrored_, renderIndices_,
                             count,         renderOut_,      renderProcessed_};
        backend_.RenderStates(buffers_, variants_, render);
        backend_.ToHost(out, renderOut_, sizeof(render::RenderState) * count);
        backend_.ToHost(processed, renderProcessed_, sizeof(uint32_t) * count);
    }

    // The scene of the chosen environments, read from their worlds as they are.
    void SceneStates(const int32_t *indices, int count, scene::SceneState *out) {
        for (int output = 0; output < count; output++) CheckEnvironment(indices[output]);
        if (count <= 0) return;
        if (count > sceneCapacity_) {
            sceneIndices_ = Allocate<int32_t>(count);
            sceneOut_ = Allocate<scene::SceneState>(count);
            sceneCapacity_ = count;
        }
        backend_.ToDevice(sceneIndices_, indices, sizeof(int32_t) * count);
        SceneBuffers scene{MirroredTable(), sceneIndices_, count, sceneOut_};
        backend_.SceneStates(buffers_, variants_, scene);
        backend_.ToHost(out, sceneOut_, sizeof(scene::SceneState) * count);
    }

    void EnableRenderEvents(int capacity) {
        if (capacity < 1)
            throw std::invalid_argument("expected a capacity of at least one render event, found " +
                                        std::to_string(capacity));
        if (buffers_.renderEvents.events) return;
        const int environments = buffers_.environmentCount;
        RenderEvents events{};
        events.events = Allocate<RenderEvent>((size_t)environments * capacity);
        events.counts = Allocate<int32_t>(environments);
        events.states = Allocate<render::RenderState>(environments);
        events.animationMirrored = MirroredTable();
        events.capacity = capacity;
        buffers_.renderEvents = events;
    }
    int RenderEventsOf(int environment, RenderEvent *out, int capacity) {
        CheckEnvironment(environment);
        const RenderEvents &events = buffers_.renderEvents;
        if (!events.events) throw std::invalid_argument("the batch records no render events");
        backend_.SynchronizeAll();
        int32_t count = 0;
        backend_.ToHost(&count, events.counts + environment, sizeof(count));
        if (count > capacity)
            throw std::invalid_argument("expected room for " + std::to_string(count) +
                                        " render events, found " + std::to_string(capacity));
        if (count > 0)
            backend_.ToHost(out, events.events + (size_t)environment * events.capacity,
                            sizeof(RenderEvent) * count);
        const int32_t none = 0;
        backend_.ToDevice(events.counts + environment, &none, sizeof(none));
        return count;
    }
    int RenderEventCapacity() const { return buffers_.renderEvents.capacity; }

    // The bytes of one environment's state, from its world copied to the host. The save and the
    // mark of a rendering match are the same for one environment and for all of them.
    std::string SaveWorld(int environment, simulation::World &world, const EnvironmentSlot &slot) {
        simulation::StateCodec codec;
        codec.freeControllerCount = Variant(slot.variant).freeControllerCount;
        std::string bytes = codec.SaveFrame(world, 0, nullptr);
        // GRF: a save marks every node of a rendering match to compute afresh.
        if (world.renderMode) {
            simulation::ProcessStateMarks(world);
            backend_.ToDevice(buffers_.worlds[environment].rootCaches, world.rootCaches,
                              sizeof(world.rootCaches));
            backend_.ToDevice(&buffers_.worlds[environment].stateProcessed, &world.stateProcessed,
                              sizeof(world.stateProcessed));
        }
        return bytes;
    }

    std::string GetState(int environment, simulation::EnvironmentCore &core,
                         EnvironmentSlot &slot) {
        CheckEnvironment(environment);
        auto world = std::make_unique<simulation::World>();
        backend_.ToHost(world.get(), buffers_.worlds + environment, sizeof(simulation::World));
        backend_.ToHost(&core, buffers_.cores + environment, sizeof(core));
        backend_.ToHost(&slot, buffers_.slots + environment, sizeof(slot));
        return SaveWorld(environment, *world, slot);
    }

    // The states of every environment, with one copy of the worlds, the cores and the slots for
    // each chunk of environments, and the serializer of `GetState` on each one.
    void GetStates(std::vector<std::string> &states,
                   std::vector<simulation::EnvironmentCore> &cores,
                   std::vector<EnvironmentSlot> &slots) {
        const int count = buffers_.environmentCount;
        states.assign(count, std::string());
        cores.assign(count, simulation::EnvironmentCore());
        slots.assign(count, EnvironmentSlot());
        const int chunk = std::max(1, std::min(StateChunk(), count));
        auto worlds = std::make_unique<simulation::World[]>(chunk);
        for (int first = 0; first < count; first += chunk) {
            const int size = std::min(chunk, count - first);
            backend_.ToHost(worlds.get(), buffers_.worlds + first,
                            sizeof(simulation::World) * size);
            backend_.ToHost(cores.data() + first, buffers_.cores + first,
                            sizeof(simulation::EnvironmentCore) * size);
            backend_.ToHost(slots.data() + first, buffers_.slots + first,
                            sizeof(EnvironmentSlot) * size);
            for (int index = 0; index < size; index++)
                states[first + index] =
                    SaveWorld(first + index, worlds[index], slots[first + index]);
        }
    }

    // Rebuilds one environment from the bytes of a state into `world`, `core` and `slot`. What
    // the load keeps of the environment's present state comes from `current`.
    void BuildState(int environment, const std::string &bytes, int previousScoreDifference,
                    int previousGameMode, int previousBallOwnedTeam, const int32_t *collected,
                    const simulation::World &current, const EnvironmentSlot &currentSlot,
                    simulation::World &world, simulation::EnvironmentCore &core,
                    EnvironmentSlot &slot) {
        simulation::StateCodec codec;
        codec.animationCount = (int)store_->animations.size();
        const int bundle = hostSlots_[environment].bundle;
        if (!codec.LoadFrame(bytes, 0, world))
            throw std::invalid_argument(
                "expected the bytes of a state, found bytes that do not load" +
                LoadFailure(bundle, bytes));
        // GRF: scenarios compare without the state's `real_time`, so the bundle's stands in.
        const bool realTime = world.serialized.scenario.realTime;
        world.serialized.scenario.realTime =
            bundles_[bundle].variants[0].world->serialized.scenario.realTime;
        const std::string scenario = SaveScenario(world);
        int parity = 0;
        while (parity < 4 && bundles_[bundle].variants[parity].scenario != scenario) parity++;
        if (parity == 4)
            throw std::invalid_argument(
                "expected a state of the scenario of the environment, found another: " +
                ScenarioDifference(world.serialized.scenario, BundleScenarios(bundle)));
        world.serialized.scenario.realTime = realTime;
        // GRF: an environment runs one scenario, so a state continues only under its difficulties.
        const state::ScenarioState &bundleScenario =
            bundles_[bundle].variants[parity].world->serialized.scenario;
        const state::ScenarioState &loaded = world.serialized.scenario;
        float expected[2] = {bundleScenario.leftTeamDifficulty, bundleScenario.rightTeamDifficulty};
        for (int side = 0; side < 2; side++)
            if (!hostDifficulty_.empty() && hostDifficulty_[environment * 2 + side] >= 0.0f)
                expected[side] = hostDifficulty_[environment * 2 + side];
        if (memcmp(&loaded.leftTeamDifficulty, &expected[0], sizeof(float)) != 0 ||
            memcmp(&loaded.rightTeamDifficulty, &expected[1], sizeof(float)) != 0)
            throw std::invalid_argument(
                "expected a state of the difficulties of the environment, " +
                std::to_string(expected[0]) + " and " + std::to_string(expected[1]) + ", found " +
                std::to_string(loaded.leftTeamDifficulty) + " and " +
                std::to_string(loaded.rightTeamDifficulty));
        // GRF: the state's order of the teams loads into the scenario, while the players keep the
        // formation entries of the reset in their `TeamData`, so an environment that was reset
        // plays on with the variant of the state's order and of its reset's parity.
        int index = parity;
        if (currentSlot.episode > 0)
            index = (parity & 2) | ((currentSlot.variant - bundle * 4) & 1);
        const int variant = bundle * 4 + index;
        core = simulation::EnvironmentCore();
        core.previousScoreDifference = previousScoreDifference;
        core.previousGameMode = previousGameMode;
        core.previousBallOwnedTeam = previousBallOwnedTeam;
        // GRF: a sent-off keeper's hand-over lives in team data outside the state, so it carries
        // over, and the observation after the load already reads the roles through it.
        memcpy(world.formationOverride, current.formationOverride, sizeof(world.formationOverride));
        simulation::EnvironmentGetState(world, HostContext(variant), core.observation);
        slot = currentSlot;
        slot.variant = variant;
        const int agents = slot.leftAgents + slot.rightAgents;
        for (int agent = 0; agent < MAXIMUM_AGENTS; agent++)
            slot.collected[agent] = collected && agent < agents ? collected[agent] : 0;
        for (int agent = 0; agent < agents; agent++) {
            const bool leftTeam = agent < slot.leftAgents;
            simulation::StickyActions(world, buffers_.actionSet, leftTeam,
                                      leftTeam ? agent : agent - slot.leftAgents,
                                      slot.stickyActions[agent]);
        }
        // GRF: the referee's flag, fixed at the match's reset, carries over as well.
        memcpy(&world.refereeAnimations, &current.refereeAnimations,
               sizeof(world.refereeAnimations));
        memcpy(&world.renderMode, &current.renderMode, sizeof(world.renderMode));
        memcpy(&world.message, &current.message, sizeof(world.message));
        memcpy(&world.draws, &current.draws, sizeof(world.draws));
        memcpy(world.scoreboardGoals, current.scoreboardGoals, sizeof(world.scoreboardGoals));
        memcpy(world.humanoidHidden, current.humanoidHidden, sizeof(world.humanoidHidden));
        memcpy(&world.stateProcessed, &current.stateProcessed, sizeof(world.stateProcessed));
        world.stateProcessed++;
    }

    // Loads the states of `count` environments from `first`, with one copy of their worlds and
    // slots to the host and one copy of the rebuilt ones back, and the rebuild of `BuildState`
    // for each. The episode values of an environment are three counters and its collected
    // checkpoints, `stride` integers in all.
    void SetStates(int first, int count, const std::string *states, const int *episodes,
                   int stride) {
        if (count < 1) throw std::invalid_argument("expected at least one state, found none");
        CheckEnvironment(first);
        CheckEnvironment(first + count - 1);
        const int chunk = std::min(StateChunk(), count);
        auto current = std::make_unique<simulation::World[]>(chunk);
        auto rebuilt = std::make_unique<simulation::World[]>(chunk);
        std::vector<simulation::EnvironmentCore> cores(chunk);
        std::vector<EnvironmentSlot> before(chunk), slots(chunk);
        for (int start = 0; start < count; start += chunk) {
            const int size = std::min(chunk, count - start);
            const int base = first + start;
            backend_.ToHost(current.get(), buffers_.worlds + base,
                            sizeof(simulation::World) * size);
            backend_.ToHost(before.data(), buffers_.slots + base, sizeof(EnvironmentSlot) * size);
            for (int index = 0; index < size; index++) {
                const int *episode = episodes + (size_t)(start + index) * stride;
                new (&rebuilt[index]) simulation::World();
                BuildState(base + index, states[start + index], episode[0], episode[1], episode[2],
                           episode + 3, current[index], before[index], rebuilt[index], cores[index],
                           slots[index]);
            }
            backend_.ToDevice(buffers_.worlds + base, rebuilt.get(),
                              sizeof(simulation::World) * size);
            backend_.ToDevice(buffers_.cores + base, cores.data(),
                              sizeof(simulation::EnvironmentCore) * size);
            backend_.ToDevice(buffers_.slots + base, slots.data(), sizeof(EnvironmentSlot) * size);
            backend_.Fill(buffers_.reward + (size_t)base * Agents(), 0,
                          sizeof(float) * Agents() * size);
            backend_.Fill(buffers_.done + base, 0, size);
            backend_.Fill(buffers_.scoreReward + base, 0, sizeof(int32_t) * size);
            backend_.Fill(outputMask_, 0, buffers_.environmentCount);
            backend_.Fill(outputMask_ + base, 1, size);
            backend_.Outputs(buffers_, outputMask_, false);
            backend_.SynchronizeAll();
            for (int index = 0; index < size; index++) {
                if (!started_[base + index]) {
                    started_[base + index] = true;
                    unstarted_--;
                }
            }
        }
    }

    void SetState(int environment, const std::string &bytes, int previousScoreDifference,
                  int previousGameMode, int previousBallOwnedTeam, const int32_t *collected) {
        CheckEnvironment(environment);
        std::vector<int> episode(3 + MAXIMUM_AGENTS, 0);
        episode[0] = previousScoreDifference;
        episode[1] = previousGameMode;
        episode[2] = previousBallOwnedTeam;
        for (int agent = 0; collected && agent < MAXIMUM_AGENTS; agent++)
            episode[3 + agent] = collected[agent];
        SetStates(environment, 1, &bytes, episode.data(), 3 + MAXIMUM_AGENTS);
    }

    // The environments that one chunk of states holds in host memory, about 128 MiB of worlds.
    static int StateChunk() {
        return (int)std::max<size_t>(1, ((size_t)128 << 20) / sizeof(simulation::World));
    }

    const simulation::SimulationContext &HostContext(int variant) {
        hostContext_ = simulation::SimulationContext{&Variant(variant).staticData, &store_->view,
                                                     derived_.data(),
                                                     Variant(variant).skeletons.data(), nullptr};
        hostContext_.ticksPerStep = configuration_.ticksPerStep;
        return hostContext_;
    }

private:
    template <class T>
    T *Allocate(size_t count) {
        return backend_.template Allocate<T>(count);
    }
    const uint8_t *MirroredTable() {
        if (!animationMirrored_) {
            const std::vector<uint8_t> mirrored = AnimationMirroredNames(store_->names);
            uint8_t *table = Allocate<uint8_t>(mirrored.size());
            backend_.ToDevice(table, mirrored.data(), mirrored.size());
            animationMirrored_ = table;
        }
        return animationMirrored_;
    }
    template <class T>
    const T *Scratch(const T *host, int count, T *&slot) {
        if (Backend::HOST_MEMORY) return host;
        if (!slot) slot = Allocate<T>(count);
        backend_.ToDevice(slot, host, sizeof(T) * count);
        return slot;
    }
    template <class T>
    const T *AsyncScratch(const T *host, int count, T *&slot, Stream stream) {
        if (Backend::HOST_MEMORY) return host;
        if (!slot) {
            slot = Allocate<T>(count);
            backend_.SynchronizeAll();
        }
        backend_.ToDeviceAsync(slot, host, sizeof(T) * count, stream);
        return slot;
    }
    void CheckReverse(const int32_t *reverse) const {
        for (int environment = 0; environment < buffers_.environmentCount; environment++)
            if (reverse[environment] != 0 && reverse[environment] != 1)
                throw std::invalid_argument("the reverse flag of environment " +
                                            std::to_string(environment) + " is " +
                                            std::to_string(reverse[environment]) + ", not 0 or 1");
    }
    // GRF: the scenario holds its seed as a 32-bit `unsigned int`, and -1 draws.
    void CheckSeeds(const int64_t *seeds) const {
        for (int environment = 0; environment < buffers_.environmentCount; environment++)
            if (seeds[environment] < -1 || seeds[environment] > 4294967295LL)
                throw std::invalid_argument(
                    "expected seeds from 0 to 4294967295, or -1 to draw, found " +
                    std::to_string(seeds[environment]) + " for environment " +
                    std::to_string(environment));
    }
    const AssetVariant &Variant(int variant) const {
        return bundles_[variant / 4].variants[variant % 4];
    }
    const state::ScenarioState &Scenario(int environment) const {
        return Variant(Bundle(environment) * 4).world->serialized.scenario;
    }
    void CheckStarted() const {
        if (unstarted_ > 0)
            throw std::logic_error(
                "expected a reset of the batch, or a state for every environment, "
                "before its first step");
    }
    // GRF: each action indexes a Python list, which raises outside `[-count, count)` too.
    void CheckActions(const int32_t *actions) const {
        const int count = simulation::ActionSetSize(buffers_.actionSet);
        for (int environment = 0; environment < buffers_.environmentCount; environment++) {
            const EnvironmentSlot &slot = hostSlots_[environment];
            const int32_t *row = actions + (size_t)environment * buffers_.agents;
            for (int agent = 0; agent < slot.leftAgents + slot.rightAgents; agent++)
                if (row[agent] < -count || row[agent] >= count)
                    throw std::out_of_range(
                        "expected action indices from " + std::to_string(-count) + " to " +
                        std::to_string(count - 1) + ", found " + std::to_string(row[agent]));
        }
    }
    void CheckEnvironment(int environment) const {
        if (environment < 0 || environment >= buffers_.environmentCount)
            throw std::invalid_argument("expected an environment from 0 to " +
                                        std::to_string(buffers_.environmentCount - 1) + ", found " +
                                        std::to_string(environment));
    }
    std::vector<const state::ScenarioState *> BundleScenarios(int bundle) const {
        std::vector<const state::ScenarioState *> scenarios;
        for (const AssetVariant &variant : bundles_[bundle].variants)
            scenarios.push_back(&variant.world->serialized.scenario);
        return scenarios;
    }
    std::string LoadFailure(int bundle, const std::string &bytes) const {
        auto environment = std::make_unique<state::EnvironmentStateFull>();
        try {
            state::Archive archive = state::Archive::Loader(bytes);
            state::ProcessEnvironmentScenario(archive, *environment);
        } catch (const std::exception &exception) {
            return std::string(": ") + exception.what();
        }
        environment->scenario.realTime =
            bundles_[bundle].variants[0].world->serialized.scenario.realTime;
        const std::string difference =
            ScenarioDifference(environment->scenario, BundleScenarios(bundle));
        if (!difference.empty()) return ", of another scenario: " + difference;
        try {
            state::Archive archive = state::Archive::Loader(bytes);
            state::ProcessEnvironmentState(archive, *environment,
                                           bundles_[bundle].variants[0].freeControllerCount);
            if (!archive.AtEnd())
                return ": " + std::to_string(bytes.size() - archive.position()) +
                       " bytes follow the end of the state";
        } catch (const std::exception &exception) {
            return std::string(": ") + exception.what();
        }
        return ": a player, team, gamer, keyboard or animation index lies out of range";
    }

    Backend backend_;
    BatchEnvironmentConfiguration configuration_;
    std::vector<Assets> bundles_;
    std::vector<EnvironmentSlot> hostSlots_;
    std::vector<float> hostDifficulty_;
    std::unique_ptr<animation::AnimationStore> store_;
    std::vector<animation::AnimationDerived> derived_;
    Variants variants_{};
    render::RenderState *renderStates_ = nullptr, *renderOut_ = nullptr;
    int32_t *renderEpisodes_ = nullptr, *renderIndices_ = nullptr;
    uint32_t *renderProcessed_ = nullptr;
    const uint8_t *animationMirrored_ = nullptr;
    int renderCapacity_ = 0;
    scene::SceneState *sceneOut_ = nullptr;
    int32_t *sceneIndices_ = nullptr;
    int sceneCapacity_ = 0;
    int players_ = 0;
    std::vector<bool> started_;
    int unstarted_ = 0;
    EnvironmentBuffers buffers_{};
    int32_t *actions_ = nullptr;
    uint8_t *outputMask_ = nullptr;
    int64_t *scratchSeeds_ = nullptr;
    int32_t *scratchReverse_ = nullptr;
    uint8_t *maskedMask_ = nullptr;
    int64_t *maskedSeeds_ = nullptr;
    int32_t *maskedReverse_ = nullptr;
    int32_t *asyncActions_ = nullptr;
    simulation::SimulationContext hostContext_{};
};

}  // namespace batch
}  // namespace cuju
