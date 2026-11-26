// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(_WIN32)
#include <malloc.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "../batch/batch_environment.h"
#include "../floating_point.h"
#include "job_thread.h"
#include "placement.h"
#include "processors.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cuju {
namespace cpu {
using namespace batch;

inline void *AllocateZeroed(size_t bytes, size_t smallAlignment = 64) {
    constexpr size_t HUGE_PAGE = size_t(2) << 20;
    const size_t alignment = bytes >= HUGE_PAGE ? HUGE_PAGE : smallAlignment;
    const size_t size = (bytes + alignment - 1) / alignment * alignment;
#if defined(_WIN32)
    void *pointer = _aligned_malloc(size ? size : alignment, alignment);
    if (!pointer) throw std::bad_alloc();
#else
    void *pointer = nullptr;
    if (posix_memalign(&pointer, alignment, size ? size : alignment) != 0) throw std::bad_alloc();
#endif
#if defined(MADV_HUGEPAGE)
    if (alignment == HUGE_PAGE) madvise(pointer, size, MADV_HUGEPAGE);
#endif
    memset(pointer, 0, size ? size : alignment);
    return pointer;
}

struct HostTables {
    static constexpr bool HOST = true;
    std::vector<void *> allocations;
    simulation::SimulationContext context{};
    size_t bytes = 0;

    HostTables() = default;
    HostTables(const HostTables &) = delete;
    HostTables &operator=(const HostTables &) = delete;
    ~HostTables() {
#if defined(_WIN32)
        for (void *allocation : allocations) _aligned_free(allocation);
#else
        for (void *allocation : allocations) free(allocation);
#endif
    }

    template <class T>
    T *Upload(const T *data, size_t count) {
        static_assert(std::is_trivially_copyable<T>::value, "tables must be trivially copyable");
        if (count == 0) return nullptr;
        T *copy = static_cast<T *>(AllocateZeroed(count * sizeof(T)));
        memcpy(static_cast<void *>(copy), static_cast<const void *>(data), count * sizeof(T));
        allocations.push_back(copy);
        bytes += count * sizeof(T);
        return copy;
    }
    template <class T>
    T *Upload(const std::vector<T> &values) {
        return Upload(values.data(), values.size());
    }
};

inline void Pause() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("isb" : : : "memory");
#else
    std::this_thread::yield();
#endif
}

inline int AvailableProcessors() {
    int processors = (int)std::max(1u, std::thread::hardware_concurrency());
#if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) processors = std::max(1, CPU_COUNT(&set));
    const int quota = QuotaProcessors();
    if (quota > 0) processors = std::min(processors, quota);
#endif
    return processors;
}

inline std::atomic<uint64_t> &ForkCount() {
    static std::atomic<uint64_t> count{0};
    return count;
}
inline void CountFork() { ForkCount().fetch_add(1, std::memory_order_relaxed); }
inline uint64_t ForksSoFar() {
#if !defined(_WIN32)
    static const bool watching = pthread_atfork(nullptr, nullptr, CountFork) == 0;
    (void)watching;
#endif
    return ForkCount().load(std::memory_order_relaxed);
}

class ThreadPool {
public:
    explicit ThreadPool(int threads, bool pin = false, bool numa = false) : forks_(ForksSoFar()) {
        if (threads <= 0) threads = AvailableProcessors();
        threads_ = threads;
        placement_ = Placement::Make(threads, pin, numa);
        groups_ = placement_.Groups();
        ranges_.reset(new GroupRange[groups_]);
        for (int slot = 0; slot < threads && groups_ > 1; slot++)
            ranges_[placement_.SlotGroup(slot)].slots++;
        for (int worker = 1; worker < threads; worker++)
            workers_.emplace_back([this, worker] { Work(worker); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            generation_.fetch_add(1, std::memory_order_release);
        }
        wake_.notify_all();
        for (std::thread &worker : workers_) worker.join();
    }
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int Threads() const { return threads_; }
    bool Lost() const { return forks_ != ForkCount().load(std::memory_order_relaxed); }
    bool Busy() const { return busy_.load(std::memory_order_relaxed); }
    void PlaceCaller() const { placement_.PinSlot(0); }

    template <class Body>
    void For(int count, Body &&body, int grain = 1) {
        if (count <= 0) return;
        if (workers_.empty() || count <= grain) {
            busy_.store(true, std::memory_order_relaxed);
            try {
                for (int index = 0; index < count; index++) body(index);
            } catch (...) {
                busy_.store(false, std::memory_order_relaxed);
                throw;
            }
            busy_.store(false, std::memory_order_relaxed);
            return;
        }
        using Function = typename std::remove_reference<Body>::type;
        body_ = static_cast<const void *>(&body);
        call_ = [](const void *function, int index) {
            (*static_cast<Function *>(const_cast<void *>(function)))(index);
        };
        count_ = count;
        grain_ = grain;
        next_.store(0, std::memory_order_relaxed);
        for (int group = 0, begin = 0, slots = 0; group < groups_ && groups_ > 1; group++) {
            slots += ranges_[group].slots;
            ranges_[group].next.store(begin, std::memory_order_relaxed);
            ranges_[group].end = begin = (int)((long)count * slots / threads_);
        }
        remaining_.store((int)workers_.size(), std::memory_order_relaxed);
        busy_.store(true, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            generation_.fetch_add(1, std::memory_order_release);
        }
        wake_.notify_all();
        const int own = placement_.CurrentGroup();
        for (int group = 0; group < groups_; group++) Drain((own + group) % groups_);
        while (remaining_.load(std::memory_order_acquire) != 0) Pause();
        busy_.store(false, std::memory_order_relaxed);
        if (failure_) {
            std::exception_ptr failure = failure_;
            failure_ = nullptr;
            std::rethrow_exception(failure);
        }
    }

private:
    static constexpr int SPINS = 1 << 12;

    struct GroupRange {
        alignas(64) std::atomic<int> next{0};
        int end = 0, slots = 0;
    };

    void Drain(int group) {
        std::atomic<int> &next = groups_ > 1 ? ranges_[group].next : next_;
        const int last = groups_ > 1 ? ranges_[group].end : count_;
        try {
            for (;;) {
                const int start = next.fetch_add(grain_, std::memory_order_relaxed);
                if (start >= last) return;
                const int end = std::min(last, start + grain_);
                for (int index = start; index < end; index++) call_(body_, index);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
        }
    }
    void Work(int slot) {
        DefaultFloatingPoint floatingPoint;
        placement_.PinSlot(slot);
        const int group = placement_.SlotGroup(slot);
        uint64_t seen = 0;
        for (;;) {
            uint64_t generation = generation_.load(std::memory_order_acquire);
            for (int spin = 0; spin < SPINS && generation == seen; spin++) {
                Pause();
                generation = generation_.load(std::memory_order_acquire);
            }
            if (generation == seen) {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock,
                           [&] { return generation_.load(std::memory_order_acquire) != seen; });
                generation = generation_.load(std::memory_order_acquire);
            }
            seen = generation;
            if (stop_) return;
            Drain(group);
            remaining_.fetch_sub(1, std::memory_order_release);
        }
    }

    uint64_t forks_;
    std::atomic<bool> busy_{false};
    Placement placement_;
    int groups_ = 1;
    std::unique_ptr<GroupRange[]> ranges_;
    int threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::atomic<uint64_t> generation_{0};
    bool stop_ = false;
    const void *body_ = nullptr;
    void (*call_)(const void *, int) = nullptr;
    int count_ = 0, grain_ = 1;
    std::exception_ptr failure_;
    alignas(64) std::atomic<int> next_{0};
    alignas(64) std::atomic<int> remaining_{0};
};

class CpuBackend {
public:
    using Stream = void *;
    static constexpr bool HOST_MEMORY = true;

    CpuBackend() = default;
    CpuBackend(const CpuBackend &) = delete;
    CpuBackend &operator=(const CpuBackend &) = delete;
    ~CpuBackend() {
        if (StepThreadLost())
            (void)stepThread_.release();
        else
            stepThread_->Stop();
        if (pool_ && pool_->Lost()) (void)pool_.release();
        pool_.reset();
#if defined(_WIN32)
        for (void *pointer : allocations_) _aligned_free(pointer);
#else
        for (void *pointer : allocations_) free(pointer);
#endif
        for (const auto &[pointer, bytes] : untouched_) ReleaseUntouched(pointer, bytes);
    }

    void Configure(const BatchEnvironmentConfiguration &configuration) {
        threads_ = configuration.threads;
        pinThreads_ = configuration.pinThreads;
        numa_ = configuration.numa;
        pageAligned_ = configuration.pageAligned;
        pool_ = std::make_unique<ThreadPool>(threads_, pinThreads_, numa_);
    }
    void Reserve(const EnvironmentBuffers &, const Variants &) {}
    static size_t LocalMemoryBytes() { return 0; }
    int Threads() const { return pool_ ? pool_->Threads() : 1; }
    template <class T>
    T *Allocate(size_t count) {
        const size_t bytes = sizeof(T) * (count ? count : 1);
        allocatedBytes_ += bytes;
        if (numa_ && bytes >= (size_t(2) << 20)) {
            untouched_.push_back({AllocateUntouched(bytes), bytes});
            return static_cast<T *>(untouched_.back().first);
        }
        T *pointer = static_cast<T *>(AllocateZeroed(bytes, pageAligned_ ? 4096 : 64));
        allocations_.push_back(pointer);
        return pointer;
    }
    size_t AllocatedBytes() const { return allocatedBytes_ + tables_.bytes; }
    HostTables &Tables() { return tables_; }
    void ToDevice(void *destination, const void *source, size_t bytes) const {
        Steps().Settle();
        memcpy(destination, source, bytes);
    }
    void ToDeviceAsync(void *destination, const void *source, size_t bytes, Stream) const {
        Steps().Settle();
        memcpy(destination, source, bytes);
    }
    void ToHost(void *destination, const void *source, size_t bytes) const {
        Steps().Settle();
        memcpy(destination, source, bytes);
    }
    void Copy(void *destination, const void *source, size_t bytes) const {
        Steps().Settle();
        memcpy(destination, source, bytes);
    }
    void Fill(void *destination, int value, size_t bytes) const {
        Steps().Settle();
        memset(destination, value, bytes);
    }
    void Synchronize(Stream) const {}
    void SynchronizeAll() const {}

    void Launch(std::function<void()> job) {
        Pool();
        Steps().Launch(std::move(job), [this] { pool_->PlaceCaller(); });
    }
    void Wait() { Steps().Wait(); }

    void Reset(const EnvironmentBuffers &buffers, const Variants &variants, const uint8_t *mask,
               const int64_t *seeds, const int32_t *reverse, Stream) {
        Steps().Settle();
        Pool().For(buffers.environmentCount, [&](int environment) {
            if (mask && !mask[environment]) return;
            ResetAndWriteEnvironment(buffers, variants, environment, seeds, reverse);
        });
    }
    void Step(const EnvironmentBuffers &buffers, const Variants &variants, const int32_t *actions,
              bool autoReset, Stream) {
        Steps().Settle();
        Pool().For(buffers.environmentCount, [&](int environment) {
            StepEnvironment(buffers, variants, environment, actions, autoReset);
        });
    }
    void RenderStates(const EnvironmentBuffers &buffers, const Variants &variants,
                      const RenderBuffers &render) {
        Steps().Settle();
        Pool().For(
            render.count, [&](int output) { WriteRenderState(buffers, variants, render, output); },
            LIGHT_GRAIN);
    }
    void Outputs(const EnvironmentBuffers &buffers, const uint8_t *mask, bool afterStep) {
        Steps().Settle();
        Pool().For(
            buffers.environmentCount,
            [&](int environment) {
                if (mask && !mask[environment]) return;
                const EnvironmentSlot &slot = buffers.slots[environment];
                for (int agent = 0; agent < slot.leftAgents + slot.rightAgents; agent++)
                    WriteAgentOutputs(buffers, environment, agent, afterStep);
            },
            LIGHT_GRAIN);
    }

private:
    static constexpr int LIGHT_GRAIN = 8;

    ThreadPool &Pool() {
        if (pool_->Lost()) {
            if (pool_->Busy())
                throw std::runtime_error(
                    "the process forked while another thread was running this batch, so the "
                    "child's copy of the batch is incomplete; create the batch after the fork");
            (void)pool_.release();
            pool_ = std::make_unique<ThreadPool>(threads_, pinThreads_, numa_);
        }
        return *pool_;
    }

    bool StepThreadLost() const {
        return stepForks_ != ForkCount().load(std::memory_order_relaxed);
    }
    JobThread &Steps() const {
        if (StepThreadLost()) {
            if (stepThread_->Pending())
                throw std::runtime_error(
                    "the process forked while another thread was running this batch, so the "
                    "child's copy of the batch is incomplete; create the batch after the fork");
            (void)stepThread_.release();
            stepThread_ = std::make_unique<JobThread>();
            stepForks_ = ForkCount().load(std::memory_order_relaxed);
        }
        return *stepThread_;
    }

    HostTables tables_;
    std::vector<void *> allocations_;
    std::vector<std::pair<void *, size_t>> untouched_;
    bool numa_ = false, pageAligned_ = false;
    size_t allocatedBytes_ = 0;
    std::unique_ptr<ThreadPool> pool_;
    int threads_ = 0;
    bool pinThreads_ = false;
    mutable std::unique_ptr<JobThread> stepThread_ = std::make_unique<JobThread>();
    mutable uint64_t stepForks_ = ForksSoFar();
};

using BatchEnvironment = batch::BackendBatchEnvironment<CpuBackend>;

}  // namespace cpu
}  // namespace cuju
