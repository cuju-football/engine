// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__linux__)
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#endif
#if defined(_WIN32)
#include <malloc.h>
#else
#include <sys/mman.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace cuju {
namespace cpu {
inline std::vector<int> ParseProcessorList(const std::string &text) {
    std::vector<int> processors;
    size_t position = 0;
    while (position < text.size()) {
        size_t end = text.find(',', position);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(position, end - position);
        const size_t dash = item.find('-');
        const int first = atoi(item.c_str());
        const int last = dash == std::string::npos ? first : atoi(item.c_str() + dash + 1);
        for (int processor = first; processor <= last && !item.empty(); processor++)
            processors.push_back(processor);
        position = end + 1;
    }
    return processors;
}

#if defined(__linux__)
inline std::vector<int> ProcessorNodes() {
    std::vector<int> nodes;
    if (const char *given = getenv("CUJU_NUMA_NODES")) {
        for (const char *text = given; *text;) {
            nodes.push_back(atoi(text));
            while (*text && *text != ',') text++;
            if (*text == ',') text++;
        }
        return nodes;
    }
    DIR *directory = opendir("/sys/devices/system/node");
    if (!directory) return nodes;
    while (const dirent *entry = readdir(directory)) {
        int node = -1;
        if (sscanf(entry->d_name, "node%d", &node) != 1 || node < 0) continue;
        std::ifstream file(std::string("/sys/devices/system/node/") + entry->d_name + "/cpulist");
        std::string text;
        std::getline(file, text);
        for (int processor : ParseProcessorList(text)) {
            if (processor >= (int)nodes.size()) nodes.resize(processor + 1, -1);
            nodes[processor] = node;
        }
    }
    closedir(directory);
    return nodes;
}
#endif

struct Placement {
    bool pin = false, numa = false;
    std::vector<int> slotProcessor, slotGroup;
    std::vector<std::vector<int>> groupProcessors;
    std::vector<int> processorGroup;

    int Groups() const { return numa ? (int)groupProcessors.size() : 1; }

    static Placement Make(int threads, bool pin, bool numa) {
        Placement placement;
        placement.pin = pin;
        placement.numa = numa;
        if (!pin && !numa) return placement;
#if !defined(__linux__)
        (void)threads;
        return Placement();
#else
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) != 0) return Placement();
        const std::vector<int> nodes = numa ? ProcessorNodes() : std::vector<int>();
        std::vector<std::pair<int, int>> allowed;
        for (int processor = 0; processor < CPU_SETSIZE; processor++)
            if (CPU_ISSET(processor, &set))
                allowed.push_back(
                    {processor < (int)nodes.size() ? std::max(0, nodes[processor]) : 0, processor});
        if (allowed.empty()) return Placement();
        std::sort(allowed.begin(), allowed.end());
        std::vector<int> nodeGroup;
        int highest = 0;
        for (const auto &[node, processor] : allowed) highest = std::max(highest, processor);
        placement.processorGroup.assign(highest + 1, -1);
        for (const auto &[node, processor] : allowed) {
            if (node >= (int)nodeGroup.size()) nodeGroup.resize(node + 1, -1);
            if (nodeGroup[node] < 0) {
                nodeGroup[node] = (int)placement.groupProcessors.size();
                placement.groupProcessors.emplace_back();
            }
            placement.groupProcessors[nodeGroup[node]].push_back(processor);
            placement.processorGroup[processor] = nodeGroup[node];
        }
        for (int slot = 0; slot < threads; slot++) {
            const auto &[node, processor] =
                allowed[(size_t)slot * allowed.size() / threads % allowed.size()];
            placement.slotProcessor.push_back(processor);
            placement.slotGroup.push_back(numa ? nodeGroup[node] : 0);
        }
        if (numa) {
            std::vector<std::vector<int>> used(placement.groupProcessors.size());
            std::vector<int> renumber(placement.groupProcessors.size(), -1);
            int groups = 0;
            for (int &group : placement.slotGroup) {
                if (renumber[group] < 0) {
                    renumber[group] = groups++;
                    used[renumber[group]] = placement.groupProcessors[group];
                }
                group = renumber[group];
            }
            for (int &group : placement.processorGroup)
                if (group >= 0) group = renumber[group];
            used.resize(groups);
            placement.groupProcessors = used;
        }
        return placement;
#endif
    }

    int SlotGroup(int slot) const { return numa ? slotGroup[slot] : 0; }

    int CurrentGroup() const {
        if (!numa) return 0;
#if defined(__linux__)
        const int processor = sched_getcpu();
        if (processor < 0 || processor >= (int)processorGroup.size()) return 0;
        return std::max(0, processorGroup[processor]);
#else
        return 0;
#endif
    }

    void PinSlot(int slot) const {
        if (slotProcessor.empty()) return;
#if defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        if (pin)
            CPU_SET(slotProcessor[slot], &set);
        else
            for (int processor : groupProcessors[slotGroup[slot]]) CPU_SET(processor, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
        (void)slot;
#endif
    }
};

inline void *AllocateUntouched(size_t bytes) {
#if defined(_WIN32)
    void *pointer = _aligned_malloc(bytes, 4096);
    if (!pointer) throw std::bad_alloc();
    memset(pointer, 0, bytes);
    return pointer;
#else
    void *pointer =
        mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pointer == MAP_FAILED) throw std::bad_alloc();
#if defined(MADV_HUGEPAGE)
    madvise(pointer, bytes, MADV_HUGEPAGE);
#endif
    return pointer;
#endif
}

inline void ReleaseUntouched(void *pointer, size_t bytes) {
#if defined(_WIN32)
    (void)bytes;
    _aligned_free(pointer);
#else
    munmap(pointer, bytes);
#endif
}

}
}
