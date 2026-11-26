// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "cuju/simulation/state_io.h"

#if !defined(_MSC_VER)
#pragma GCC visibility push(default)
#endif
extern "C" {
int cuju_state_padding(const char *bytes, unsigned long long size, unsigned char *mask) {
    try {
        const std::string match(bytes, size);
        cuju::simulation::StateCodec codec;
        auto world = std::make_unique<cuju::simulation::World>();
        if (!codec.Load(match, *world))
            throw std::invalid_argument("expected the bytes of a match");
        std::vector<cuju::state::LayoutEntry> layout;
        cuju::state::Archive archive = cuju::state::Archive::Saver();
        archive.layout = &layout;
        archive.layoutBase = &world->serialized;
        archive.layoutStorage = false;
        cuju::state::ProcessEnvironmentState(archive, world->serialized, codec.freeControllerCount);
        std::memset(mask, 0, size);
        // GRF: a `blunted::radian` saves its float, its flag and three bytes of padding, which hold
        // whatever its memory held, so two GRF environments of one match differ there.
        for (const cuju::state::LayoutEntry &entry : layout)
            if (std::strcmp(entry.type, TYPE_RADIAN) == 0 && entry.size == 8 &&
                entry.position + 8 <= (int)size)
                std::memset(mask + entry.position + 5, 1, 3);
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

const char *cuju_state_layout(unsigned long long *size) {
    try {
        static const std::string text = [] {
            auto world = std::make_unique<cuju::simulation::World>();
            std::vector<cuju::state::LayoutEntry> layout;
            cuju::state::Archive archive = cuju::state::Archive::Saver();
            archive.layout = &layout;
            archive.layoutBase = &world->serialized;
            archive.layoutStorage = true;
            cuju::state::ProcessEnvironmentState(archive, world->serialized,
                                                 cuju::state::MAXIMUM_CONTROLLERS);
            std::string lines;
            for (const cuju::state::LayoutEntry &entry : layout)
                if (entry.offset <= sizeof(world->serialized) &&
                    (size_t)entry.size <= sizeof(world->serialized) - entry.offset)
                    lines += entry.path + "\t" + std::to_string(entry.offset) + "\t" +
                             std::to_string(entry.size) + "\t" + entry.type + "\n";
            return lines;
        }();
        *size = text.size();
        return text.c_str();
    } catch (const std::exception &exception) {
        Fail(exception);
        return nullptr;
    }
}

int cuju_state_gather(const char *bytes, unsigned long long size, int frame,
                      const unsigned long long *offsets, const unsigned int *sizes, int count,
                      unsigned char *out) {
    try {
        if (frame != 0 && frame != 1)
            throw std::invalid_argument("expected the frame 0 or 1, found " +
                                        std::to_string(frame));
        thread_local cuju::simulation::StateCodec codec;
        thread_local std::unique_ptr<cuju::simulation::World> world =
            std::make_unique<cuju::simulation::World>();
        if (!codec.LoadFrame(std::string(bytes, size), frame ? 2 : 0, *world))
            throw std::invalid_argument("expected the bytes of a match");
        cuju::simulation::BallMaterialize(world->serialized.match.ball);
        for (int image = 0; image < world->serialized.match.mentalImages.count; image++)
            cuju::simulation::MentalMaterialize(world->serialized.match.mentalImages[image]);
        const unsigned char *base = reinterpret_cast<const unsigned char *>(&world->serialized);
        for (int index = 0; index < count; index++) {
            if (offsets[index] > sizeof(world->serialized) ||
                sizes[index] > sizeof(world->serialized) - offsets[index])
                throw std::invalid_argument("expected a field inside the state, found offset " +
                                            std::to_string(offsets[index]));
            std::memcpy(out, base + offsets[index], sizes[index]);
            out += sizes[index];
        }
        return 0;
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}
}
#if !defined(_MSC_VER)
#pragma GCC visibility pop
#endif
