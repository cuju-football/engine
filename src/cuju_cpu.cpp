// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include "cuju/cpu/batch_environment.h"

using cuju::cpu::BatchEnvironment;
#define CUJU_BACKEND_NAME "cpu"

#include "environment_api.h"
#include "engine_api.h"
#include "scenario_api.h"
#include "state_api.h"

#if defined(_MSC_VER)
#define CUJU_CPU_EXPORT __declspec(dllexport)
#else
#define CUJU_CPU_EXPORT
#pragma GCC visibility push(default)
#endif
extern "C" {
CUJU_CPU_EXPORT int cuju_cpu_quota(const char *root) {
    try {
        return cuju::cpu::QuotaProcessors(root ? root : "");
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}
}
#if !defined(_MSC_VER)
#pragma GCC visibility pop
#endif
