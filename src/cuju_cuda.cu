// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#include "cuju/gpu/batch_environment.cuh"

using cuju::gpu::BatchEnvironment;
#define CUJU_BACKEND_NAME "cuda"

#include "environment_api.h"
#include "engine_api.h"
#include "scenario_api.h"
#include "state_api.h"
