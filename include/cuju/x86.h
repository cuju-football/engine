// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host_device.h"

namespace cuju {
CUJU_HOST_DEVICE int32_t FloatToInt32(float value) {
    if (!(value >= -2147483648.0f && value < 2147483648.0f)) return INT32_MIN;
    return (int32_t)value;
}

CUJU_HOST_DEVICE uint32_t FloatToUnsigned32(float value) {
    if (!(value >= -9223372036854775808.0f && value < 9223372036854775808.0f)) return 0u;
    return (uint32_t)(uint64_t)(int64_t)value;
}

}

namespace cuju {
CUJU_HOST_DEVICE int32_t DoubleToInt32(double value) {
    if (!(value > -2147483649.0 && value < 2147483648.0)) return INT32_MIN;
    return (int32_t)value;
}

}
