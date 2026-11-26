// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(CUJU_NVTX)
#include <nvtx3/nvToolsExt.h>

namespace cuju {
inline nvtxDomainHandle_t NvtxDomain() {
    static const nvtxDomainHandle_t domain = nvtxDomainCreateA("cuju");
    return domain;
}

class NvtxRange {
public:
    explicit NvtxRange(nvtxStringHandle_t name) {
        nvtxEventAttributes_t attributes = {};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        attributes.message.registered = name;
        nvtxDomainRangePushEx(NvtxDomain(), &attributes);
    }
    ~NvtxRange() { nvtxDomainRangePop(NvtxDomain()); }
    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};

}  // namespace cuju

#define CUJU_NVTX_RANGE(name)                                  \
    static const nvtxStringHandle_t cujuNvtxName =             \
        nvtxDomainRegisterStringA(::cuju::NvtxDomain(), name); \
    ::cuju::NvtxRange cujuNvtxRange(cujuNvtxName)
#else
#define CUJU_NVTX_RANGE(name)
#endif
