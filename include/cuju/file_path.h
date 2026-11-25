// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#if defined(_WIN32)
#include <filesystem>
#endif

namespace cuju {
#if defined(_WIN32)
inline std::filesystem::path FilePath(const std::string &utf8) {
    return std::filesystem::u8path(utf8);
}
#else
inline const std::string &FilePath(const std::string &path) { return path; }
#endif

}
