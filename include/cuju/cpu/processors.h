// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace cuju {
namespace cpu {
inline std::vector<std::string> SplitFields(const std::string &line, char separator) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, separator)) fields.push_back(field);
    if (!line.empty() && line.back() == separator) fields.push_back("");
    return fields;
}

inline bool HasItem(const std::string &list, const std::string &item) {
    for (const std::string &entry : SplitFields(list, ','))
        if (entry == item) return true;
    return false;
}

inline bool ReadWords(const std::string &path, std::string &first, std::string &second) {
    std::ifstream file(path);
    if (!file) return false;
    first.clear();
    second.clear();
    file >> first >> second;
    return !first.empty();
}

inline bool ParseInteger(const std::string &text, int64_t &value) {
    if (text.empty()) return false;
    size_t used = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception &) {
        return false;
    }
    return used == text.size();
}

inline int QuotaToProcessors(int64_t quota, int64_t period) {
    if (quota <= 0 || period <= 0) return 0;
    const int64_t processors = (quota + period - 1) / period;
    return processors > (1 << 20) ? 1 << 20 : (int)processors;
}

inline int DirectoryQuota(const std::string &directory, bool version2) {
    std::string first, second;
    int64_t quota = 0, period = 0;
    if (version2) {
        if (!ReadWords(directory + "/cpu.max", first, second) || first == "max") return 0;
        if (!ParseInteger(first, quota)) return 0;
        if (second.empty())
            period = 100000;
        else if (!ParseInteger(second, period))
            return 0;
        return QuotaToProcessors(quota, period);
    }
    if (!ReadWords(directory + "/cpu.cfs_quota_us", first, second) || !ParseInteger(first, quota))
        return 0;
    if (!ReadWords(directory + "/cpu.cfs_period_us", first, second) || !ParseInteger(first, period))
        return 0;
    return QuotaToProcessors(quota, period);
}

inline int QuotaProcessors(const std::string &root = "") {
    std::string version2Path, version1Path;
    bool haveVersion2 = false, haveVersion1 = false;
    {
        std::ifstream file(root + "/proc/self/cgroup");
        std::string line;
        while (std::getline(file, line)) {
            const size_t first = line.find(':'), second = line.find(':', first + 1);
            if (first == std::string::npos || second == std::string::npos) continue;
            const std::string hierarchy = line.substr(0, first);
            const std::string controllers = line.substr(first + 1, second - first - 1);
            const std::string path = line.substr(second + 1);
            if (hierarchy == "0" && controllers.empty()) {
                version2Path = path;
                haveVersion2 = true;
            } else if (HasItem(controllers, "cpu")) {
                version1Path = path;
                haveVersion1 = true;
            }
        }
    }
    int smallest = 0;
    std::ifstream file(root + "/proc/self/mountinfo");
    std::string line;
    while (std::getline(file, line)) {
        const size_t dash = line.find(" - ");
        if (dash == std::string::npos) continue;
        const std::vector<std::string> before = SplitFields(line.substr(0, dash), ' ');
        const std::vector<std::string> after = SplitFields(line.substr(dash + 3), ' ');
        if (before.size() < 5 || after.empty()) continue;
        bool version2;
        if (after[0] == "cgroup2" && haveVersion2)
            version2 = true;
        else if (after[0] == "cgroup" && haveVersion1 && after.size() >= 3 &&
                 HasItem(after[2], "cpu"))
            version2 = false;
        else
            continue;
        const std::string &mountRoot = before[3];
        const std::string mountPoint = root + before[4];
        std::string path = version2 ? version2Path : version1Path;
        if (mountRoot != "/") {
            if (path.compare(0, mountRoot.size(), mountRoot) == 0 &&
                (path.size() == mountRoot.size() || path[mountRoot.size()] == '/'))
                path = path.substr(mountRoot.size());
            else
                path.clear();
        }
        if (path.find("/..") != std::string::npos) path.clear();
        while (!path.empty() && path.back() == '/') path.pop_back();
        for (;;) {
            const int quota = DirectoryQuota(mountPoint + path, version2);
            if (quota > 0 && (smallest == 0 || quota < smallest)) smallest = quota;
            if (path.empty()) break;
            const size_t slash = path.rfind('/');
            path = slash == std::string::npos ? std::string() : path.substr(0, slash);
        }
    }
    return smallest;
}

}
}
