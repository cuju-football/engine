// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "assets/text.h"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <system_error>

#include "cuju/file_path.h"

namespace cuju::render::assets::text {
namespace {
bool IsSpace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

[[maybe_unused]] bool IsHexDigit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

}  // namespace

double Atof(const char *s) {
    while (IsSpace(*s)) s++;
    bool negative = false;
    if (*s == '+' || *s == '-') negative = *s++ == '-';
    const char *end = s;
    while (*end) end++;
    double value = 0.0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    std::from_chars_result result;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X') &&
        (IsHexDigit(s[2]) || (s[2] == '.' && IsHexDigit(s[3])))) {
        result = std::from_chars(s + 2, end, value, std::chars_format::hex);
    } else {
        result = std::from_chars(s, end, value, std::chars_format::general);
    }
    if (result.ec == std::errc::result_out_of_range) {
        value = std::strtod(s, nullptr);
    } else if (result.ec != std::errc()) {
        value = 0.0;
    }
#else
    value = std::strtod(s, nullptr);
#endif
    return negative ? -value : value;
}

int Atoi(const char *s) {
    while (IsSpace(*s)) s++;
    bool negative = false;
    if (*s == '+' || *s == '-') negative = *s++ == '-';
    const std::uint64_t limit = negative ? (std::uint64_t)INT64_MAX + 1 : (std::uint64_t)INT64_MAX;
    std::uint64_t magnitude = 0;
    for (; *s >= '0' && *s <= '9'; s++) {
        const std::uint64_t digit = (std::uint64_t)(*s - '0');
        magnitude = magnitude > (limit - digit) / 10 ? limit : magnitude * 10 + digit;
    }
    const std::uint64_t bits = negative ? 0 - magnitude : magnitude;
    return (int)(std::int32_t)(std::uint32_t)bits;
}

void Tokenize(const std::string &str, std::vector<std::string> &tokens,
              const std::string &delimiters) {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
}

std::string FileName(const std::string &filename) {
    std::string chomped = filename.substr(filename.find_last_of('\\') + 1);
    const std::size_t slash = filename.find_last_of('/') + 1;
    return slash <= chomped.size() ? chomped.substr(slash) : std::string();
}

std::string DataPath(const std::string &data_dir, const std::string &path) {
    if (!path.empty() && path[0] == '/') return path;
    return data_dir + '/' + path;
}

bool ReadFile(const std::string &path, std::string *contents, std::string *error) {
    std::ifstream file(cuju::FilePath(path), std::ios::binary);
    if (!file) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    file.seekg(0, std::ios::end);
    contents->resize((std::size_t)file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(&(*contents)[0], (std::streamsize)contents->size());
    if (!file) {
        if (error) *error = "cannot read " + path;
        return false;
    }
    return true;
}

}  // namespace cuju::render::assets::text
