// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// The text helpers of GRF's loaders (GRF, Apache-2.0): glibc's atof and atoi as GRF calls them,
// base/utils.cpp's tokenize and get_file_name, and the whole-file read of GetFile.

#ifndef CUJU_RENDER_ASSETS_TEXT_H
#define CUJU_RENDER_ASSETS_TEXT_H

#include <string>
#include <vector>

namespace cuju::render::assets::text {
// strtod(s, NULL) of glibc in the C locale, correctly rounded: leading white space, a sign, a
// decimal or hexadecimal number, inf or nan, and 0 when nothing parses.
double Atof(const char *s);
inline double Atof(const std::string &s) { return Atof(s.c_str()); }

// (int)strtol(s, NULL, 10) of glibc.
int Atoi(const char *s);
inline int Atoi(const std::string &s) { return Atoi(s.c_str()); }

void Tokenize(const std::string &str, std::vector<std::string> &tokens,
              const std::string &delimiters);

std::string FileName(const std::string &filename);

std::string DataPath(const std::string &data_dir, const std::string &path);

bool ReadFile(const std::string &path, std::string *contents, std::string *error);

}  // namespace cuju::render::assets::text

#endif
