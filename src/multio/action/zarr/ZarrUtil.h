/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

/// Small helpers of the zarr action: atomic file writes, directory creation and a
/// minimal JSON writer/scanner. The scanner keeps the raw text of JSON values, so
/// metadata copied from another store is reproduced verbatim (no number reformatting).

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace multio::action::zarr {

//----------------------------------------------------------------------------------------------------------------------
// JSON writing

/// Quoted and escaped JSON string
std::string jsonString(std::string_view s);

/// JSON array of integers, e.g. [4,90,69,49]
std::string jsonIntArray(const std::vector<std::int64_t>& v);

/// JSON array of strings, e.g. ["time","y","x"]
std::string jsonStringArray(const std::vector<std::string>& v);

//----------------------------------------------------------------------------------------------------------------------
// JSON scanning (raw)

/// Position just behind the JSON value that starts at (or after whitespace from) `pos`
std::size_t jsonSkipValue(std::string_view s, std::size_t pos);

/// Members (key, raw value text) of a JSON object, in document order
std::vector<std::pair<std::string, std::string>> jsonObjectMembers(std::string_view raw);

/// Elements (raw text) of a JSON array
std::vector<std::string> jsonArrayElements(std::string_view raw);

/// Value of a JSON string literal (handles the common escapes)
std::string jsonUnquote(std::string_view raw);

/// Raw text of member `key` of a JSON object, empty if absent
std::string jsonMember(std::string_view raw, std::string_view key);

//----------------------------------------------------------------------------------------------------------------------
// File system

/// mkdir -p
void makeDirs(const std::string& path);

/// Write `data` to `<path>` through a temporary file in the same directory and a rename,
/// so that readers never see a partial file.
void writeFileAtomic(const std::string& path, const void* data, std::size_t nbytes);

inline void writeTextAtomic(const std::string& path, const std::string& text) {
    writeFileAtomic(path, text.data(), text.size());
}

std::string readFile(const std::string& path);

bool pathExists(const std::string& path);

void renamePath(const std::string& from, const std::string& to);

/// Remove a file, ignoring errors
void removeFile(const std::string& path) noexcept;

/// Cache of directories known to exist, to avoid repeated mkdir system calls
class DirectoryCache {
public:
    void ensure(const std::string& dir);
    void clear() { known_.clear(); }

private:
    std::unordered_set<std::string> known_;
};

/// Directory part of a path ("a/b/c" -> "a/b")
std::string dirName(const std::string& path);

}  // namespace multio::action::zarr
