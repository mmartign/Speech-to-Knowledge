// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Test-only helpers shared across the drone speech_core unit tests.
#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace drone::speech_core_test {

// A unique path under the system temp directory with the given
// extension (including the dot). The counter is a function-local
// static in an inline function, so it's shared (one instance, per the
// standard's inline-function guarantee) across every test file that
// includes this header, avoiding filename collisions between them.
inline std::filesystem::path uniqueTempPath(const std::string& extension)
{
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("drone_test_" + std::to_string(counter++) + extension);
}

// Writes `contents` to a uniquely-named temp text file (e.g. an INI
// config) and returns its path. Removed when the guard goes out of scope.
class TempTextFile {
public:
    explicit TempTextFile(const std::string& contents, const std::string& extension = ".ini")
        : path_(uniqueTempPath(extension))
    {
        std::ofstream out(path_);
        out << contents;
    }

    ~TempTextFile() { std::filesystem::remove(path_); }

    TempTextFile(const TempTextFile&) = delete;
    TempTextFile& operator=(const TempTextFile&) = delete;

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

// RAII guard that removes a path (e.g. one written by the code under
// test, such as writeMonoWavFile()) on destruction.
class TempPathGuard {
public:
    explicit TempPathGuard(std::filesystem::path path) : path_(std::move(path)) {}
    ~TempPathGuard() { std::filesystem::remove(path_); }

    TempPathGuard(const TempPathGuard&) = delete;
    TempPathGuard& operator=(const TempPathGuard&) = delete;

    // Movable (not just copy-deleted) so a guard can be returned by
    // value from a helper function without relying on NRVO, which the
    // standard doesn't guarantee.
    TempPathGuard(TempPathGuard&& other) noexcept : path_(std::move(other.path_))
    {
        other.path_.clear();
    }
    TempPathGuard& operator=(TempPathGuard&& other) noexcept
    {
        if (this != &other) {
            std::filesystem::remove(path_);
            path_ = std::move(other.path_);
            other.path_.clear();
        }
        return *this;
    }

    const std::filesystem::path& path() const { return path_; }
    std::string string() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

}  // namespace drone::speech_core_test
