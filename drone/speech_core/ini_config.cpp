// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Copyright (c) 2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
#include "ini_config.h"

#include <fstream>
#include <iostream>
#include <vector>

namespace drone::speech_core {

bool trimInPlace(std::string& s)
{
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        s.clear();
        return false;
    }

    const auto end = s.find_last_not_of(" \t\r\n");
    s = s.substr(start, end - start + 1);
    return true;
}

std::map<std::string, std::string> parseIni(const std::string& filename)
{
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    if (!file.is_open()) {
        return config;
    }

    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        if (!trimInPlace(line)) {
            continue;
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] == ';' || line[0] == '#') {
            continue;
        }

        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            trimInPlace(section);
            continue;
        }

        const size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, eqPos);
        std::string value = line.substr(eqPos + 1);

        if (!trimInPlace(key)) {
            continue;
        }
        trimInPlace(value);

        if (!section.empty()) {
            key = section + "." + key;
        }

        config[key] = value;
    }

    return config;
}

bool loadAsrConfig(const std::string& path, AsrConfig& cfg)
{
    const auto config = parseIni(path);

    auto getValue = [&](const std::string& key, std::string& out) {
        const auto it = config.find(key);
        if (it != config.end()) {
            out = it->second;
        }
    };

    getValue("asr.base_url", cfg.baseUrl);
    getValue("asr.api_key", cfg.apiKey);
    getValue("asr.model_name", cfg.modelName);

    std::vector<std::string> missing;
    if (cfg.baseUrl.empty()) {
        missing.push_back("asr.base_url");
    }
    if (cfg.modelName.empty()) {
        missing.push_back("asr.model_name");
    }
    // asr.api_key is intentionally not required: a local transcription
    // server on the same Jetson typically has no auth at all.

    if (!missing.empty()) {
        std::cerr << "[ERROR] Missing config values in " << path << ":";
        for (const auto& key : missing) {
            std::cerr << ' ' << key;
        }
        std::cerr << "\n";
        return false;
    }

    if (!cfg.baseUrl.empty() && cfg.baseUrl.back() != '/') {
        cfg.baseUrl += "/";
    }
    return true;
}

}  // namespace drone::speech_core
