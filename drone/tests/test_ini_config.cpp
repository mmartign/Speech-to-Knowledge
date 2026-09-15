// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/ini_config.h.
#include "../speech_core/ini_config.h"

#include "test_utils.h"

#include <catch2/catch_test_macros.hpp>

using namespace drone::speech_core;
using drone::speech_core_test::TempTextFile;

TEST_CASE("parseIni flattens sections into dotted keys", "[ini]")
{
    TempTextFile file(
        "[asr]\n"
        "base_url = http://localhost:8081/v1/\n"
        "model_name = whisper-quantized\n");

    const auto config = parseIni(file.path());
    REQUIRE(config.at("asr.base_url") == "http://localhost:8081/v1/");
    REQUIRE(config.at("asr.model_name") == "whisper-quantized");
}

TEST_CASE("loadAsrConfig succeeds without an api_key", "[ini]")
{
    TempTextFile file(
        "[asr]\n"
        "base_url = http://localhost:8081/v1\n"
        "model_name = whisper-quantized\n");

    AsrConfig cfg;
    REQUIRE(loadAsrConfig(file.path(), cfg));
    REQUIRE(cfg.baseUrl == "http://localhost:8081/v1/");  // trailing slash added
    REQUIRE(cfg.modelName == "whisper-quantized");
    REQUIRE(cfg.apiKey.empty());
}

TEST_CASE("loadAsrConfig fails when base_url or model_name is missing", "[ini]")
{
    TempTextFile file("[asr]\nbase_url = http://localhost:8081/v1\n");

    AsrConfig cfg;
    REQUIRE_FALSE(loadAsrConfig(file.path(), cfg));
}

TEST_CASE("loadAsrConfig fails for a missing file", "[ini]")
{
    AsrConfig cfg;
    REQUIRE_FALSE(loadAsrConfig("/nonexistent/drone_config.ini", cfg));
}
