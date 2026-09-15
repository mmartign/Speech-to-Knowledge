// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/verbal_response.h.
#include "../speech_core/verbal_response.h"

#include <catch2/catch_test_macros.hpp>

using namespace drone::speech_core;

TEST_CASE("assessVerbalResponse reports no response for an empty transcript", "[verbal_response]")
{
    const auto result = assessVerbalResponse("");
    REQUIRE_FALSE(result.responded);
    REQUIRE_FALSE(result.distressLanguage);
    REQUIRE(result.matchedKeywords.empty());
}

TEST_CASE("assessVerbalResponse reports no response for a whitespace-only transcript", "[verbal_response]")
{
    const auto result = assessVerbalResponse("   \n\t ");
    REQUIRE_FALSE(result.responded);
}

TEST_CASE("assessVerbalResponse reports a coherent response with no distress language", "[verbal_response]")
{
    const auto result = assessVerbalResponse("I am okay, over here.");
    REQUIRE(result.responded);
    REQUIRE_FALSE(result.distressLanguage);
    REQUIRE(result.matchedKeywords.empty());
}

TEST_CASE("assessVerbalResponse detects a distress keyword", "[verbal_response]")
{
    const auto result = assessVerbalResponse("Please help, I am hurt.");
    REQUIRE(result.responded);
    REQUIRE(result.distressLanguage);
    REQUIRE_FALSE(result.matchedKeywords.empty());
}

TEST_CASE("assessVerbalResponse detects a severe keyword phrase", "[verbal_response]")
{
    const auto result = assessVerbalResponse("I am trapped and I can't breathe.");
    REQUIRE(result.distressLanguage);
}

TEST_CASE("assessVerbalResponse matches Italian keywords", "[verbal_response]")
{
    const auto result = assessVerbalResponse("Aiuto, sono ferito.");
    REQUIRE(result.distressLanguage);
}

TEST_CASE("assessVerbalResponse is case-insensitive", "[verbal_response]")
{
    const auto result = assessVerbalResponse("HELP ME PLEASE");
    REQUIRE(result.distressLanguage);
}
