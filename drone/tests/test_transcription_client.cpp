// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/transcription_client.h.
#include "../speech_core/transcription_client.h"

#include <catch2/catch_test_macros.hpp>

using namespace drone::speech_core;

TEST_CASE("extractTranscriptText handles the primary OpenAI-compatible shape", "[transcription]")
{
    const json response = {{"text", "help me please"}};
    REQUIRE(extractTranscriptText(response) == "help me please");
}

TEST_CASE("extractTranscriptText falls back to a transcript field", "[transcription]")
{
    const json response = {{"transcript", "I am trapped"}};
    REQUIRE(extractTranscriptText(response) == "I am trapped");
}

TEST_CASE("extractTranscriptText falls back to results[0].text", "[transcription]")
{
    const json response = {
        {"results", json::array({{{"text", "over here"}}})}
    };
    REQUIRE(extractTranscriptText(response) == "over here");
}

TEST_CASE("extractTranscriptText returns empty for unrecognized shapes", "[transcription]")
{
    const json response = {{"unexpected", "shape"}};
    REQUIRE(extractTranscriptText(response).empty());

    const json emptyResults = {{"results", json::array()}};
    REQUIRE(extractTranscriptText(emptyResults).empty());
}
