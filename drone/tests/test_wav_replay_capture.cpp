// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for io/wav_replay_capture.h, and incidentally the only
// coverage of readWavFile()'s multi-channel de-interleaving path (see
// test_wav_file.cpp for the mono round-trip).
#include "../io/wav_replay_capture.h"
#include "../io/wav_file.h"

#include "test_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace drone::speech_core;
using namespace drone::io;
using drone::speech_core_test::TempPathGuard;
using drone::speech_core_test::uniqueTempPath;

namespace {

TempPathGuard makeFourChannelFixture()
{
    MultiChannelFrame frame;
    frame.sampleRateHz = 16000;
    frame.channels.resize(4);
    for (int c = 0; c < 4; ++c) {
        auto& channel = frame.channels[static_cast<size_t>(c)];
        channel.resize(100);
        for (int i = 0; i < 100; ++i) {
            // Distinct, easily-checkable values per channel.
            channel[static_cast<size_t>(i)] = static_cast<float>(c + 1) * 0.1f;
        }
    }

    TempPathGuard path(uniqueTempPath(".wav"));
    REQUIRE(writeMultiChannelWavFile(path.string(), frame));
    return path;
}

}  // namespace

TEST_CASE("WavReplayCapture reports the fixture's channel count and sample rate", "[wav_replay]")
{
    const auto fixture = makeFourChannelFixture();
    WavReplayCapture capture(fixture.string());

    REQUIRE(capture.ok());
    REQUIRE(capture.numChannels() == 4);
    REQUIRE(capture.sampleRateHz() == 16000);
}

TEST_CASE("WavReplayCapture de-interleaves channels correctly", "[wav_replay]")
{
    const auto fixture = makeFourChannelFixture();
    WavReplayCapture capture(fixture.string());
    REQUIRE(capture.ok());

    MultiChannelFrame out;
    REQUIRE(capture.readFrame(out, 100));
    REQUIRE(out.numChannels() == 4);
    REQUIRE(out.numSamples() == 100);

    for (int c = 0; c < 4; ++c) {
        const float expected = static_cast<float>(c + 1) * 0.1f;
        REQUIRE(std::fabs(out.channels[static_cast<size_t>(c)][0] - expected) < 0.001f);
    }
}

TEST_CASE("WavReplayCapture reads in chunks and reports exhaustion", "[wav_replay]")
{
    const auto fixture = makeFourChannelFixture();
    WavReplayCapture capture(fixture.string());
    REQUIRE(capture.ok());

    MultiChannelFrame out;
    REQUIRE(capture.readFrame(out, 40));  // 40 of 100
    REQUIRE(out.numSamples() == 40);

    REQUIRE(capture.readFrame(out, 40));  // 40 more, 80 of 100 total
    REQUIRE(out.numSamples() == 80);

    REQUIRE(capture.readFrame(out, 40));  // only 20 left, but still "more data"
    REQUIRE(out.numSamples() == 100);

    MultiChannelFrame exhausted;
    REQUIRE_FALSE(capture.readFrame(exhausted, 10));  // nothing left
}

TEST_CASE("WavReplayCapture reports failure for a nonexistent file", "[wav_replay]")
{
    WavReplayCapture capture("/nonexistent/fixture.wav");
    REQUIRE_FALSE(capture.ok());

    MultiChannelFrame out;
    REQUIRE_FALSE(capture.readFrame(out, 10));
}
