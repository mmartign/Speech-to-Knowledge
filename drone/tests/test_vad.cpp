// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/vad.h.
#include "../speech_core/vad.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace drone::speech_core;

namespace {

// A synthetic "voiced speech-like" tone: moderate amplitude, moderate
// frequency, giving both RMS energy and zero-crossing rate in speech-
// like ranges under the default VadConfig.
std::vector<float> makeToneBurst(size_t n, float amplitude, double freqRadPerSample)
{
    std::vector<float> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = amplitude * static_cast<float>(std::sin(static_cast<double>(i) * freqRadPerSample));
    }
    return s;
}

}  // namespace

TEST_CASE("detectSpeechSegments finds a tone burst surrounded by silence", "[vad]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;

    std::vector<float> silenceBefore(1024, 0.0f);
    std::vector<float> tone = makeToneBurst(2048, 0.5f, 0.2);
    std::vector<float> silenceAfter(1024, 0.0f);

    frame.samples = silenceBefore;
    frame.samples.insert(frame.samples.end(), tone.begin(), tone.end());
    frame.samples.insert(frame.samples.end(), silenceAfter.begin(), silenceAfter.end());

    const auto segments = detectSpeechSegments(frame);

    REQUIRE(segments.size() == 1);
    // The detected segment should fall within (or very close to) the
    // tone region; exact frame-boundary alignment isn't guaranteed
    // given block-based analysis, so check approximate containment.
    REQUIRE(segments[0].startSample >= 768);
    REQUIRE(segments[0].endSample <= 3328);
    REQUIRE(segments[0].endSample > segments[0].startSample);
}

TEST_CASE("detectSpeechSegments finds nothing in pure silence", "[vad]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;
    frame.samples.assign(4096, 0.0f);

    REQUIRE(detectSpeechSegments(frame).empty());
}

TEST_CASE("detectSpeechSegments drops segments shorter than the minimum length", "[vad]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;

    std::vector<float> silence(2048, 0.0f);
    // A single block's worth of tone -- shorter than the default
    // minSegmentFrames -- surrounded by silence.
    std::vector<float> blip = makeToneBurst(256, 0.5f, 0.2);

    frame.samples = silence;
    frame.samples.insert(frame.samples.end(), blip.begin(), blip.end());
    frame.samples.insert(frame.samples.end(), silence.begin(), silence.end());

    REQUIRE(detectSpeechSegments(frame).empty());
}

TEST_CASE("detectSpeechSegments bridges brief gaps within one utterance", "[vad]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;

    const auto tone = makeToneBurst(512, 0.5f, 0.2);
    const std::vector<float> briefGap(256, 0.0f);  // one block of silence

    frame.samples.insert(frame.samples.end(), tone.begin(), tone.end());
    frame.samples.insert(frame.samples.end(), briefGap.begin(), briefGap.end());
    frame.samples.insert(frame.samples.end(), tone.begin(), tone.end());

    const auto segments = detectSpeechSegments(frame);
    REQUIRE(segments.size() == 1);
}
