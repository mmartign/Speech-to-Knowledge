// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/non_verbal_distress.h. Note: this tests
// the documented heuristic placeholder, not "a convolutional
// classifier" -- see the header comment for why there isn't one here.
#include "../speech_core/non_verbal_distress.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>

using namespace drone::speech_core;

TEST_CASE("classifyNonVerbalDistress reports None for a calm steady tone", "[non_verbal_distress]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;
    for (size_t i = 0; i < 2000; ++i) {
        frame.samples.push_back(
            0.3f * static_cast<float>(std::sin(2.0 * M_PI * 200.0 * i / 16000)));
    }

    const auto result = classifyNonVerbalDistress(frame);
    REQUIRE(result.level == NonVerbalDistressLevel::None);
    REQUIRE(result.confidence < 0.3);
}

TEST_CASE("classifyNonVerbalDistress flags a signal with sharp intensity spikes", "[non_verbal_distress]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;
    std::minstd_rand rng(7);
    std::uniform_real_distribution<float> noiseDist(-0.02f, 0.02f);

    for (size_t i = 0; i < 3000; ++i) {
        float v = noiseDist(rng);
        const float freq = ((i / 256) % 2 == 0) ? 150.0f : 900.0f;
        v += 0.05f * static_cast<float>(std::sin(2.0 * M_PI * freq * i / 16000));
        if (i % 500 == 0) {
            v = 1.0f;  // sharp, scream-like spike
        }
        frame.samples.push_back(v);
    }

    const auto result = classifyNonVerbalDistress(frame);
    REQUIRE(result.level != NonVerbalDistressLevel::None);
    REQUIRE(result.confidence > 0.0);
}

TEST_CASE("classifyNonVerbalDistress on an empty frame returns None", "[non_verbal_distress]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;
    const auto result = classifyNonVerbalDistress(frame);
    REQUIRE(result.level == NonVerbalDistressLevel::None);
    REQUIRE(result.confidence == 0.0);
}
