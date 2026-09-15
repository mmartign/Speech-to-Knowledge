// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/notch_filter.h.
#include "../speech_core/notch_filter.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

using namespace drone::speech_core;

namespace {

MonoFrame makeTone(int sampleRateHz, double freqHz, size_t n)
{
    MonoFrame frame;
    frame.sampleRateHz = sampleRateHz;
    frame.samples.resize(n);
    for (size_t i = 0; i < n; ++i) {
        frame.samples[i] = static_cast<float>(
            std::sin(2.0 * M_PI * freqHz * static_cast<double>(i) / sampleRateHz));
    }
    return frame;
}

double rmsAfterSettling(const std::vector<float>& samples)
{
    // Skip the filter's initial transient; measure steady-state RMS.
    const size_t skip = std::min(samples.size(), samples.size() / 4);
    double sumSq = 0.0;
    for (size_t i = skip; i < samples.size(); ++i) {
        sumSq += static_cast<double>(samples[i]) * samples[i];
    }
    const size_t count = samples.size() - skip;
    return count > 0 ? std::sqrt(sumSq / static_cast<double>(count)) : 0.0;
}

}  // namespace

TEST_CASE("applyAdaptiveNotchFilter strongly attenuates a tone at the blade-pass frequency", "[notch]")
{
    MotorTelemetry telemetry;
    telemetry.rpm = 6000.0;       // 6000/60 * 2 blades = 200 Hz fundamental
    telemetry.bladesPerRotor = 2;

    auto frame = makeTone(16000, telemetry.bladePassHz(), 4000);
    const double rmsBefore = rmsAfterSettling(frame.samples);

    applyAdaptiveNotchFilter(frame, telemetry);
    const double rmsAfter = rmsAfterSettling(frame.samples);

    REQUIRE(rmsAfter < rmsBefore * 0.1);
}

TEST_CASE("applyAdaptiveNotchFilter leaves an off-harmonic tone largely intact", "[notch]")
{
    MotorTelemetry telemetry;
    telemetry.rpm = 6000.0;  // harmonics at 200, 400, 600 Hz
    telemetry.bladesPerRotor = 2;

    auto frame = makeTone(16000, 1000.0, 4000);  // well clear of any harmonic
    const double rmsBefore = rmsAfterSettling(frame.samples);

    applyAdaptiveNotchFilter(frame, telemetry);
    const double rmsAfter = rmsAfterSettling(frame.samples);

    REQUIRE(rmsAfter > rmsBefore * 0.8);
}

TEST_CASE("applyAdaptiveNotchFilter is a no-op with no motor telemetry", "[notch]")
{
    MotorTelemetry telemetry;  // rpm defaults to 0
    auto frame = makeTone(16000, 200.0, 1000);
    const auto before = frame.samples;

    applyAdaptiveNotchFilter(frame, telemetry);
    REQUIRE(frame.samples == before);
}
