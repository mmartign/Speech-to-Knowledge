// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/vocal_band_filter.h.
#include "../speech_core/vocal_band_filter.h"

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
    const size_t skip = std::min(samples.size(), samples.size() / 4);
    double sumSq = 0.0;
    for (size_t i = skip; i < samples.size(); ++i) {
        sumSq += static_cast<double>(samples[i]) * samples[i];
    }
    const size_t count = samples.size() - skip;
    return count > 0 ? std::sqrt(sumSq / static_cast<double>(count)) : 0.0;
}

}  // namespace

TEST_CASE("applyVocalBandFilter passes a mid-band tone through with little loss", "[vocal_band]")
{
    auto frame = makeTone(16000, 1000.0, 4000);  // well inside [300, 3400] Hz
    const double rmsBefore = rmsAfterSettling(frame.samples);

    applyVocalBandFilter(frame);
    const double rmsAfter = rmsAfterSettling(frame.samples);

    REQUIRE(rmsAfter > rmsBefore * 0.7);
}

TEST_CASE("applyVocalBandFilter strongly attenuates a low-frequency tone", "[vocal_band]")
{
    auto frame = makeTone(16000, 60.0, 4000);  // well below the 300 Hz low cutoff
    const double rmsBefore = rmsAfterSettling(frame.samples);

    applyVocalBandFilter(frame);
    const double rmsAfter = rmsAfterSettling(frame.samples);

    REQUIRE(rmsAfter < rmsBefore * 0.3);
}

TEST_CASE("applyVocalBandFilter strongly attenuates a high-frequency tone", "[vocal_band]")
{
    auto frame = makeTone(16000, 7000.0, 4000);  // well above the 3400 Hz high cutoff
    const double rmsBefore = rmsAfterSettling(frame.samples);

    applyVocalBandFilter(frame);
    const double rmsAfter = rmsAfterSettling(frame.samples);

    REQUIRE(rmsAfter < rmsBefore * 0.3);
}

TEST_CASE("applyVocalBandFilter is a no-op on an empty frame", "[vocal_band]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;
    applyVocalBandFilter(frame);
    REQUIRE(frame.samples.empty());
}
