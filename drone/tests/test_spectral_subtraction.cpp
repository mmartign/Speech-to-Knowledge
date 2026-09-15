// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/spectral_subtraction.h.
#include "../speech_core/spectral_subtraction.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>

using namespace drone::speech_core;

namespace {

std::vector<float> makeNoise(size_t n, unsigned seed, float amplitude)
{
    std::minstd_rand rng(seed);
    std::uniform_real_distribution<float> dist(-amplitude, amplitude);
    std::vector<float> out(n);
    for (float& s : out) {
        s = dist(rng);
    }
    return out;
}

double rms(const std::vector<float>& samples)
{
    if (samples.empty()) {
        return 0.0;
    }
    double sumSq = 0.0;
    for (float s : samples) {
        sumSq += static_cast<double>(s) * s;
    }
    return std::sqrt(sumSq / static_cast<double>(samples.size()));
}

}  // namespace

TEST_CASE("estimateNoiseProfile returns a positive profile for real noise", "[spectral_subtraction]")
{
    MonoFrame noise;
    noise.sampleRateHz = 16000;
    noise.samples = makeNoise(4096, /*seed=*/1, 0.3f);

    const auto profile = estimateNoiseProfile(noise, 512);
    REQUIRE(profile.size() == 512 / 2 + 1);

    double sum = 0.0;
    for (float v : profile) {
        sum += v;
    }
    REQUIRE(sum > 0.0);
}

TEST_CASE("estimateNoiseProfile returns all zeros for a signal shorter than one FFT frame", "[spectral_subtraction]")
{
    MonoFrame tooShort;
    tooShort.sampleRateHz = 16000;
    tooShort.samples = makeNoise(100, /*seed=*/1, 0.3f);  // < 512

    const auto profile = estimateNoiseProfile(tooShort, 512);
    for (float v : profile) {
        REQUIRE(v == 0.0f);
    }
}

TEST_CASE("applySpectralSubtraction reduces the energy of stationary noise", "[spectral_subtraction]")
{
    MonoFrame frame;
    frame.sampleRateHz = 16000;
    frame.samples = makeNoise(8192, /*seed=*/2, 0.3f);

    const double rmsBefore = rms(frame.samples);
    applySpectralSubtraction(frame);  // self-estimates the profile from its own lead-in
    const double rmsAfter = rms(frame.samples);

    REQUIRE(rmsAfter < rmsBefore * 0.7);
}

TEST_CASE("applySpectralSubtraction preserves most of a tone's energy given an accurate noise profile", "[spectral_subtraction]")
{
    constexpr int sampleRate = 16000;
    constexpr size_t n = 8192;

    // Calibration: noise alone, used to build an accurate profile.
    MonoFrame calibration;
    calibration.sampleRateHz = sampleRate;
    calibration.samples = makeNoise(n, /*seed=*/3, 0.2f);
    const auto profile = estimateNoiseProfile(calibration, 512);

    // Test signal: the same noise characteristics plus a strong tone
    // (standing in for speech), using a different noise seed so this
    // isn't literally the calibration signal itself.
    MonoFrame frame;
    frame.sampleRateHz = sampleRate;
    frame.samples = makeNoise(n, /*seed=*/4, 0.2f);
    for (size_t i = 0; i < n; ++i) {
        frame.samples[i] += 0.8f * static_cast<float>(std::sin(2.0 * M_PI * 1000.0 * i / sampleRate));
    }

    SpectralSubtractionConfig config;
    applySpectralSubtraction(frame, config, profile);

    // A pure 0.8-amplitude tone alone would have RMS ~= 0.8/sqrt(2) =~ 0.566.
    // After removing most of the noise, the result should still be in
    // that neighborhood, not collapsed toward zero.
    const double rmsAfter = rms(frame.samples);
    REQUIRE(rmsAfter > 0.3);
}
