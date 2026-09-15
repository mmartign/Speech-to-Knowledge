// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/fft.h.
#include "../speech_core/fft.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace drone::speech_core;

TEST_CASE("fftInPlace rejects a non-power-of-two size", "[fft]")
{
    std::vector<std::complex<float>> data(100, {1.0f, 0.0f});
    REQUIRE_FALSE(fftInPlace(data, false));
}

TEST_CASE("fftInPlace accepts size 1 as a no-op", "[fft]")
{
    std::vector<std::complex<float>> data = {{3.0f, 2.0f}};
    REQUIRE(fftInPlace(data, false));
    REQUIRE(data[0] == std::complex<float>(3.0f, 2.0f));
}

TEST_CASE("forward FFT then inverse FFT round-trips the original signal", "[fft]")
{
    constexpr size_t n = 64;
    std::vector<std::complex<float>> original(n);
    for (size_t i = 0; i < n; ++i) {
        original[i] = std::complex<float>(
            static_cast<float>(std::sin(static_cast<double>(i) * 0.3)), 0.0f);
    }

    auto data = original;
    REQUIRE(fftInPlace(data, false));
    REQUIRE(fftInPlace(data, true));

    for (size_t i = 0; i < n; ++i) {
        REQUIRE(std::abs(data[i] - original[i]) < 1e-4f);
    }
}

TEST_CASE("FFT of a pure tone concentrates energy at the expected bin", "[fft]")
{
    constexpr size_t n = 64;
    constexpr int binIndex = 8;  // frequency = binIndex * fs / n

    std::vector<std::complex<float>> data(n);
    for (size_t i = 0; i < n; ++i) {
        const double angle = 2.0 * M_PI * binIndex * static_cast<double>(i) / static_cast<double>(n);
        data[i] = std::complex<float>(static_cast<float>(std::sin(angle)), 0.0f);
    }

    REQUIRE(fftInPlace(data, false));

    // Energy should be concentrated at bin 8 and its mirror (n - 8);
    // every other bin should be comparatively negligible.
    const float targetMag = std::abs(data[binIndex]);
    REQUIRE(targetMag > 1.0f);

    for (size_t k = 0; k < n; ++k) {
        if (k == binIndex || k == n - binIndex) {
            continue;
        }
        REQUIRE(std::abs(data[k]) < targetMag * 0.05f);
    }
}
