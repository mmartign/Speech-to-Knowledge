// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Copyright (c) 2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
#include "spectral_subtraction.h"

#include "fft.h"

#include <algorithm>
#include <cmath>

namespace drone::speech_core {

namespace {

std::vector<float> hannWindow(int n)
{
    std::vector<float> w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        w[static_cast<size_t>(i)] =
            0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (n - 1)));
    }
    return w;
}

// FFT of one windowed analysis frame starting at `samples[offset]`.
std::vector<std::complex<float>> analyzeFrame(
    const std::vector<float>& samples,
    size_t offset,
    int fftSize,
    const std::vector<float>& window)
{
    std::vector<std::complex<float>> buf(static_cast<size_t>(fftSize));
    for (int i = 0; i < fftSize; ++i) {
        const size_t idx = offset + static_cast<size_t>(i);
        const float s = idx < samples.size() ? samples[idx] : 0.0f;
        buf[static_cast<size_t>(i)] = s * window[static_cast<size_t>(i)];
    }
    fftInPlace(buf, /*inverse=*/false);
    return buf;
}

}  // namespace

std::vector<float> estimateNoiseProfile(const MonoFrame& noiseOnlySignal, int fftSize)
{
    const int numBins = fftSize / 2 + 1;
    std::vector<float> profile(static_cast<size_t>(numBins), 0.0f);

    const auto window = hannWindow(fftSize);
    const int hop = fftSize / 2;
    const size_t n = noiseOnlySignal.samples.size();

    int frameCount = 0;
    for (size_t start = 0; start + static_cast<size_t>(fftSize) <= n;
         start += static_cast<size_t>(hop)) {
        const auto spectrum = analyzeFrame(noiseOnlySignal.samples, start, fftSize, window);
        for (int k = 0; k < numBins; ++k) {
            profile[static_cast<size_t>(k)] += std::abs(spectrum[static_cast<size_t>(k)]);
        }
        ++frameCount;
    }

    if (frameCount > 0) {
        for (float& v : profile) {
            v /= static_cast<float>(frameCount);
        }
    }
    return profile;
}

void applySpectralSubtraction(
    MonoFrame& frame,
    const SpectralSubtractionConfig& config,
    const std::vector<float>& noiseProfile)
{
    if (frame.samples.empty() || config.fftSize <= 0 || config.hopSize <= 0) {
        return;
    }

    const int fftSize = config.fftSize;
    const int numBins = fftSize / 2 + 1;
    const auto window = hannWindow(fftSize);

    std::vector<float> profile = noiseProfile;
    if (profile.empty()) {
        const size_t noiseSamples =
            std::min(frame.samples.size(),
                     static_cast<size_t>(config.noiseProfileFrames) * static_cast<size_t>(config.hopSize) +
                         static_cast<size_t>(fftSize));
        MonoFrame lead;
        lead.sampleRateHz = frame.sampleRateHz;
        lead.samples.assign(frame.samples.begin(), frame.samples.begin() + static_cast<long>(noiseSamples));
        profile = estimateNoiseProfile(lead, fftSize);
    }
    if (static_cast<int>(profile.size()) != numBins) {
        profile.resize(static_cast<size_t>(numBins), 0.0f);
    }

    const size_t originalLength = frame.samples.size();
    std::vector<float> output(originalLength + static_cast<size_t>(fftSize), 0.0f);

    for (size_t start = 0; start < originalLength; start += static_cast<size_t>(config.hopSize)) {
        auto spectrum = analyzeFrame(frame.samples, start, fftSize, window);

        for (int k = 0; k < numBins; ++k) {
            const auto& bin = spectrum[static_cast<size_t>(k)];
            const float mag = std::abs(bin);
            const float phase = std::arg(bin);

            const float subtracted = mag - config.oversubtractionFactor * profile[static_cast<size_t>(k)];
            const float newMag = std::max(subtracted, config.spectralFloor * mag);

            const std::complex<float> newBin =
                std::polar(newMag, phase);
            spectrum[static_cast<size_t>(k)] = newBin;

            // Mirror to the conjugate-symmetric upper half so the
            // inverse transform of this real-valued-input signal stays
            // real; DC (k=0) and Nyquist (k=fftSize/2) have no partner.
            if (k > 0 && k < fftSize / 2) {
                spectrum[static_cast<size_t>(fftSize - k)] = std::conj(newBin);
            }
        }

        fftInPlace(spectrum, /*inverse=*/true);

        for (int i = 0; i < fftSize; ++i) {
            const size_t outIdx = start + static_cast<size_t>(i);
            if (outIdx < output.size()) {
                output[outIdx] += spectrum[static_cast<size_t>(i)].real() * window[static_cast<size_t>(i)];
            }
        }
    }

    output.resize(originalLength);
    frame.samples = std::move(output);
}

}  // namespace drone::speech_core
