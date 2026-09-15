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
#include "non_verbal_distress.h"

#include <algorithm>
#include <cmath>

namespace drone::speech_core {

namespace {

constexpr int kBlockSize = 256;

// Fraction of adjacent-sample sign changes within [start, start+count).
float zeroCrossingRate(const std::vector<float>& samples, size_t start, size_t count)
{
    if (count < 2) {
        return 0.0f;
    }
    size_t crossings = 0;
    for (size_t i = start + 1; i < start + count; ++i) {
        if ((samples[i - 1] >= 0.0f) != (samples[i] >= 0.0f)) {
            ++crossings;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(count - 1);
}

}  // namespace

NonVerbalDistressResult classifyNonVerbalDistress(const MonoFrame& segment)
{
    NonVerbalDistressResult result;
    const auto& samples = segment.samples;
    if (samples.empty()) {
        return result;
    }

    // Peak-to-RMS ratio: a crude proxy for vocal intensity -- a scream
    // or shout has more transient energy spikes than steady speech at
    // the same average loudness.
    double sumSq = 0.0;
    float peak = 0.0f;
    for (float s : samples) {
        sumSq += static_cast<double>(s) * s;
        peak = std::max(peak, std::fabs(s));
    }
    const float rms = static_cast<float>(std::sqrt(sumSq / static_cast<double>(samples.size())));
    const float peakRatio = rms > 1e-6f ? peak / rms : 1.0f;

    // Zero-crossing-rate variance across sub-blocks: an irregular pitch
    // contour (a strained scream, a sob) swings the ZCR block-to-block
    // more than the comparatively steady pitch of calm spoken vowels.
    std::vector<float> blockZcr;
    for (size_t start = 0; start + kBlockSize <= samples.size();
         start += static_cast<size_t>(kBlockSize)) {
        blockZcr.push_back(zeroCrossingRate(samples, start, static_cast<size_t>(kBlockSize)));
    }

    float zcrVariance = 0.0f;
    if (blockZcr.size() > 1) {
        float mean = 0.0f;
        for (float z : blockZcr) {
            mean += z;
        }
        mean /= static_cast<float>(blockZcr.size());
        for (float z : blockZcr) {
            zcrVariance += (z - mean) * (z - mean);
        }
        zcrVariance /= static_cast<float>(blockZcr.size());
    }

    double score = 0.0;
    if (peakRatio > 5.0f) {
        score += 0.5;
    }
    if (zcrVariance > 0.01f) {
        score += 0.3;
    }
    if (peakRatio > 8.0f) {
        score += 0.2;
    }
    score = std::min(1.0, score);

    result.confidence = score;
    if (score >= 0.6) {
        result.level = NonVerbalDistressLevel::High;
    } else if (score >= 0.3) {
        result.level = NonVerbalDistressLevel::Elevated;
    } else {
        result.level = NonVerbalDistressLevel::None;
    }
    return result;
}

}  // namespace drone::speech_core
