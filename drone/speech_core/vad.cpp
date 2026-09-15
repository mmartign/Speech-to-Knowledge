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
#include "vad.h"

#include <algorithm>
#include <cmath>

namespace drone::speech_core {

namespace {

float rms(const float* data, size_t count)
{
    if (count == 0) {
        return 0.0f;
    }
    double sumSq = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sumSq += static_cast<double>(data[i]) * data[i];
    }
    return static_cast<float>(std::sqrt(sumSq / static_cast<double>(count)));
}

// Fraction of adjacent-sample sign changes, a standard proxy for how
// much high-frequency content a frame has.
float zeroCrossingRate(const float* data, size_t count)
{
    if (count < 2) {
        return 0.0f;
    }
    size_t crossings = 0;
    for (size_t i = 1; i < count; ++i) {
        const bool signChanged =
            (data[i - 1] >= 0.0f) != (data[i] >= 0.0f);
        if (signChanged) {
            ++crossings;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(count - 1);
}

}  // namespace

std::vector<SpeechSegment> detectSpeechSegments(
    const MonoFrame& frame,
    const VadConfig& config)
{
    std::vector<SpeechSegment> segments;

    const size_t n = frame.samples.size();
    if (n == 0 || config.frameSamples <= 0) {
        return segments;
    }

    const size_t blockSize = static_cast<size_t>(config.frameSamples);
    const size_t numBlocks = (n + blockSize - 1) / blockSize;

    std::vector<bool> isHit(numBlocks, false);
    for (size_t b = 0; b < numBlocks; ++b) {
        const size_t start = b * blockSize;
        const size_t count = std::min(blockSize, n - start);
        const float level = rms(&frame.samples[start], count);
        const float zcr = zeroCrossingRate(&frame.samples[start], count);
        isHit[b] = level > config.energyThreshold &&
                   zcr >= config.minZcr && zcr <= config.maxZcr;
    }

    // Merge hits into segments, bridging gaps up to maxGapFrames.
    int gapRun = 0;
    bool inSegment = false;
    size_t segmentStartBlock = 0;
    size_t lastHitBlock = 0;

    auto closeSegment = [&](size_t endBlockExclusive) {
        const size_t startSample = segmentStartBlock * blockSize;
        const size_t endSample = std::min(endBlockExclusive * blockSize, n);
        const size_t lengthBlocks = endBlockExclusive - segmentStartBlock;
        if (static_cast<int>(lengthBlocks) >= config.minSegmentFrames) {
            segments.push_back({startSample, endSample});
        }
    };

    for (size_t b = 0; b < numBlocks; ++b) {
        if (isHit[b]) {
            if (!inSegment) {
                inSegment = true;
                segmentStartBlock = b;
            }
            lastHitBlock = b;
            gapRun = 0;
        } else if (inSegment) {
            ++gapRun;
            if (gapRun > config.maxGapFrames) {
                closeSegment(lastHitBlock + 1);
                inSegment = false;
                gapRun = 0;
            }
        }
    }
    if (inSegment) {
        closeSegment(lastHitBlock + 1);
    }

    return segments;
}

}  // namespace drone::speech_core
