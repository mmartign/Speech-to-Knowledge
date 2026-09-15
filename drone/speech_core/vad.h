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
// Energy + zero-crossing-rate voice activity detection. Classic,
// dependency-free VAD suitable as a gate in front of speech recognition:
// speech tends to combine moderate-to-high short-term energy with a
// zero-crossing rate in a mid range (voiced speech has more structure
// than broadband noise, but crosses zero less often than high-frequency
// noise/hiss).
#pragma once

#include "audio_frame.h"

#include <cstddef>
#include <vector>

namespace drone::speech_core {

struct VadConfig {
    int frameSamples = 256;

    // A frame is a VAD "hit" if its RMS energy exceeds this absolute
    // threshold AND its zero-crossing rate falls within
    // [minZcr, maxZcr]. Callers processing already-gain-normalized
    // audio (e.g. output of applyNoiseGate()) can tune energyThreshold
    // relative to their own signal scale.
    float energyThreshold = 0.02f;
    float minZcr = 0.02f;
    float maxZcr = 0.35f;

    // Hit frames are merged into segments; gaps of up to this many
    // consecutive non-hit frames are bridged (avoids fragmenting one
    // utterance on brief dips), and segments shorter than
    // minSegmentFrames after merging are dropped as spurious.
    int maxGapFrames = 3;
    int minSegmentFrames = 4;
};

// A contiguous span of speech-like audio, expressed in sample indices
// into the MonoFrame that was analyzed.
struct SpeechSegment {
    size_t startSample = 0;
    size_t endSample = 0;  // exclusive
};

// Detect speech segments in `frame` per `config`. Returns segments in
// ascending order; segments never overlap.
std::vector<SpeechSegment> detectSpeechSegments(
    const MonoFrame& frame,
    const VadConfig& config = {});

}  // namespace drone::speech_core
