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
// The report specifies "a convolutional classifier for non-verbal
// distress" -- a trained CNN, presumably over acoustic features like a
// mel-spectrogram, recognizing distress sounds (screaming, moaning,
// crying) that carry no recognizable words. This environment has no
// training data and no training pipeline to produce that model, so
// classifyNonVerbalDistress() below is NOT it: it's a documented
// placeholder using simple acoustic heuristics (vocal-intensity and
// pitch-irregularity proxies) as a weak stand-in, behind the same
// interface a trained model would sit behind. Treat its output as
// indicative at best, not as what the report's system actually does.
//
// This is a distinct concern from verbal_response.h, which looks at
// what the ASR transcript actually says.
#pragma once

#include "audio_frame.h"

namespace drone::speech_core {

enum class NonVerbalDistressLevel {
    None,
    Elevated,
    High,
};

struct NonVerbalDistressResult {
    NonVerbalDistressLevel level = NonVerbalDistressLevel::None;
    double confidence = 0.0;  // 0..1, heuristic, not a calibrated probability
};

NonVerbalDistressResult classifyNonVerbalDistress(const MonoFrame& segment);

}  // namespace drone::speech_core
