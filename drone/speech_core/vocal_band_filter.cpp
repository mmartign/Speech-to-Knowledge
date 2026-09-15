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
#include "vocal_band_filter.h"

#include "biquad.h"

namespace drone::speech_core {

void applyVocalBandFilter(MonoFrame& frame, const VocalBandFilterConfig& config)
{
    if (frame.sampleRateHz <= 0 || frame.samples.empty()) {
        return;
    }

    Biquad highpass = Biquad::makeHighpass(config.lowCutoffHz, frame.sampleRateHz, config.q);
    Biquad lowpass = Biquad::makeLowpass(config.highCutoffHz, frame.sampleRateHz, config.q);

    for (float& sample : frame.samples) {
        sample = lowpass.process(highpass.process(sample));
    }
}

}  // namespace drone::speech_core
