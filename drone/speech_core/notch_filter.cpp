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
#include "notch_filter.h"

#include "biquad.h"

#include <vector>

namespace drone::speech_core {

void applyAdaptiveNotchFilter(
    MonoFrame& frame,
    const MotorTelemetry& telemetry,
    const NotchFilterConfig& config)
{
    if (telemetry.rpm <= 0.0 || frame.sampleRateHz <= 0 || frame.samples.empty()) {
        return;
    }

    const double nyquist = frame.sampleRateHz / 2.0;
    const double fundamental = telemetry.bladePassHz();

    std::vector<Biquad> notches;
    for (int h = 1; h <= config.numHarmonics; ++h) {
        const double freq = fundamental * h;
        if (freq <= 0.0 || freq >= nyquist) {
            continue;
        }
        notches.push_back(Biquad::makeNotch(freq, frame.sampleRateHz, config.q));
    }

    for (float& sample : frame.samples) {
        float value = sample;
        for (auto& notch : notches) {
            value = notch.process(value);
        }
        sample = value;
    }
}

}  // namespace drone::speech_core
