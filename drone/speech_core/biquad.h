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
// A single second-order IIR filter (biquad) in direct form I, per the
// standard RBJ Audio EQ Cookbook design formulas. Shared primitive
// behind notch_filter.h and vocal_band_filter.h -- not part of the
// public pipeline API on its own.
#pragma once

namespace drone::speech_core {

struct Biquad {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;  // a0 normalized to 1
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;

    static Biquad makeNotch(double freqHz, double sampleRateHz, double q);
    static Biquad makeLowpass(double cutoffHz, double sampleRateHz, double q);
    static Biquad makeHighpass(double cutoffHz, double sampleRateHz, double q);

    float process(float input);
};

}  // namespace drone::speech_core
