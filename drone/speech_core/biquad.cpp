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
#include "biquad.h"

#include <cmath>

namespace drone::speech_core {

Biquad Biquad::makeNotch(double freqHz, double sampleRateHz, double q)
{
    const double w0 = 2.0 * M_PI * freqHz / sampleRateHz;
    const double alpha = std::sin(w0) / (2.0 * q);
    const double cosW0 = std::cos(w0);
    const double a0 = 1.0 + alpha;

    Biquad bq;
    bq.b0 = 1.0 / a0;
    bq.b1 = -2.0 * cosW0 / a0;
    bq.b2 = 1.0 / a0;
    bq.a1 = -2.0 * cosW0 / a0;
    bq.a2 = (1.0 - alpha) / a0;
    return bq;
}

Biquad Biquad::makeLowpass(double cutoffHz, double sampleRateHz, double q)
{
    const double w0 = 2.0 * M_PI * cutoffHz / sampleRateHz;
    const double alpha = std::sin(w0) / (2.0 * q);
    const double cosW0 = std::cos(w0);
    const double a0 = 1.0 + alpha;

    Biquad bq;
    bq.b0 = ((1.0 - cosW0) / 2.0) / a0;
    bq.b1 = (1.0 - cosW0) / a0;
    bq.b2 = ((1.0 - cosW0) / 2.0) / a0;
    bq.a1 = -2.0 * cosW0 / a0;
    bq.a2 = (1.0 - alpha) / a0;
    return bq;
}

Biquad Biquad::makeHighpass(double cutoffHz, double sampleRateHz, double q)
{
    const double w0 = 2.0 * M_PI * cutoffHz / sampleRateHz;
    const double alpha = std::sin(w0) / (2.0 * q);
    const double cosW0 = std::cos(w0);
    const double a0 = 1.0 + alpha;

    Biquad bq;
    bq.b0 = ((1.0 + cosW0) / 2.0) / a0;
    bq.b1 = -(1.0 + cosW0) / a0;
    bq.b2 = ((1.0 + cosW0) / 2.0) / a0;
    bq.a1 = -2.0 * cosW0 / a0;
    bq.a2 = (1.0 - alpha) / a0;
    return bq;
}

float Biquad::process(float input)
{
    const double x0 = input;
    const double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
    return static_cast<float>(y0);
}

}  // namespace drone::speech_core
