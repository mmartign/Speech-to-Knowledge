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
// Stage 1 of the report's three-stage rotor-noise rejection: "adaptive
// notch filtering synchronized to motor telemetry". Rotor blade-pass
// noise is narrowband and tonal (a fundamental at
// blade_count * RPM / 60 plus low harmonics), unlike speech, so a bank
// of narrow IIR notches at those frequencies removes it while leaving
// broadband speech content largely intact. "Adaptive" here means the
// notch frequencies track the motor's current RPM telemetry rather than
// being fixed -- callers re-derive MotorTelemetry from live RPM
// feedback and re-run the filter per block; there's no RPM sensor to
// read from in this environment, so the caller supplies it.
#pragma once

#include "audio_frame.h"

namespace drone::speech_core {

struct MotorTelemetry {
    double rpm = 0.0;
    int bladesPerRotor = 2;  // typical for small quadcopter propellers

    // Fundamental blade-pass frequency in Hz.
    double bladePassHz() const { return rpm / 60.0 * bladesPerRotor; }
};

struct NotchFilterConfig {
    // How many harmonics of the blade-pass frequency to notch
    // (fundamental + harmonics 2..numHarmonics).
    int numHarmonics = 3;

    // Notch quality factor: higher Q = narrower notch (less collateral
    // damage to nearby speech content, but less tolerant of RPM drift
    // between telemetry updates).
    double q = 8.0;
};

// Applies the notch bank to `frame` in place. No-op if
// telemetry.rpm <= 0 or frame.sampleRateHz <= 0. Harmonics at or above
// the Nyquist frequency are skipped.
void applyAdaptiveNotchFilter(
    MonoFrame& frame,
    const MotorTelemetry& telemetry,
    const NotchFilterConfig& config = {});

}  // namespace drone::speech_core
