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
// Band-limiting half of the report's stage 3, "voice-activity gating to
// the vocal band": a cascaded high-pass/low-pass filter restricting the
// signal to the frequency range that carries speech intelligibility,
// attenuating broadband rotor downwash energy outside it (most of which
// falls below the band) in addition to whatever spectral_subtraction.h
// already removed. The gating half of stage 3 is vad.h's
// detectSpeechSegments(), applied after this filter in the pipeline.
#pragma once

#include "audio_frame.h"

namespace drone::speech_core {

struct VocalBandFilterConfig {
    // The classic telephony intelligibility band: narrow enough to
    // reject most rotor downwash energy (concentrated at low
    // frequencies) while preserving speech intelligibility.
    double lowCutoffHz = 300.0;
    double highCutoffHz = 3400.0;
    double q = 0.7071;  // maximally-flat (Butterworth-like) response
};

// Applies the band-pass filter to `frame` in place.
void applyVocalBandFilter(MonoFrame& frame, const VocalBandFilterConfig& config = {});

}  // namespace drone::speech_core
