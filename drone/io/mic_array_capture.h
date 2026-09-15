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
// Capture interface for the four-microphone array -- a ReSpeaker 4-Mic
// Array (Seeed Studio) per the report, 38 g, ~0.5 W.
//
// NOT IMPLEMENTED HERE: a live hardware backend is intentionally left
// as a follow-up. Implementing and testing one without the physical
// array and Jetson board would mean shipping unverified code that just
// claims to work; instead, WavReplayCapture below implements this same
// interface against a WAV file, so the rest of the pipeline (notch
// filtering, spectral subtraction, vocal-band filtering, VAD,
// transcription, distress classification) is real and testable
// end-to-end today, and a live backend can be dropped in later against
// this same interface without touching any caller.
//
// One integration detail worth flagging for whoever builds that
// backend: the ReSpeaker 4-Mic Array is sold as a Raspberry Pi HAT
// (I2S, 40-pin header) with its own seeed-voicecard ALSA driver;
// wiring it to a Jetson Orin NX's I2S/40-pin header is a different
// (if similar-shaped) integration than the Raspberry Pi driver
// assumes, and the report's Methods don't detail that electrical/driver
// path -- worth verifying against Seeed's Jetson-specific guidance (if
// any) or their USB-based sibling products before committing to this
// exact part for a Jetson build.
#pragma once

#include "../speech_core/audio_frame.h"

namespace drone::io {

class MicArrayCapture {
public:
    virtual ~MicArrayCapture() = default;

    // Reads up to `numSamples` samples per channel into `out`, appending
    // to any content already there. Returns false once no more data is
    // available (e.g. end of file for a replay backend); returns true
    // otherwise, even if fewer than `numSamples` samples per channel
    // were actually available and returned.
    virtual bool readFrame(drone::speech_core::MultiChannelFrame& out, size_t numSamples) = 0;

    virtual int numChannels() const = 0;
    virtual int sampleRateHz() const = 0;
};

}  // namespace drone::io
