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
// A MicArrayCapture backend that replays a pre-recorded multi-channel
// WAV file, for offline testing and simulation without the physical
// microphone array.
#pragma once

#include "mic_array_capture.h"

#include <string>

namespace drone::io {

class WavReplayCapture : public MicArrayCapture {
public:
    // Loads the entire file into memory up front. `ok()` reports
    // whether that succeeded; a failed load behaves as an
    // immediately-exhausted capture (readFrame() always returns false).
    explicit WavReplayCapture(const std::string& path);

    bool ok() const { return loaded_; }

    bool readFrame(drone::speech_core::MultiChannelFrame& out, size_t numSamples) override;
    int numChannels() const override;
    int sampleRateHz() const override;

private:
    drone::speech_core::MultiChannelFrame data_;
    size_t cursor_ = 0;
    bool loaded_ = false;
};

}  // namespace drone::io
