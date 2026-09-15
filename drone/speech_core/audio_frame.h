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
// Shared audio buffer types for the drone speech-to-knowledge pipeline.
// Samples are floating point in [-1, 1]; conversion to/from 16-bit PCM
// happens only at the WAV file I/O boundary (see io/wav_file.h).
#pragma once

#include <cstddef>
#include <vector>

namespace drone::speech_core {

// One buffer per microphone channel, all of equal length.
struct MultiChannelFrame {
    int sampleRateHz = 0;
    std::vector<std::vector<float>> channels;

    int numChannels() const { return static_cast<int>(channels.size()); }

    size_t numSamples() const
    {
        return channels.empty() ? 0 : channels.front().size();
    }
};

// A single-channel buffer, e.g. after beamforming down to one enhanced
// signal, or a segment extracted for transcription.
struct MonoFrame {
    int sampleRateHz = 0;
    std::vector<float> samples;
};

}  // namespace drone::speech_core
