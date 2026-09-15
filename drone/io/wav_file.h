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
// Minimal RIFF/WAVE PCM16 reader and mono writer. Only 16-bit PCM is
// supported (documented limitation, sufficient for both the four-mic
// array's expected capture format and the mono segments sent to the
// transcription server); other formats are rejected rather than
// misread.
#pragma once

#include "../speech_core/audio_frame.h"

#include <string>

namespace drone::io {

// Reads a PCM16 WAV file (mono or multi-channel) into `out`. Returns
// false (leaving `out` unspecified) if the file can't be opened, isn't
// a valid RIFF/WAVE file, or isn't 16-bit PCM.
bool readWavFile(const std::string& path, drone::speech_core::MultiChannelFrame& out);

// Writes `frame` as a mono PCM16 WAV file. Samples are clamped to
// [-1, 1] before quantization. Returns false if the file can't be
// created.
bool writeMonoWavFile(const std::string& path, const drone::speech_core::MonoFrame& frame);

// Writes `frame` as a multi-channel PCM16 WAV file (samples clamped to
// [-1, 1] before quantization). Useful for logging/replaying a raw
// mic-array capture, e.g. offline test fixtures for WavReplayCapture.
// Returns false if the file can't be created or the channels have
// unequal lengths.
bool writeMultiChannelWavFile(
    const std::string& path,
    const drone::speech_core::MultiChannelFrame& frame);

}  // namespace drone::io
