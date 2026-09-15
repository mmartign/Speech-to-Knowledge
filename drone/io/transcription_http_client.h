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
// libcurl multipart transport for the transcription request. Kept
// separate from speech_core/transcription_client.h's pure request/
// response logic so that logic stays unit-testable without a live ASR
// server; this file is the untested, network-dependent boundary (same
// role as sendFrameToOpenAI() in the parent Speech-to-Knowledge project).
#pragma once

#include "../speech_core/transcription_client.h"

#include <string>

namespace drone::io {

// Uploads the PCM16 mono WAV file at `wavPath` to
// `cfg.baseUrl + "audio/transcriptions"` (the OpenAI-compatible
// transcription endpoint shape) and returns the transcribed text.
//
// Throws std::runtime_error on a transport-level failure (server
// unreachable, non-JSON response, etc.), mirroring how the parent
// project's openai-cpp client reports failures.
std::string transcribeWavFile(
    const drone::speech_core::AsrConfig& cfg,
    const std::string& wavPath);

}  // namespace drone::io
