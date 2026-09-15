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
// Pure request/response logic for the "quantized speech recognition"
// stage. Consistent with how the video pipeline in this repo calls out
// to a local OpenAI-compatible vision model server rather than
// embedding one, this does not run any ASR model itself: it talks to a
// local transcription server (e.g. whisper.cpp's server example running
// a quantized GGML Whisper model on the Jetson, or any other server
// implementing the OpenAI /v1/audio/transcriptions shape) over HTTP.
//
// The actual HTTP transport (multipart upload via libcurl) lives in
// io/transcription_http_client.h, kept separate from this pure request-
// field/response-parsing logic so the latter is unit-testable without a
// live server -- the same split used throughout pipeline_core in the
// parent Speech-to-Knowledge project.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace drone::speech_core {

using json = nlohmann::json;

struct AsrConfig {
    std::string baseUrl;
    std::string apiKey;
    std::string modelName;
};

// Extract the transcribed text from an ASR server's JSON response.
//
// Primary shape (OpenAI's /v1/audio/transcriptions, and mirrored by
// whisper.cpp's server example and other compatible servers):
// {"text": "..."}. A couple of defensive fallbacks are included for
// servers that wrap it differently.
std::string extractTranscriptText(const json& response);

}  // namespace drone::speech_core
