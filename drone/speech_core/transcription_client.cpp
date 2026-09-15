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
#include "transcription_client.h"

namespace drone::speech_core {

std::string extractTranscriptText(const json& response)
{
    const auto textIt = response.find("text");
    if (textIt != response.end() && textIt->is_string()) {
        return textIt->get<std::string>();
    }

    // Defensive fallbacks seen on some OpenAI-compatible ASR servers.
    const auto transcriptIt = response.find("transcript");
    if (transcriptIt != response.end() && transcriptIt->is_string()) {
        return transcriptIt->get<std::string>();
    }

    const auto resultsIt = response.find("results");
    if (resultsIt != response.end() && resultsIt->is_array() && !resultsIt->empty()) {
        const auto& first = (*resultsIt)[0];
        const auto innerText = first.find("text");
        if (innerText != first.end() && innerText->is_string()) {
            return innerText->get<std::string>();
        }
    }

    return {};
}

}  // namespace drone::speech_core
