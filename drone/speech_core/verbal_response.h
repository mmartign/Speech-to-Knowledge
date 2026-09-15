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
// What the ASR transcript ("short clinical replies", per the report)
// tells us, as two separate signals: whether the casualty said
// anything intelligible at all, and whether recognizable distress/pain
// language was present. Deliberately not called a "distress
// classifier" -- the report reserves that term for the convolutional
// classifier over non-verbal audio (see non_verbal_distress.h); this is
// a plain rule-based reading of recognized words, mainly useful for the
// "responds to voice" / "obeys the dispersal command" signals the
// (out-of-scope here) cross-pipeline aggregator's decision tree needs.
#pragma once

#include <string>
#include <vector>

namespace drone::speech_core {

struct VerbalResponseResult {
    // True if the transcript contains any recognizable words at all --
    // a proxy for the casualty being conscious and responsive to the
    // aircraft's voice prompt, independent of what was said.
    bool responded = false;

    // True if recognized distress/pain keywords were present.
    bool distressLanguage = false;

    std::vector<std::string> matchedKeywords;
};

VerbalResponseResult assessVerbalResponse(const std::string& transcript);

}  // namespace drone::speech_core
