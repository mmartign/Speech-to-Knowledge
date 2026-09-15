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
#include "verbal_response.h"

#include <algorithm>
#include <array>
#include <cctype>

namespace drone::speech_core {

namespace {

std::string toLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

// Distress/pain indicators. English and Italian, since this deploys in
// an Italian rescue context; extend as needed for other languages the
// ASR model supports.
constexpr std::array<const char*, 15> kDistressKeywords = {
    "help", "aiuto", "pain", "hurt", "injured", "bleeding", "broken",
    "stuck", "trapped", "can't breathe", "cannot breathe", "dying",
    "male", "ferito", "non respiro",
};

bool isBlank(const std::string& s)
{
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
}

}  // namespace

VerbalResponseResult assessVerbalResponse(const std::string& transcript)
{
    VerbalResponseResult result;
    result.responded = !isBlank(transcript);
    if (!result.responded) {
        return result;
    }

    const std::string lower = toLowerAscii(transcript);
    for (const char* keyword : kDistressKeywords) {
        if (lower.find(keyword) != std::string::npos) {
            result.matchedKeywords.emplace_back(keyword);
        }
    }
    result.distressLanguage = !result.matchedKeywords.empty();

    return result;
}

}  // namespace drone::speech_core
