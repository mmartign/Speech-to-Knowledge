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
// The standard outbound dispersal commands broadcast over the drone's
// loudspeaker (a Zenmuse V1 per the report, driven "in text-to-speech
// mode with a short set of pre-scripted phrases"), and the rule-based
// mapping from what the speech-to-knowledge pipeline observed to the
// command a first responder would want issued.
#pragma once

#include "non_verbal_distress.h"
#include "verbal_response.h"

#include <optional>
#include <string>

namespace drone::speech_core {

enum class DispersalCommand {
    StayCalmHelpComing,  // acknowledged presence, reassurance
    DoNotMove,            // suspected severe injury: immobilize
    MoveToOpenArea,       // mobile casualty: guide to a visible/reachable spot
    EvacuateAreaNow,      // scene hazard (not injury-specific; see note below)
};

// The pre-scripted phrase text for `command`, sent to the loudspeaker's
// text-to-speech engine (see io/loudspeaker_output.h). Fixed, reviewed
// wording rather than generated text: what gets broadcast to an injured
// person matters too much to leave to open-ended generation.
std::string dispersalCommandPhraseText(DispersalCommand command);

// Filename (without directory) of a pre-recorded WAV fallback for
// `command`, used only by the offline/simulation loudspeaker backend
// when no TTS-capable device is available (see io/loudspeaker_output.h
// for why that backend exists).
std::string dispersalCommandAudioFile(DispersalCommand command);

// Rule-based selection of which command to broadcast, given both the
// (placeholder) non-verbal distress classification and what the ASR
// transcript showed. Returns std::nullopt when nothing rises to the
// level of warranting an announcement yet.
std::optional<DispersalCommand> selectDispersalCommand(
    const NonVerbalDistressResult& nonVerbal,
    const VerbalResponseResult& verbal);

}  // namespace drone::speech_core
