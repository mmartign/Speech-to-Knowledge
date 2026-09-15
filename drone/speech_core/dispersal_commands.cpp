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
#include "dispersal_commands.h"

namespace drone::speech_core {

std::string dispersalCommandPhraseText(DispersalCommand command)
{
    switch (command) {
        case DispersalCommand::StayCalmHelpComing:
            return "Stay calm. Help is on the way.";
        case DispersalCommand::DoNotMove:
            return "Do not move. Rescue is coming to you.";
        case DispersalCommand::MoveToOpenArea:
            return "If you can move safely, go to open ground where you can be seen.";
        case DispersalCommand::EvacuateAreaNow:
            return "Warning. Leave this area now if you are able.";
    }
    return dispersalCommandPhraseText(DispersalCommand::StayCalmHelpComing);
}

std::string dispersalCommandAudioFile(DispersalCommand command)
{
    switch (command) {
        case DispersalCommand::StayCalmHelpComing:
            return "stay_calm_help_coming.wav";
        case DispersalCommand::DoNotMove:
            return "do_not_move.wav";
        case DispersalCommand::MoveToOpenArea:
            return "move_to_open_area.wav";
        case DispersalCommand::EvacuateAreaNow:
            return "evacuate_area_now.wav";
    }
    return "stay_calm_help_coming.wav";
}

std::optional<DispersalCommand> selectDispersalCommand(
    const NonVerbalDistressResult& nonVerbal,
    const VerbalResponseResult& verbal)
{
    // Note: MoveToOpenArea and EvacuateAreaNow are legitimate standard
    // phrases (see dispersalCommandPhraseText()) but aren't reachable
    // from these two signals alone -- selecting them needs casualty
    // mobility and scene-hazard signals this pipeline doesn't have
    // (video-to-knowledge and the cross-pipeline aggregator's job,
    // both out of scope here).
    if (nonVerbal.level == NonVerbalDistressLevel::High) {
        // Severe non-verbal distress (e.g. screaming in pain): safest
        // default is to keep the casualty still until rescue arrives.
        return DispersalCommand::DoNotMove;
    }
    if (verbal.distressLanguage || nonVerbal.level == NonVerbalDistressLevel::Elevated) {
        return DispersalCommand::StayCalmHelpComing;
    }

    // A calm, coherent response -- or no response at all -- is
    // genuinely ambiguous from this signal set alone without the full
    // aggregator's cross-pipeline reasoning and timing, so stay silent
    // rather than guess.
    return std::nullopt;
}

}  // namespace drone::speech_core
