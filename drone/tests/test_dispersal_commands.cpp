// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for speech_core/dispersal_commands.h.
#include "../speech_core/dispersal_commands.h"

#include <catch2/catch_test_macros.hpp>

using namespace drone::speech_core;

namespace {

NonVerbalDistressResult nonVerbal(NonVerbalDistressLevel level)
{
    NonVerbalDistressResult r;
    r.level = level;
    return r;
}

VerbalResponseResult verbal(bool responded, bool distressLanguage)
{
    VerbalResponseResult r;
    r.responded = responded;
    r.distressLanguage = distressLanguage;
    return r;
}

}  // namespace

TEST_CASE("dispersalCommandPhraseText and dispersalCommandAudioFile cover every command", "[dispersal]")
{
    for (auto cmd : {DispersalCommand::StayCalmHelpComing, DispersalCommand::DoNotMove,
                      DispersalCommand::MoveToOpenArea, DispersalCommand::EvacuateAreaNow}) {
        REQUIRE_FALSE(dispersalCommandPhraseText(cmd).empty());
        REQUIRE_FALSE(dispersalCommandAudioFile(cmd).empty());
    }
}

TEST_CASE("selectDispersalCommand returns nullopt for a calm coherent response", "[dispersal]")
{
    const auto result = selectDispersalCommand(
        nonVerbal(NonVerbalDistressLevel::None), verbal(/*responded=*/true, /*distressLanguage=*/false));
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("selectDispersalCommand returns nullopt for no response at all", "[dispersal]")
{
    const auto result = selectDispersalCommand(
        nonVerbal(NonVerbalDistressLevel::None), verbal(/*responded=*/false, /*distressLanguage=*/false));
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("selectDispersalCommand reassures on distress language", "[dispersal]")
{
    const auto result = selectDispersalCommand(
        nonVerbal(NonVerbalDistressLevel::None), verbal(/*responded=*/true, /*distressLanguage=*/true));
    REQUIRE(result == DispersalCommand::StayCalmHelpComing);
}

TEST_CASE("selectDispersalCommand reassures on elevated non-verbal distress", "[dispersal]")
{
    const auto result = selectDispersalCommand(
        nonVerbal(NonVerbalDistressLevel::Elevated), verbal(/*responded=*/false, /*distressLanguage=*/false));
    REQUIRE(result == DispersalCommand::StayCalmHelpComing);
}

TEST_CASE("selectDispersalCommand tells severe non-verbal distress to stay still", "[dispersal]")
{
    const auto result = selectDispersalCommand(
        nonVerbal(NonVerbalDistressLevel::High), verbal(/*responded=*/false, /*distressLanguage=*/false));
    REQUIRE(result == DispersalCommand::DoNotMove);
}

TEST_CASE("selectDispersalCommand prioritizes high non-verbal distress over calm language", "[dispersal]")
{
    // e.g. screaming in pain without recognizable words.
    const auto result = selectDispersalCommand(
        nonVerbal(NonVerbalDistressLevel::High), verbal(/*responded=*/true, /*distressLanguage=*/false));
    REQUIRE(result == DispersalCommand::DoNotMove);
}
