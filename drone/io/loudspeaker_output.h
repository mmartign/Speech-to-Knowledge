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
// The report's actual outbound channel is a Zenmuse V1 loudspeaker
// (DJI), "driven in text-to-speech mode with a short set of
// pre-scripted phrases" over DJI's Payload SDK -- the companion
// computer sends text (dispersalCommandPhraseText()), and the V1's own
// firmware synthesizes and broadcasts it. There's no DJI Payload SDK or
// V1 hardware in this environment to integrate against, so that's not
// implemented here.
//
// playDispersalCommand() below is an offline/simulation stand-in: it
// plays a pre-recorded WAV file via a subprocess call to `aplay`
// (ubiquitous on Linux/Jetson via alsa-utils), the same external-tool-
// via-popen pattern this repo already uses for ffprobe in the video
// pipeline. Useful for testing the rest of the pipeline end-to-end
// without DJI hardware, but it is not the production V1 TTS path.
#pragma once

#include "../speech_core/dispersal_commands.h"

#include <string>

namespace drone::io {

// Plays the pre-recorded WAV fallback for `command`, resolving
// dispersalCommandAudioFile(command) against `audioDir`. Blocks until
// playback finishes. Returns false if the file doesn't exist or the
// player subprocess reports a non-zero exit status.
bool playDispersalCommand(
    drone::speech_core::DispersalCommand command,
    const std::string& audioDir);

}  // namespace drone::io
