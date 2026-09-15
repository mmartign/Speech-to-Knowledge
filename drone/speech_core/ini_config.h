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
// INI parsing for the drone speech-to-knowledge config. Deliberately a
// self-contained copy of the same small parser used by the parent
// Speech-to-Knowledge project's pipeline_core, rather than a cross
// include: this subsystem targets different hardware (Jetson/aarch64,
// no OpenCV/CURL-for-video) and is meant to be independently buildable
// and deployable, so it doesn't reach into ../pipeline_core.
#pragma once

#include <map>
#include <string>

#include "transcription_client.h"

namespace drone::speech_core {

// Trim leading/trailing whitespace in place. Returns false if the
// resulting string is empty.
bool trimInPlace(std::string& s);

// Same [section]/key=value/comment semantics as the parent project's
// pipeline_core::parseIni(): flattens sections into "section.key".
std::map<std::string, std::string> parseIni(const std::string& filename);

// Load and validate the required [asr] config values.
bool loadAsrConfig(const std::string& path, AsrConfig& cfg);

}  // namespace drone::speech_core
