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
// Linux/Jetson-only (unlike the parent video pipeline, this subsystem
// targets a single deployment platform, so there's no Windows branch
// here).
#include "loudspeaker_output.h"

#include <cstdio>
#include <fstream>

namespace drone::io {

bool playDispersalCommand(
    drone::speech_core::DispersalCommand command,
    const std::string& audioDir)
{
    const std::string fileName = drone::speech_core::dispersalCommandAudioFile(command);

    std::string path = audioDir;
    if (!path.empty() && path.back() != '/') {
        path += '/';
    }
    path += fileName;

    {
        std::ifstream check(path);
        if (!check.is_open()) {
            return false;
        }
    }

    const std::string cmd = "aplay \"" + path + "\" >/dev/null 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        return false;
    }

    const int rc = pclose(pipe);
    return rc == 0;
}

}  // namespace drone::io
