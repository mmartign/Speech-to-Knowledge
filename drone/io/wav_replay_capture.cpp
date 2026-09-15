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
#include "wav_replay_capture.h"

#include "wav_file.h"

#include <algorithm>

namespace drone::io {

WavReplayCapture::WavReplayCapture(const std::string& path)
{
    loaded_ = readWavFile(path, data_);
}

bool WavReplayCapture::readFrame(drone::speech_core::MultiChannelFrame& out, size_t numSamples)
{
    if (!loaded_ || cursor_ >= data_.numSamples()) {
        return false;
    }

    const size_t available = data_.numSamples() - cursor_;
    const size_t take = std::min(numSamples, available);

    if (out.channels.size() != data_.channels.size()) {
        out.channels.assign(data_.channels.size(), {});
        out.sampleRateHz = data_.sampleRateHz;
    }

    for (size_t c = 0; c < data_.channels.size(); ++c) {
        out.channels[c].insert(
            out.channels[c].end(),
            data_.channels[c].begin() + static_cast<long>(cursor_),
            data_.channels[c].begin() + static_cast<long>(cursor_ + take));
    }

    cursor_ += take;
    return true;
}

int WavReplayCapture::numChannels() const
{
    return data_.numChannels();
}

int WavReplayCapture::sampleRateHz() const
{
    return data_.sampleRateHz;
}

}  // namespace drone::io
