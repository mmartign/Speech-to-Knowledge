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
// Stage 2 of the report's three-stage rotor-noise rejection: "spectral
// subtraction of broadband downwash". Classic magnitude spectral
// subtraction (Boll, 1979): short-time-Fourier-transform the signal,
// subtract a (scaled, floored) estimate of the noise magnitude spectrum
// from each analysis frame's magnitude while keeping its phase, and
// reconstruct via inverse STFT with overlap-add. Downwash turbulence is
// broadband and relatively stationary, so this removes the energy the
// narrowband notch filter (stage 1) can't touch, while a spectral floor
// keeps the result from collapsing to silence or "musical noise".
#pragma once

#include "audio_frame.h"

#include <vector>

namespace drone::speech_core {

struct SpectralSubtractionConfig {
    int fftSize = 512;    // must be a power of two
    int hopSize = 256;     // 50% overlap with a Hann analysis/synthesis window
    int noiseProfileFrames = 4;  // used only if no noiseProfile is supplied
    float oversubtractionFactor = 2.0f;  // alpha: how aggressively to subtract
    float spectralFloor = 0.05f;          // beta: fraction of original magnitude kept as a floor
};

// Estimates a noise magnitude profile (one value per FFT bin,
// fftSize/2 + 1 of them) from a signal assumed to contain no speech --
// e.g. a brief lead-in recorded at a fixed hover position before a
// casualty is addressed. Prefer this over self-estimation (see
// applySpectralSubtraction()) whenever an actual noise-only reference
// is available.
std::vector<float> estimateNoiseProfile(const MonoFrame& noiseOnlySignal, int fftSize = 512);

// Applies spectral subtraction to `frame` in place. If `noiseProfile`
// is empty, one is self-estimated from `frame`'s own leading
// config.noiseProfileFrames analysis frames -- workable when the start
// of the segment is reliably noise-only, but an explicit profile from
// estimateNoiseProfile() is preferable when available.
void applySpectralSubtraction(
    MonoFrame& frame,
    const SpectralSubtractionConfig& config = {},
    const std::vector<float>& noiseProfile = {});

}  // namespace drone::speech_core
