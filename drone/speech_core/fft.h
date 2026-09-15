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
// A small, self-contained radix-2 FFT: just enough for the STFT
// framework behind spectral subtraction (see spectral_subtraction.h).
// Not a general-purpose signal processing library, and deliberately not
// an added third-party dependency for this one need.
#pragma once

#include <complex>
#include <vector>

namespace drone::speech_core {

// In-place iterative radix-2 Cooley-Tukey FFT/IFFT.
//
// `data.size()` must be a power of two; returns false (leaving `data`
// unmodified) otherwise. `inverse` selects IFFT (which applies the 1/N
// normalization) instead of the forward transform.
bool fftInPlace(std::vector<std::complex<float>>& data, bool inverse);

}  // namespace drone::speech_core
