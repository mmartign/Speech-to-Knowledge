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
#include "fft.h"

#include <cmath>
#include <utility>

namespace drone::speech_core {

namespace {

bool isPowerOfTwo(size_t n)
{
    return n != 0 && (n & (n - 1)) == 0;
}

}  // namespace

bool fftInPlace(std::vector<std::complex<float>>& data, bool inverse)
{
    const size_t n = data.size();
    if (!isPowerOfTwo(n)) {
        return false;
    }
    if (n <= 1) {
        return true;
    }

    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // Iterative Cooley-Tukey butterflies.
    for (size_t len = 2; len <= n; len <<= 1) {
        const double angleStep = (inverse ? 2.0 : -2.0) * M_PI / static_cast<double>(len);
        const std::complex<float> wLen(
            static_cast<float>(std::cos(angleStep)),
            static_cast<float>(std::sin(angleStep)));

        for (size_t start = 0; start < n; start += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                const auto u = data[start + k];
                const auto v = data[start + k + len / 2] * w;
                data[start + k] = u + v;
                data[start + k + len / 2] = u - v;
                w *= wLen;
            }
        }
    }

    if (inverse) {
        for (auto& x : data) {
            x /= static_cast<float>(n);
        }
    }

    return true;
}

}  // namespace drone::speech_core
