// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for io/wav_file.h.
#include "../io/wav_file.h"

#include "test_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

using namespace drone::speech_core;
using namespace drone::io;
using drone::speech_core_test::TempPathGuard;
using drone::speech_core_test::uniqueTempPath;

TEST_CASE("writeMonoWavFile / readWavFile round-trips within PCM16 quantization", "[wav]")
{
    MonoFrame original;
    original.sampleRateHz = 16000;
    for (int i = 0; i < 800; ++i) {
        original.samples.push_back(
            0.5f * static_cast<float>(std::sin(static_cast<double>(i) * 0.1)));
    }

    TempPathGuard path(uniqueTempPath(".wav"));
    REQUIRE(writeMonoWavFile(path.string(), original));

    MultiChannelFrame readBack;
    REQUIRE(readWavFile(path.string(), readBack));

    REQUIRE(readBack.sampleRateHz == 16000);
    REQUIRE(readBack.numChannels() == 1);
    REQUIRE(readBack.numSamples() == original.samples.size());

    for (size_t i = 0; i < original.samples.size(); ++i) {
        // PCM16 quantization step is 1/32768; allow a small margin.
        REQUIRE(std::fabs(readBack.channels[0][i] - original.samples[i]) < 0.001f);
    }
}

TEST_CASE("readWavFile rejects a nonexistent file", "[wav]")
{
    MultiChannelFrame frame;
    REQUIRE_FALSE(readWavFile("/nonexistent/path.wav", frame));
}

TEST_CASE("readWavFile rejects a file that isn't RIFF/WAVE", "[wav]")
{
    TempPathGuard path(uniqueTempPath(".wav"));
    {
        std::ofstream out(path.string());
        out << "not a wav file";
    }

    MultiChannelFrame frame;
    REQUIRE_FALSE(readWavFile(path.string(), frame));
}

TEST_CASE("readWavFile accepts WAVE_FORMAT_EXTENSIBLE PCM16", "[wav]")
{
    // Hand-built minimal WAVE_FORMAT_EXTENSIBLE file: the shape ffmpeg
    // (and most tools) actually write for >2-channel PCM, which a
    // reader that only accepts the plain WAVE_FORMAT_PCM (1) tag will
    // wrongly reject. Two mono int16 samples: 1000 and -1000.
    std::vector<unsigned char> bytes;
    auto pushU32 = [&](std::uint32_t v) {
        bytes.push_back(static_cast<unsigned char>(v & 0xFF));
        bytes.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
        bytes.push_back(static_cast<unsigned char>((v >> 16) & 0xFF));
        bytes.push_back(static_cast<unsigned char>((v >> 24) & 0xFF));
    };
    auto pushU16 = [&](std::uint16_t v) {
        bytes.push_back(static_cast<unsigned char>(v & 0xFF));
        bytes.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
    };
    auto pushTag = [&](const char* tag) {
        bytes.insert(bytes.end(), tag, tag + 4);
    };

    const std::uint32_t dataSize = 4;  // 2 samples * 2 bytes
    pushTag("RIFF");
    pushU32(4 + (8 + 40) + (8 + dataSize));
    pushTag("WAVE");

    pushTag("fmt ");
    pushU32(40);              // WAVE_FORMAT_EXTENSIBLE fmt chunk size
    pushU16(0xFFFE);          // format tag: EXTENSIBLE
    pushU16(1);                // 1 channel
    pushU32(8000);             // sample rate
    pushU32(16000);            // byte rate
    pushU16(2);                 // block align
    pushU16(16);                 // bits per sample
    pushU16(22);                 // cbSize
    pushU16(16);                 // valid bits per sample
    pushU32(0);                  // channel mask
    pushU32(1);                  // SubFormat GUID first 4 bytes: PCM (1)
    // Remaining 12 bytes of the standard SubFormat GUID suffix.
    const unsigned char guidSuffix[12] = {
        0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
    bytes.insert(bytes.end(), guidSuffix, guidSuffix + 12);

    pushTag("data");
    pushU32(dataSize);
    pushU16(1000);
    pushU16(static_cast<std::uint16_t>(-1000));

    TempPathGuard path(uniqueTempPath(".wav"));
    {
        std::ofstream out(path.string(), std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }

    MultiChannelFrame frame;
    REQUIRE(readWavFile(path.string(), frame));
    REQUIRE(frame.numChannels() == 1);
    REQUIRE(frame.sampleRateHz == 8000);
    REQUIRE(frame.numSamples() == 2);
    REQUIRE(frame.channels[0][0] > 0.0f);
    REQUIRE(frame.channels[0][1] < 0.0f);
}

TEST_CASE("writeMonoWavFile clamps out-of-range samples instead of wrapping", "[wav]")
{
    MonoFrame frame;
    frame.sampleRateHz = 8000;
    frame.samples = {2.0f, -2.0f, 0.0f};

    TempPathGuard path(uniqueTempPath(".wav"));
    REQUIRE(writeMonoWavFile(path.string(), frame));

    MultiChannelFrame readBack;
    REQUIRE(readWavFile(path.string(), readBack));
    REQUIRE(readBack.channels[0][0] > 0.99f);   // clamped near +1, not wrapped negative
    REQUIRE(readBack.channels[0][1] < -0.99f);  // clamped near -1, not wrapped positive
}
