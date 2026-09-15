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
#include "wav_file.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace drone::io {

namespace {

// All WAV integers are little-endian; x86/ARM run little-endian by
// default, which covers the development/CI hosts and the Jetson target
// alike, so we read/write directly without byte-swapping.

std::uint32_t readU32(const unsigned char* p)
{
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint16_t readU16(const unsigned char* p)
{
    return static_cast<std::uint16_t>(p[0]) |
           static_cast<std::uint16_t>(p[1] << 8);
}

void writeU32(std::ofstream& out, std::uint32_t v)
{
    unsigned char bytes[4] = {
        static_cast<unsigned char>(v & 0xFF),
        static_cast<unsigned char>((v >> 8) & 0xFF),
        static_cast<unsigned char>((v >> 16) & 0xFF),
        static_cast<unsigned char>((v >> 24) & 0xFF),
    };
    out.write(reinterpret_cast<const char*>(bytes), 4);
}

void writeU16(std::ofstream& out, std::uint16_t v)
{
    unsigned char bytes[2] = {
        static_cast<unsigned char>(v & 0xFF),
        static_cast<unsigned char>((v >> 8) & 0xFF),
    };
    out.write(reinterpret_cast<const char*>(bytes), 2);
}

std::int16_t quantizeSample(float s)
{
    const float clamped = std::max(-1.0f, std::min(1.0f, s));
    return static_cast<std::int16_t>(clamped * 32767.0f);
}

void writeWavHeader(
    std::ofstream& file,
    std::uint16_t numChannels,
    std::uint32_t sampleRate,
    std::uint32_t numFrames)
{
    constexpr std::uint16_t bitsPerSample = 16;
    const std::uint32_t byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const std::uint16_t blockAlign = static_cast<std::uint16_t>(numChannels * (bitsPerSample / 8));
    const std::uint32_t dataSize = numFrames * blockAlign;

    file.write("RIFF", 4);
    writeU32(file, 36 + dataSize);
    file.write("WAVE", 4);

    file.write("fmt ", 4);
    writeU32(file, 16);
    writeU16(file, 1);  // PCM
    writeU16(file, numChannels);
    writeU32(file, sampleRate);
    writeU32(file, byteRate);
    writeU16(file, blockAlign);
    writeU16(file, bitsPerSample);

    file.write("data", 4);
    writeU32(file, dataSize);
}

}  // namespace

bool readWavFile(const std::string& path, drone::speech_core::MultiChannelFrame& out)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    std::vector<unsigned char> header(12);
    file.read(reinterpret_cast<char*>(header.data()), 12);
    if (!file || std::memcmp(header.data(), "RIFF", 4) != 0 ||
        std::memcmp(header.data() + 8, "WAVE", 4) != 0) {
        return false;
    }

    int numChannels = 0;
    std::uint32_t sampleRate = 0;
    std::uint16_t bitsPerSample = 0;
    bool isPcm = false;
    bool haveFmt = false;
    std::vector<unsigned char> pcmBytes;
    bool haveData = false;

    while (file) {
        unsigned char chunkHeader[8];
        file.read(reinterpret_cast<char*>(chunkHeader), 8);
        if (!file) {
            break;
        }
        const std::uint32_t chunkSize = readU32(chunkHeader + 4);

        if (std::memcmp(chunkHeader, "fmt ", 4) == 0) {
            std::vector<unsigned char> fmt(chunkSize);
            file.read(reinterpret_cast<char*>(fmt.data()), chunkSize);
            if (!file || chunkSize < 16) {
                return false;
            }
            const std::uint16_t formatTag = readU16(fmt.data() + 0);
            numChannels = readU16(fmt.data() + 2);
            sampleRate = readU32(fmt.data() + 4);
            bitsPerSample = readU16(fmt.data() + 14);

            constexpr std::uint16_t kFormatPcm = 1;
            constexpr std::uint16_t kFormatExtensible = 0xFFFE;
            isPcm = (formatTag == kFormatPcm);
            if (formatTag == kFormatExtensible && chunkSize >= 40) {
                // WAVE_FORMAT_EXTENSIBLE, ubiquitous for >2-channel PCM
                // (e.g. ffmpeg defaults to it for 4-channel captures):
                // the real format lives in the SubFormat GUID's first
                // 4 bytes, right after the 2-byte cbSize (=22), 2-byte
                // valid-bits-per-sample, and 4-byte channel mask fields.
                const std::uint32_t subFormat = readU32(fmt.data() + 24);
                isPcm = (subFormat == kFormatPcm);
            }
            haveFmt = true;
        } else if (std::memcmp(chunkHeader, "data", 4) == 0) {
            pcmBytes.resize(chunkSize);
            file.read(reinterpret_cast<char*>(pcmBytes.data()), chunkSize);
            if (!file) {
                return false;
            }
            haveData = true;
        } else {
            file.seekg(static_cast<std::streamoff>(chunkSize), std::ios::cur);
        }

        // Chunks are word-aligned: an odd-sized chunk has one pad byte.
        if (chunkSize % 2 != 0) {
            file.seekg(1, std::ios::cur);
        }
    }

    if (!haveFmt || !haveData || !isPcm ||
        bitsPerSample != 16 || numChannels <= 0) {
        return false;
    }

    const size_t bytesPerSample = 2;
    const size_t frameBytes = bytesPerSample * static_cast<size_t>(numChannels);
    const size_t numFrames = pcmBytes.size() / frameBytes;

    out.sampleRateHz = static_cast<int>(sampleRate);
    out.channels.assign(static_cast<size_t>(numChannels), std::vector<float>(numFrames));

    for (size_t f = 0; f < numFrames; ++f) {
        for (int c = 0; c < numChannels; ++c) {
            const size_t byteOffset = f * frameBytes + static_cast<size_t>(c) * bytesPerSample;
            const std::int16_t raw = static_cast<std::int16_t>(
                readU16(&pcmBytes[byteOffset]));
            out.channels[static_cast<size_t>(c)][f] = static_cast<float>(raw) / 32768.0f;
        }
    }

    return true;
}

bool writeMonoWavFile(const std::string& path, const drone::speech_core::MonoFrame& frame)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    const auto numFrames = static_cast<std::uint32_t>(frame.samples.size());
    const auto sampleRate = static_cast<std::uint32_t>(frame.sampleRateHz);
    writeWavHeader(file, /*numChannels=*/1, sampleRate, numFrames);

    for (float s : frame.samples) {
        writeU16(file, static_cast<std::uint16_t>(quantizeSample(s)));
    }

    return static_cast<bool>(file);
}

bool writeMultiChannelWavFile(
    const std::string& path,
    const drone::speech_core::MultiChannelFrame& frame)
{
    if (frame.numChannels() == 0) {
        return false;
    }
    const size_t numFrames = frame.numSamples();
    for (const auto& channel : frame.channels) {
        if (channel.size() != numFrames) {
            return false;  // channels of unequal length: malformed input
        }
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    const auto numChannels = static_cast<std::uint16_t>(frame.numChannels());
    const auto sampleRate = static_cast<std::uint32_t>(frame.sampleRateHz);
    writeWavHeader(file, numChannels, sampleRate, static_cast<std::uint32_t>(numFrames));

    for (size_t f = 0; f < numFrames; ++f) {
        for (int c = 0; c < frame.numChannels(); ++c) {
            writeU16(file, static_cast<std::uint16_t>(
                quantizeSample(frame.channels[static_cast<size_t>(c)][f])));
        }
    }

    return static_cast<bool>(file);
}

}  // namespace drone::io
