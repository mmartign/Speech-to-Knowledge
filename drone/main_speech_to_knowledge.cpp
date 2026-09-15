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
// Entry point for the drone speech-to-knowledge pipeline: four-mic WAV
// in -> mono mixdown -> the report's three-stage rotor-noise rejection
// (adaptive notch filtering, spectral subtraction, vocal-band filtering
// + VAD gating) -> transcription -> verbal-response and non-verbal-
// distress classification -> structured result out (and optionally, a
// loudspeaker dispersal command).
//
// This operates in offline/replay mode over a WAV file rather than a
// live real-time capture loop: there is no live microphone-array
// backend yet (see io/mic_array_capture.h), so a real-time loop would
// have nothing genuine to drive it. This is otherwise the same pipeline
// a live backend would feed, and is a straightforward drop-in point for
// one once available.
#include "speech_core/audio_frame.h"
#include "speech_core/notch_filter.h"
#include "speech_core/spectral_subtraction.h"
#include "speech_core/vocal_band_filter.h"
#include "speech_core/vad.h"
#include "speech_core/non_verbal_distress.h"
#include "speech_core/verbal_response.h"
#include "speech_core/dispersal_commands.h"
#include "speech_core/ini_config.h"

#include "io/wav_file.h"
#include "io/wav_replay_capture.h"
#include "io/transcription_http_client.h"
#include "io/loudspeaker_output.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>

using json = nlohmann::json;
using namespace drone::speech_core;

namespace {

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " <4ch_input.wav> [config.ini] [options]\n"
        << "Options:\n"
        << "  --audio-dir <dir>    Directory holding dispersal command WAV files (default: ./audio)\n"
        << "  --dispatch           Actually play the selected dispersal command over the loudspeaker\n"
        << "  --rotor-rpm <rpm>    Motor RPM for adaptive notch filtering (default: 0 = disabled;\n"
        << "                       there is no live motor telemetry feed in this offline mode)\n"
        << "  --rotor-blades <n>   Blades per rotor for notch filtering (default: 2)\n";
}

// Simple channel-average mixdown. The report doesn't describe spatial
// combination of the four channels beyond the three named noise-
// rejection stages, so this deliberately doesn't attempt delay-and-sum
// beamforming or similar -- just an unweighted mean across channels.
MonoFrame mixDownToMono(const MultiChannelFrame& frame)
{
    MonoFrame out;
    out.sampleRateHz = frame.sampleRateHz;
    const size_t n = frame.numSamples();
    if (frame.numChannels() == 0 || n == 0) {
        return out;
    }

    out.samples.assign(n, 0.0f);
    for (const auto& channel : frame.channels) {
        for (size_t i = 0; i < n; ++i) {
            out.samples[i] += channel[i];
        }
    }
    const float scale = 1.0f / static_cast<float>(frame.numChannels());
    for (float& s : out.samples) {
        s *= scale;
    }
    return out;
}

float peakEnergyRatio(const std::vector<float>& samples)
{
    if (samples.empty()) {
        return 1.0f;
    }
    double sumSq = 0.0;
    float peak = 0.0f;
    for (float s : samples) {
        sumSq += static_cast<double>(s) * s;
        peak = std::max(peak, std::fabs(s));
    }
    const float rms = static_cast<float>(std::sqrt(sumSq / static_cast<double>(samples.size())));
    return rms > 1e-6f ? peak / rms : 1.0f;
}

std::string nonVerbalLevelName(NonVerbalDistressLevel level)
{
    switch (level) {
        case NonVerbalDistressLevel::None: return "none";
        case NonVerbalDistressLevel::Elevated: return "elevated";
        case NonVerbalDistressLevel::High: return "high";
    }
    return "unknown";
}

std::string dispersalCommandName(DispersalCommand cmd)
{
    switch (cmd) {
        case DispersalCommand::StayCalmHelpComing: return "stay_calm_help_coming";
        case DispersalCommand::DoNotMove: return "do_not_move";
        case DispersalCommand::MoveToOpenArea: return "move_to_open_area";
        case DispersalCommand::EvacuateAreaNow: return "evacuate_area_now";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string wavPath = argv[1];
    std::string configPath = "config.ini";
    std::string audioDir = "./audio";
    bool dispatch = false;
    MotorTelemetry telemetry;  // rpm defaults to 0 -> notch filter is a no-op

    int argi = 2;
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
        configPath = argv[2];
        argi = 3;
    }
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (a == "--audio-dir" && argi + 1 < argc) {
            audioDir = argv[++argi];
        } else if (a == "--dispatch") {
            dispatch = true;
        } else if (a == "--rotor-rpm" && argi + 1 < argc) {
            telemetry.rpm = std::stod(argv[++argi]);
        } else if (a == "--rotor-blades" && argi + 1 < argc) {
            telemetry.bladesPerRotor = std::stoi(argv[++argi]);
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "[ERROR] Unknown option: " << a << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    AsrConfig cfg;
    if (!loadAsrConfig(configPath, cfg)) {
        return 1;
    }
    std::cerr << "[INFO] ASR base URL: " << cfg.baseUrl << "\n";
    std::cerr << "[INFO] ASR model: " << cfg.modelName << "\n";

    drone::io::WavReplayCapture capture(wavPath);
    if (!capture.ok()) {
        std::cerr << "[ERROR] Could not read WAV file: " << wavPath << "\n";
        return 1;
    }

    drone::speech_core::MultiChannelFrame raw;
    // Pull everything available in one call; WavReplayCapture is backed
    // by the whole file already loaded in memory, so an oversized
    // request just means "read to end of file" for offline/replay use.
    constexpr size_t kReadEverything = 1ull << 40;
    capture.readFrame(raw, kReadEverything);
    raw.sampleRateHz = capture.sampleRateHz();

    std::cerr << "[INFO] Loaded " << raw.numChannels() << "-channel audio, "
              << raw.numSamples() << " samples at " << raw.sampleRateHz << " Hz\n";

    MonoFrame audio = mixDownToMono(raw);

    // The report's three-stage rotor-noise rejection.
    if (telemetry.rpm > 0.0) {
        applyAdaptiveNotchFilter(audio, telemetry);
    } else {
        std::cerr << "[INFO] --rotor-rpm not given; skipping adaptive notch filtering\n";
    }
    applySpectralSubtraction(audio);
    applyVocalBandFilter(audio);
    const auto segments = detectSpeechSegments(audio);  // the "gating" half of stage 3

    std::cerr << "[INFO] Detected " << segments.size() << " speech segment(s)\n";

    const auto tempDir = std::filesystem::temp_directory_path();
    int segmentIdx = 0;

    for (const auto& segment : segments) {
        MonoFrame segmentFrame;
        segmentFrame.sampleRateHz = audio.sampleRateHz;
        segmentFrame.samples.assign(
            audio.samples.begin() + static_cast<long>(segment.startSample),
            audio.samples.begin() + static_cast<long>(segment.endSample));

        const float ratio = peakEnergyRatio(segmentFrame.samples);
        const NonVerbalDistressResult nonVerbal = classifyNonVerbalDistress(segmentFrame);

        const auto segmentWavPath =
            tempDir / ("drone_speech_segment_" + std::to_string(segmentIdx++) + ".wav");
        if (!drone::io::writeMonoWavFile(segmentWavPath.string(), segmentFrame)) {
            std::cerr << "[ERROR] Failed to write segment WAV for transcription\n";
            continue;
        }

        std::string transcript;
        try {
            transcript = drone::io::transcribeWavFile(cfg, segmentWavPath.string());
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Transcription failed for segment starting at sample "
                      << segment.startSample << ": " << e.what() << "\n";
        }
        std::filesystem::remove(segmentWavPath);

        const VerbalResponseResult verbal = assessVerbalResponse(transcript);
        const auto command = selectDispersalCommand(nonVerbal, verbal);

        json result = {
            {"segment_start_sample", segment.startSample},
            {"segment_end_sample", segment.endSample},
            {"sample_rate_hz", audio.sampleRateHz},
            {"transcript", transcript},
            {"peak_energy_ratio", ratio},
            {"responded", verbal.responded},
            {"verbal_distress_language", verbal.distressLanguage},
            {"matched_keywords", verbal.matchedKeywords},
            {"non_verbal_distress_level", nonVerbalLevelName(nonVerbal.level)},
            {"non_verbal_distress_confidence", nonVerbal.confidence},
        };
        if (command.has_value()) {
            result["dispersal_command"] = dispersalCommandName(*command);
            result["dispersal_phrase"] = dispersalCommandPhraseText(*command);
        }

        std::cout << result.dump() << std::endl;

        if (dispatch && command.has_value()) {
            if (!drone::io::playDispersalCommand(*command, audioDir)) {
                std::cerr << "[WARN] Could not play dispersal command audio from "
                          << audioDir << "\n";
            }
        }
    }

    return 0;
}
