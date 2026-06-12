// -*- coding: utf-8 -*-
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// Copyright (C) 2025-2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
// This file is based on https://github.com/davabase/whisper_real_time, with modifications
// for real-time performance, concurrency improvements, debug tracing, and improved
// adaptive energy VAD (hangover + conservative updates to prevent speech clipping).
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//
// Single-file version of the Spazio IT Speech-to-Knowledge application
// with support for:
//   1. local microphone input (PortAudio)
//   2. audio file input
//   3. remote mobile audio over WebSocket (PCM16LE mono 16 kHz)
//
// Notes:
// - This file preserves the original application structure as much as possible.
// - WebSocket support uses Boost.Beast / Boost.Asio.
// - For simplicity and maintainability, file mode remains a dedicated branch,
//   while microphone and websocket modes share the live transcription pipeline.
// - This code expects Boost headers/libraries to be available in your build.
// - The WebSocket protocol is:
//     * first text frame: JSON start message
//     * binary frames: PCM16LE mono 16k audio
//     * optional text frame: stop message
//
// Example start frame:
// {
//   "type": "start",
//   "sessionId": "abc123",
//   "sampleRate": 16000,
//   "channels": 1,
//   "bitsPerSample": 16,
//   "format": "pcm_s16le",
//   "language": "en",
//   "frameMs": 100,
//   "timestamp": true,
//   "source": "maui-mobile"
// }
//

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>

#include "portaudio.h"
#include "whisper.h"

namespace net = boost::asio;
namespace beast = boost::beast;
namespace websocket = beast::websocket;
using tcp = net::ip::tcp;

namespace Constants {
    // Audio pipeline defaults (chosen to match Whisper's native 16 kHz mono path).
    constexpr int SAMPLE_RATE = 16000;
    constexpr unsigned long FRAMES_PER_BUFFER = 1024;
    constexpr int CHANNELS = 1;
    constexpr int MIN_AUDIO_LENGTH_MS = 100;
    constexpr size_t MIN_AUDIO_SAMPLES =
        static_cast<size_t>(SAMPLE_RATE * MIN_AUDIO_LENGTH_MS / 1000.0);
    constexpr double AMBIENT_NOISE_DURATION_SECONDS = 3.0;
    constexpr double ENERGY_THRESHOLD_MULTIPLIER = 2.5;
    constexpr double ADAPTIVE_NOISE_ALPHA = 0.05;
    constexpr double ADAPTIVE_UPDATE_MAX_RMS_FRACTION = 0.80;
    constexpr double ADAPTIVE_THRESHOLD_STEP_FRACTION = 0.05;
    constexpr double ADAPTIVE_THRESHOLD_MAX_BASE_MULTIPLIER = 4.0;
    constexpr int DEFAULT_ENERGY_THRESHOLD = 1000;
    constexpr int ADAPTIVE_THRESHOLD_MIN = 200;
    constexpr int ADAPTIVE_HANGOVER_CHUNKS = 1;
    constexpr double ADAPTIVE_ONSET_TRIGGER_RATIO = 0.85;
    constexpr double VAD_PRE_ROLL_SECONDS = 0.30;
    constexpr int WHISPER_MAX_THREADS = 4;
    constexpr int MAIN_LOOP_TIMEOUT_MS = 250;
    constexpr double PHRASE_TIMEOUT_MULTIPLIER = 1.5;
    constexpr size_t MAX_QUEUED_AUDIO_CHUNKS = 64;
    constexpr size_t RAW_RING_CHUNK_CAPACITY = 256;
    constexpr double CALIBRATION_WAIT_MARGIN_SECONDS = 2.0;
    constexpr size_t MAX_WS_FRAME_BYTES = SAMPLE_RATE * sizeof(int16_t) * 1;
    constexpr int WS_IDLE_TIMEOUT_SECONDS_DEFAULT = 30;
    constexpr const char* BUILD_FINGERPRINT = "transcribe_audio.cpp|" __DATE__ " " __TIME__;
}

// Project-specific exception type so operational failures can be reported
// uniformly from PortAudio, Whisper, ffmpeg, and WebSocket code.
class AudioException : public std::runtime_error {
public:
    explicit AudioException(const std::string& message) : std::runtime_error(message) {}
};

// Whisper accepts several language aliases; normalize once so all sources
// (CLI and WebSocket clients) use the same canonical two-letter code.
static std::string normalize_whisper_language(const std::string& raw) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    const auto first = std::find_if(raw.begin(), raw.end(), not_space);
    if (first == raw.end()) {
        return "en";
    }
    const auto last = std::find_if(raw.rbegin(), raw.rend(), not_space).base();
    std::string lang(first, last);
    std::transform(lang.begin(), lang.end(), lang.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lang == "auto") {
        return lang;
    }
    const int lang_id = whisper_lang_id(lang.c_str());
    if (lang_id < 0) {
        throw AudioException("Unknown Whisper language '" + lang +
                             "'. Use a code like en, it, fr, de, es, he, sv, or auto.");
    }
    const char* canonical = whisper_lang_str(lang_id);
    return canonical ? canonical : lang;
}

// Optional context text can bias very short streaming chunks toward a domain.
//
// IMPORTANT for recent whisper.cpp releases:
// - Language selection should be expressed with params.language.
// - Translation should be expressed only with params.translate.
// - The prompt is not a reliable way to force a language, and in some model/API
//   combinations it can make short chunks drift toward plausible English text.
//
// Therefore the default prompt is intentionally empty. If domain bias is ever
// reintroduced, keep it factual and never use it as a substitute for the
// language/task parameters.
static std::string transcription_context_for_language(const std::string& language) {
    (void)language;
    return {};
}

class PortAudioRuntime {
public:
    // PortAudio has global process-wide init/terminate semantics; guard that here.
    static void ensure_initialized() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            return;
        }
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            throw AudioException(std::string("PortAudio init failed: ") + Pa_GetErrorText(err));
        }
        initialized_ = true;
    }

    static void terminate_if_initialized() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return;
        }
        Pa_Terminate();
        initialized_ = false;
    }

private:
    static std::mutex mutex_;
    static bool initialized_;
};

std::mutex PortAudioRuntime::mutex_;
bool PortAudioRuntime::initialized_ = false;

class PortAudioStream {
public:
    PortAudioStream() = default;
    ~PortAudioStream() { close(); }
    PortAudioStream(const PortAudioStream&) = delete;
    PortAudioStream& operator=(const PortAudioStream&) = delete;

    bool open(PaStreamParameters* input_parameters,
              double sample_rate,
              unsigned long frames_per_buffer,
              PaStreamCallback* callback,
              void* user_data) {
        // Re-open semantics: always close stale stream handles before opening again.
        close();
        last_err_ = Pa_OpenStream(&stream_,
                                  input_parameters,
                                  nullptr,
                                  sample_rate,
                                  frames_per_buffer,
                                  paNoFlag,
                                  callback,
                                  user_data);
        return last_err_ == paNoError;
    }

    bool start() {
        if (stream_ == nullptr) {
            last_err_ = paBadStreamPtr;
            return false;
        }
        last_err_ = Pa_StartStream(stream_);
        return last_err_ == paNoError;
    }

    void stop() noexcept {
        if (stream_ != nullptr) {
            PaError err = Pa_StopStream(stream_);
            if (err != paNoError && err != paStreamIsStopped) {
                std::cerr << "Warning: Pa_StopStream failed: " << Pa_GetErrorText(err) << std::endl;
            }
        }
    }

    void close() noexcept {
        if (stream_ != nullptr) {
            stop();
            PaError err = Pa_CloseStream(stream_);
            if (err != paNoError) {
                std::cerr << "Warning: Pa_CloseStream failed: " << Pa_GetErrorText(err) << std::endl;
            }
            stream_ = nullptr;
        }
    }

    PaError last_error() const noexcept { return last_err_; }

private:
    PaStream* stream_ = nullptr;
    PaError last_err_ = paNoError;
};

// All command-line settings are collected here so parsing and runtime setup can
// stay separate. Defaults match the live microphone path.
struct Args {
    int energy_threshold = -1;
    bool adaptive_energy = false;
    double record_timeout = 2.0;
    double phrase_timeout = 3.0;
    std::string language = "en";
    bool translate = false;
    bool pipe = false;
    bool timestamp = false;
    std::string default_microphone;
    std::string whisper_model_path;
    bool list_microphones = false;
    std::string audio_file_path;
    double vad_pre_roll = Constants::VAD_PRE_ROLL_SECONDS;
    double adaptive_silence_fraction = 0.35;
    int adaptive_hangover_chunks = 1;
    bool verbose = false;
    std::chrono::system_clock::time_point predefined_start_time{};
    bool has_predefined_start_time = false;

    // Supported sources:
    // - microphone: local capture via PortAudio
    // - file: batch transcription from media on disk
    // - websocket: remote PCM frames over WS
    std::string input_source = "microphone";
    bool websocket_server = false;
    std::string websocket_bind = "0.0.0.0";
    int websocket_port = 8080;
    bool websocket_send_transcripts = true;
    int websocket_idle_timeout = Constants::WS_IDLE_TIMEOUT_SECONDS_DEFAULT;
};

// Parsed representation of the documented "YYYY-mm-dd HH:MM:SS" timestamp.
struct DateTimeParts {
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
};

// Strict fixed-width decimal parser used by timestamp parsing. Keeping this
// local avoids locale-sensitive stream parsing for the command-line override.
bool parse_n_digits(const std::string& s, size_t pos, size_t count, int& out) {
    if (pos + count > s.size()) {
        return false;
    }

    int value = 0;
    for (size_t i = 0; i < count; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[pos + i]);
        if (!std::isdigit(c)) {
            return false;
        }
        value = value * 10 + static_cast<int>(c - '0');
    }
    out = value;
    return true;
}

bool is_leap_year(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

int days_in_month(int year, int month) {
    static constexpr int kDaysByMonth[] = {
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && is_leap_year(year)) {
        return 29;
    }
    if (month < 1 || month > 12) {
        return 0;
    }
    return kDaysByMonth[month - 1];
}

bool parse_datetime_parts(const std::string& s, DateTimeParts& out) {
    // Accept only the documented shape so ambiguous partial dates do not slip
    // through and produce surprising transcript anchors.
    if (s.size() != 19 ||
        s[4] != '-' ||
        s[7] != '-' ||
        s[10] != ' ' ||
        s[13] != ':' ||
        s[16] != ':') {
        return false;
    }

    DateTimeParts parsed{};
    if (!parse_n_digits(s, 0, 4, parsed.year) ||
        !parse_n_digits(s, 5, 2, parsed.month) ||
        !parse_n_digits(s, 8, 2, parsed.day) ||
        !parse_n_digits(s, 11, 2, parsed.hour) ||
        !parse_n_digits(s, 14, 2, parsed.minute) ||
        !parse_n_digits(s, 17, 2, parsed.second)) {
        return false;
    }

    const int month_days = days_in_month(parsed.year, parsed.month);
    if (parsed.year < 1 ||
        month_days == 0 ||
        parsed.day < 1 ||
        parsed.day > month_days ||
        parsed.hour > 23 ||
        parsed.minute > 59 ||
        parsed.second > 59) {
        return false;
    }

    out = parsed;
    return true;
}

// Days since 1970-01-01 for a Gregorian civil date. This avoids mktime so an
// explicit --predefined_start_time is not changed by local timezone or DST rules.
std::int64_t days_from_civil(int year, unsigned month, unsigned day) {
    year -= month <= 2;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(year - era * 400);
    const unsigned doy =
        (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<std::int64_t>(era) * 146097 +
           static_cast<std::int64_t>(doe) -
           719468;
}

bool parse_predefined_datetime(
    const std::string& s,
    std::chrono::system_clock::time_point& out) {
    DateTimeParts parts{};
    if (!parse_datetime_parts(s, parts)) {
        return false;
    }

    const std::int64_t days = days_from_civil(
        parts.year,
        static_cast<unsigned>(parts.month),
        static_cast<unsigned>(parts.day));
    const std::int64_t total_seconds =
        days * 86400 +
        static_cast<std::int64_t>(parts.hour) * 3600 +
        static_cast<std::int64_t>(parts.minute) * 60 +
        static_cast<std::int64_t>(parts.second);

    out = std::chrono::system_clock::time_point{
        std::chrono::seconds(total_seconds)};
    return true;
}

// Formats a time_point as UTC without applying local timezone conversion. This
// is used for explicit CLI timestamps, which are treated as already absolute.
std::string format_datetime_no_conversion(
    const std::chrono::time_point<std::chrono::system_clock>& tp) {
    const std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "[%Y-%m-%d %H:%M:%S]");
    return oss.str();
}

class EnergyVadProcessor {
public:
    using OutputCallback = std::function<void(const std::vector<int16_t>&, int energy_threshold)>;

    void configure(int sample_rate, double record_timeout, double phrase_timeout) {
        if (sample_rate <= 0 || record_timeout <= 0.0 || phrase_timeout <= 0.0) {
            throw AudioException("Invalid VAD parameters: sample rate and timeouts must be positive");
        }
        std::lock_guard<std::mutex> lock(vad_mutex_);
        sample_rate_ = sample_rate;
        record_timeout_ = record_timeout;
        phrase_timeout_ = phrase_timeout;
        recompute_limits_locked();
        clear_processing_state_locked();
    }

    void setEnergyThreshold(int threshold) {
        set_energy_threshold(threshold, true);
    }

    int getEnergyThreshold() const {
        return energy_threshold_.load(std::memory_order_relaxed);
    }

    void setAdaptiveEnergyEnabled(bool enabled) {
        adaptive_energy_enabled_.store(enabled, std::memory_order_release);
        if (!enabled) {
            // Reset the learned floor when adaptation is disabled so a later
            // re-enable starts from the current configured threshold.
            silence_floor_initialized_.store(false, std::memory_order_release);
            return;
        }
        double estimate = static_cast<double>(getEnergyThreshold()) /
                          Constants::ENERGY_THRESHOLD_MULTIPLIER;
        if (estimate <= 0.0) {
            estimate = static_cast<double>(Constants::ADAPTIVE_THRESHOLD_MIN) /
                       Constants::ENERGY_THRESHOLD_MULTIPLIER;
        }
        primeNoiseFloorEstimate(estimate);
    }

    bool adaptiveEnergyEnabled() const {
        return adaptive_energy_enabled_.load(std::memory_order_acquire);
    }

    void setVadPreRollSeconds(double seconds) {
        if (seconds < 0.0) {
            seconds = 0.0;
        }
        vad_pre_roll_seconds_.store(seconds, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(vad_mutex_);
        recompute_limits_locked();
        trim_pre_roll_locked();
    }

    void setAdaptiveHangoverChunks(int chunks) {
        adaptive_hangover_chunks_.store(std::max(1, chunks), std::memory_order_relaxed);
    }

    void setVerbose(bool verbose) {
        verbose_.store(verbose, std::memory_order_release);
    }

    void processChunk(const std::vector<int16_t>& current_chunk,
                      const OutputCallback& callback) {
        if (current_chunk.empty()) {
            return;
        }

        const double sum_squares = calculate_audio_energy(current_chunk.data(), current_chunk.size());
        const double threshold_squared =
            static_cast<double>(energy_threshold_squared_.load(std::memory_order_relaxed));
        const bool adaptive_enabled = adaptive_energy_enabled_.load(std::memory_order_relaxed);
        bool should_update_adaptive = false;
        std::vector<int16_t> completed;

        {
            std::lock_guard<std::mutex> lock(vad_mutex_);
            // All VAD state transitions are made under one mutex so microphone
            // and WebSocket paths can share the same processor safely.
            const bool speech_active = !vad_buffer_.empty();
            double speech_trigger = threshold_squared;
            if (adaptive_enabled && !speech_active) {
                // When idle, slightly lower onset threshold to reduce missed speech starts.
                const double ratio = Constants::ADAPTIVE_ONSET_TRIGGER_RATIO;
                speech_trigger *= (ratio * ratio);
            }

            const bool is_speech = sum_squares > (speech_trigger * current_chunk.size());
            if (is_speech) {
                if (vad_buffer_.empty() && !pre_roll_chunks_.empty()) {
                    // Pre-roll preserves consonant onsets just before energy crosses threshold.
                    for (const auto& pre : pre_roll_chunks_) {
                        vad_buffer_.insert(vad_buffer_.end(), pre.begin(), pre.end());
                    }
                    pre_roll_chunks_.clear();
                }
                consecutive_silence_chunks_ = 0;
                consecutive_silence_for_adaptation_ = 0;
                vad_buffer_.insert(vad_buffer_.end(), current_chunk.begin(), current_chunk.end());
            } else if (!vad_buffer_.empty()) {
                ++consecutive_silence_chunks_;
                ++consecutive_silence_for_adaptation_;
                vad_buffer_.insert(vad_buffer_.end(), current_chunk.begin(), current_chunk.end());
            } else {
                if (max_pre_roll_chunks_ > 0) {
                    pre_roll_chunks_.push_back(current_chunk);
                    trim_pre_roll_locked();
                }
                ++consecutive_silence_for_adaptation_;
            }

            if (!is_speech) {
                const int hangover_chunks = adaptive_hangover_chunks_.load(std::memory_order_relaxed);
                // Delay adaptive updates slightly after speech to avoid contaminating noise floor.
                should_update_adaptive = vad_buffer_.empty() ||
                                         consecutive_silence_for_adaptation_ >=
                                             static_cast<size_t>(hangover_chunks);
            }

            const bool flush = !vad_buffer_.empty() &&
                               (vad_buffer_.size() >= max_buffer_samples_ ||
                                consecutive_silence_chunks_ >= max_silence_chunks_);
            if (flush) {
                // Flush either because the phrase has gone quiet long enough or
                // because record_timeout capped the amount of buffered speech.
                completed = std::move(vad_buffer_);
                vad_buffer_.clear();
                consecutive_silence_chunks_ = 0;
                consecutive_silence_for_adaptation_ = 0;
            }
        }

        if (should_update_adaptive) {
            update_adaptive_threshold(sum_squares, current_chunk.size());
        }
        if (!completed.empty() && callback) {
            callback(completed, getEnergyThreshold());
        }
    }

    void flush(const OutputCallback& callback) {
        std::vector<int16_t> completed;
        {
            std::lock_guard<std::mutex> lock(vad_mutex_);
            if (vad_buffer_.empty()) {
                // No active speech; still clear pre-roll/silence counters so a
                // following session starts from a clean state.
                pre_roll_chunks_.clear();
                consecutive_silence_chunks_ = 0;
                consecutive_silence_for_adaptation_ = 0;
                return;
            }
            completed = std::move(vad_buffer_);
            clear_processing_state_locked();
        }
        if (!completed.empty() && callback) {
            callback(completed, getEnergyThreshold());
        }
    }

    void primeNoiseFloorEstimate(double rms) {
        if (rms <= 0.0) {
            return;
        }
        silence_rms_ema_.store(rms, std::memory_order_release);
        silence_floor_initialized_.store(true, std::memory_order_release);
    }

    void clearProcessingState() {
        std::lock_guard<std::mutex> lock(vad_mutex_);
        clear_processing_state_locked();
    }

private:
    static double calculate_audio_energy(const int16_t* data, size_t count) {
        double sum_squares = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const double s = static_cast<double>(data[i]);
            sum_squares += s * s;
        }
        return sum_squares;
    }

    void set_energy_threshold(int threshold, bool update_base) {
        threshold = std::max(threshold, 1);
        if (update_base) {
            base_energy_threshold_.store(threshold, std::memory_order_relaxed);
            base_threshold_initialized_.store(true, std::memory_order_release);
        } else {
            bool expected = false;
            if (base_threshold_initialized_.compare_exchange_strong(expected,
                                                                    true,
                                                                    std::memory_order_acq_rel,
                                                                    std::memory_order_acquire)) {
                base_energy_threshold_.store(threshold, std::memory_order_relaxed);
            }
        }
        energy_threshold_.store(threshold, std::memory_order_relaxed);
        energy_threshold_squared_.store(static_cast<int64_t>(threshold) * static_cast<int64_t>(threshold),
                                        std::memory_order_relaxed);
    }

    int baseEnergyThreshold() const {
        if (!base_threshold_initialized_.load(std::memory_order_acquire)) {
            return getEnergyThreshold();
        }
        return base_energy_threshold_.load(std::memory_order_relaxed);
    }

    void update_adaptive_threshold(double sum_squares, size_t sample_count) {
        if (!adaptive_energy_enabled_.load(std::memory_order_relaxed) || sample_count == 0) {
            return;
        }
        const double rms = std::sqrt(sum_squares / static_cast<double>(sample_count));
        if (rms <= 0.0) {
            return;
        }
        const int current = getEnergyThreshold();
        const double max_rms_for_noise_update =
            static_cast<double>(current) * Constants::ADAPTIVE_UPDATE_MAX_RMS_FRACTION;
        if (rms > max_rms_for_noise_update) {
            // Loud non-speech frames are likely transients: do not treat them as background noise.
            return;
        }
        if (!silence_floor_initialized_.load(std::memory_order_acquire)) {
            silence_rms_ema_.store(rms, std::memory_order_release);
            silence_floor_initialized_.store(true, std::memory_order_release);
            return;
        }
        const double prev = silence_rms_ema_.load(std::memory_order_relaxed);
        // Track silence RMS with an EMA so thresholds follow room noise without
        // reacting sharply to a single quiet or loud frame.
        const double updated = (1.0 - Constants::ADAPTIVE_NOISE_ALPHA) * prev +
                               Constants::ADAPTIVE_NOISE_ALPHA * rms;
        silence_rms_ema_.store(updated, std::memory_order_release);
        double desired = updated * Constants::ENERGY_THRESHOLD_MULTIPLIER;
        desired = std::max(desired, static_cast<double>(Constants::ADAPTIVE_THRESHOLD_MIN));

        const int base = std::max(baseEnergyThreshold(), Constants::ADAPTIVE_THRESHOLD_MIN);
        const double max_allowed =
            std::min(32767.0,
                     static_cast<double>(base) *
                         Constants::ADAPTIVE_THRESHOLD_MAX_BASE_MULTIPLIER);
        desired = std::min(desired, max_allowed);

        int target = static_cast<int>(std::lround(desired));
        int max_step = std::max(5,
                                static_cast<int>(std::lround(current *
                                                             Constants::ADAPTIVE_THRESHOLD_STEP_FRACTION)));
        // Step clamping keeps threshold transitions smooth and prevents oscillation.
        max_step = std::max(max_step, 1);
        if (target > current + max_step) {
            target = current + max_step;
        } else if (target < current - max_step) {
            target = current - max_step;
        }
        if (target != current) {
            set_energy_threshold(target, false);
            if (verbose_.load(std::memory_order_acquire)) {
                std::cout << "[Adaptive VAD] RMS=" << std::fixed << std::setprecision(1) << rms
                          << " EMA=" << updated
                          << " threshold -> " << target << std::endl;
            }
        }
    }

    void recompute_limits_locked() {
        // Convert user-facing durations into sample/chunk counts once, under
        // lock, so the real-time path does not repeat this work.
        max_buffer_samples_ = std::max<size_t>(
            Constants::MIN_AUDIO_SAMPLES,
            static_cast<size_t>(std::ceil(static_cast<double>(sample_rate_) * record_timeout_)));
        max_silence_chunks_ = std::max<size_t>(
            1,
            static_cast<size_t>(std::ceil(phrase_timeout_ * static_cast<double>(sample_rate_) /
                                          static_cast<double>(Constants::FRAMES_PER_BUFFER))));
        const double pre_roll_seconds = vad_pre_roll_seconds_.load(std::memory_order_relaxed);
        max_pre_roll_chunks_ = pre_roll_seconds <= 0.0
            ? 0
            : static_cast<size_t>(std::ceil(pre_roll_seconds * static_cast<double>(sample_rate_) /
                                            static_cast<double>(Constants::FRAMES_PER_BUFFER)));
    }

    void trim_pre_roll_locked() {
        while (pre_roll_chunks_.size() > max_pre_roll_chunks_) {
            pre_roll_chunks_.pop_front();
        }
    }

    void clear_processing_state_locked() {
        vad_buffer_.clear();
        pre_roll_chunks_.clear();
        consecutive_silence_chunks_ = 0;
        consecutive_silence_for_adaptation_ = 0;
    }

    std::atomic<int> energy_threshold_{Constants::DEFAULT_ENERGY_THRESHOLD};
    std::atomic<int64_t> energy_threshold_squared_{
        static_cast<int64_t>(Constants::DEFAULT_ENERGY_THRESHOLD) *
        static_cast<int64_t>(Constants::DEFAULT_ENERGY_THRESHOLD)};
    std::atomic<int> base_energy_threshold_{Constants::DEFAULT_ENERGY_THRESHOLD};
    std::atomic<bool> base_threshold_initialized_{false};
    std::atomic<bool> adaptive_energy_enabled_{false};
    std::atomic<double> vad_pre_roll_seconds_{Constants::VAD_PRE_ROLL_SECONDS};
    std::atomic<double> silence_rms_ema_{0.0};
    std::atomic<bool> silence_floor_initialized_{false};
    std::atomic<int> adaptive_hangover_chunks_{Constants::ADAPTIVE_HANGOVER_CHUNKS};
    std::atomic<bool> verbose_{false};
    // Mutable phrase-detection state. Access only while holding vad_mutex_.
    std::mutex vad_mutex_;
    std::vector<int16_t> vad_buffer_;
    std::deque<std::vector<int16_t>> pre_roll_chunks_;
    size_t consecutive_silence_chunks_ = 0;
    size_t consecutive_silence_for_adaptation_ = 0;
    int sample_rate_ = Constants::SAMPLE_RATE;
    double record_timeout_ = 2.0;
    double phrase_timeout_ = 3.0;
    size_t max_buffer_samples_ = static_cast<size_t>(Constants::SAMPLE_RATE * 2.0);
    size_t max_silence_chunks_ = 1;
    size_t max_pre_roll_chunks_ = 1;
};

// Abstract recorder API keeps the capture source independent from the main
// transcription loop. Today the concrete implementation is PortAudio.
class AudioRecorder {
public:
    virtual ~AudioRecorder() = default;
    virtual bool startRecording(std::function<void(const std::vector<int16_t>&)> callback,
                                int sample_rate,
                                double record_timeout,
                                double phrase_timeout) = 0;
    virtual void stopRecording() = 0;
    virtual void adjustForAmbientNoise(int energy_threshold) = 0;
    virtual void setEnergyThreshold(int threshold) = 0;
    virtual int getEnergyThreshold() const = 0;
    virtual void setAdaptiveEnergyEnabled(bool enabled) = 0;
    virtual void setVadPreRollSeconds(double seconds) = 0;
    virtual void setPreferredDeviceName(const std::string& name) = 0;
    virtual void setAdaptiveHangoverChunks(int chunks) = 0;
    virtual void setVerbose(bool verbose) = 0;
    static std::vector<std::string> listMicrophoneNames();
};

class PortAudioRecorder final : public AudioRecorder {
public:
    // Owns the PortAudio stream plus a worker that drains raw callback data
    // through the shared VAD processor.
    PortAudioRecorder() {
        PortAudioRuntime::ensure_initialized();
        raw_ring_.resize(Constants::RAW_RING_CHUNK_CAPACITY);
    }

    ~PortAudioRecorder() override {
        stopRecording();
    }

    void setPreferredDeviceName(const std::string& name) override {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        preferred_device_name_ = name;
    }

    void setAdaptiveHangoverChunks(int chunks) override {
        vad_.setAdaptiveHangoverChunks(chunks);
    }

    void setVerbose(bool v) override {
        vad_.setVerbose(v);
    }

    bool startRecording(std::function<void(const std::vector<int16_t>&)> callback,
                        int sample_rate,
                        double record_timeout,
                        double phrase_timeout) override {
        if (sample_rate <= 0 || record_timeout <= 0.0 || phrase_timeout <= 0.0) {
            throw AudioException("Invalid parameters: sample rate and timeouts must be positive");
        }

        stopRecording();
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            // Callback can be swapped during ambient calibration, so protect it
            // separately from the real-time audio ring.
            audio_callback_ = std::move(callback);
        }

        sample_rate_ = sample_rate;
        record_timeout_ = record_timeout;
        phrase_timeout_ = phrase_timeout;
        vad_.configure(sample_rate_, record_timeout_, phrase_timeout_);
        // Runtime limits are computed once per start() so hot callbacks stay lightweight.

        vad_.clearProcessingState();
        reset_ring();
        dropped_callback_chunks_.store(0, std::memory_order_relaxed);

        PaStreamParameters input_parameters{};
        input_parameters.device = pick_input_device(preferred_device_name_);
        if (input_parameters.device == paNoDevice) {
            throw AudioException("No input device available.");
        }
        const PaDeviceInfo* device_info = Pa_GetDeviceInfo(input_parameters.device);
        if (device_info == nullptr) {
            throw AudioException("Failed to query selected input device.");
        }
        input_parameters.channelCount = Constants::CHANNELS;
        input_parameters.sampleFormat = paInt16;
        input_parameters.suggestedLatency = device_info->defaultLowInputLatency;
        input_parameters.hostApiSpecificStreamInfo = nullptr;

        // Start the consumer before PortAudio begins producing callbacks so the
        // ring has a reader as soon as audio arrives.
        worker_running_.store(true, std::memory_order_release);
        recording_active_.store(true, std::memory_order_release);
        // Separate worker thread keeps callback path minimal and real-time friendly.
        worker_thread_ = std::thread(&PortAudioRecorder::processing_worker, this);

        if (!stream_.open(&input_parameters,
                          static_cast<double>(sample_rate_),
                          Constants::FRAMES_PER_BUFFER,
                          &PortAudioRecorder::pa_callback,
                          this)) {
            recording_active_.store(false, std::memory_order_release);
            worker_running_.store(false, std::memory_order_release);
            ring_cv_.notify_all();
            if (worker_thread_.joinable()) {
                worker_thread_.join();
            }
            throw AudioException(std::string("PortAudio error (open stream): ") +
                                 Pa_GetErrorText(stream_.last_error()));
        }

        if (!stream_.start()) {
            stream_.close();
            recording_active_.store(false, std::memory_order_release);
            worker_running_.store(false, std::memory_order_release);
            ring_cv_.notify_all();
            if (worker_thread_.joinable()) {
                worker_thread_.join();
            }
            throw AudioException(std::string("PortAudio error (start stream): ") +
                                 Pa_GetErrorText(stream_.last_error()));
        }

        std::cout << "Started recording on: " << device_info->name << std::endl;
        return true;
    }

    void stopRecording() override {
        const bool was_active = recording_active_.exchange(false, std::memory_order_acq_rel);
        // Stop/close first to prevent new callbacks, then wake and join the
        // worker so it can drain any already-queued chunks.
        stream_.stop();
        stream_.close();
        worker_running_.store(false, std::memory_order_release);
        ring_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        if (was_active) {
            vad_.clearProcessingState();
            reset_ring();
        }
    }

    void adjustForAmbientNoise(int user_energy_threshold) override {
        if (user_energy_threshold != -1) {
            setEnergyThreshold(user_energy_threshold);
            std::cout << "Using provided energy threshold: " << user_energy_threshold << std::endl;
            return;
        }

        std::cout << "Adjusting for ambient noise (listening for "
                  << Constants::AMBIENT_NOISE_DURATION_SECONDS << " seconds)..." << std::endl;

        auto previous_callback = get_callback_copy();
        const bool was_running = recording_active_.load(std::memory_order_acquire);
        std::vector<int16_t> collected;
        std::mutex collected_mutex;
        std::condition_variable collected_cv;
        bool done = false;

        auto collector = [&](const std::vector<int16_t>& chunk) {
            // Calibration deliberately collects raw chunks; VAD is bypassed
            // below so silence estimation is based on actual ambient audio.
            std::lock_guard<std::mutex> lock(collected_mutex);
            collected.insert(collected.end(), chunk.begin(), chunk.end());
            if (collected.size() >= static_cast<size_t>(
                    Constants::SAMPLE_RATE * Constants::AMBIENT_NOISE_DURATION_SECONDS)) {
                done = true;
                collected_cv.notify_one();
            }
        };

        set_callback(collector);
        // Bypass VAD during calibration: we want raw environmental noise, not segmented speech.
        bypass_vad_.store(true, std::memory_order_release);
        if (!was_running) {
            startRecording(collector, sample_rate_, record_timeout_, phrase_timeout_);
        }
        {
            std::unique_lock<std::mutex> lock(collected_mutex);
            const auto max_wait = std::chrono::duration<double>(
                Constants::AMBIENT_NOISE_DURATION_SECONDS + Constants::CALIBRATION_WAIT_MARGIN_SECONDS);
            (void)collected_cv.wait_for(lock, max_wait, [&] { return done; });
        }
        bypass_vad_.store(false, std::memory_order_release);
        if (!was_running) {
            stopRecording();
        }
        set_callback(previous_callback);

        if (collected.empty()) {
            throw AudioException("No ambient-noise samples collected.");
        }

        long double sum_squares = 0.0L;
        for (int16_t s : collected) {
            const long double v = static_cast<long double>(s);
            sum_squares += v * v;
        }
        const long double rms = std::sqrt(sum_squares / static_cast<long double>(collected.size()));
        const int threshold = static_cast<int>(std::llround(rms * Constants::ENERGY_THRESHOLD_MULTIPLIER));
        setEnergyThreshold(std::max(threshold, Constants::ADAPTIVE_THRESHOLD_MIN));
        if (vad_.adaptiveEnergyEnabled()) {
            vad_.primeNoiseFloorEstimate(static_cast<double>(rms));
        }
        std::cout << "Adjusted energy threshold to: " << getEnergyThreshold() << std::endl;
    }

    void setEnergyThreshold(int threshold) override {
        vad_.setEnergyThreshold(threshold);
    }

    int getEnergyThreshold() const override {
        return vad_.getEnergyThreshold();
    }

    void setAdaptiveEnergyEnabled(bool enabled) override {
        vad_.setAdaptiveEnergyEnabled(enabled);
    }

    void setVadPreRollSeconds(double seconds) override {
        vad_.setVadPreRollSeconds(seconds);
    }

private:
    // RawChunk is the unit moved between the real-time PortAudio callback and
    // the normal-priority processing worker.
    struct RawChunk {
        std::vector<int16_t> samples;
        size_t frames = 0;
    };

    static int pa_callback(const void* input_buffer,
                           void* output_buffer,
                           unsigned long frames_per_buffer,
                           const PaStreamCallbackTimeInfo*,
                           PaStreamCallbackFlags,
                           void* user_data) {
        (void)output_buffer;
        auto* recorder = static_cast<PortAudioRecorder*>(user_data);
        if (input_buffer == nullptr ||
            !recorder->recording_active_.load(std::memory_order_acquire)) {
            return paContinue;
        }
        const int16_t* in = static_cast<const int16_t*>(input_buffer);
        // Never block in PortAudio callback: enqueue and return immediately.
        recorder->push_raw_chunk_from_callback(in, static_cast<size_t>(frames_per_buffer));
        return paContinue;
    }

    void push_raw_chunk_from_callback(const int16_t* data, size_t frames) noexcept {
        if (frames == 0) {
            return;
        }
        std::unique_lock<std::mutex> lock(ring_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            // The audio callback must never block. Dropping a frame is better
            // than risking PortAudio underflow or a stalled input device.
            dropped_callback_chunks_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (ring_size_ == raw_ring_.size()) {
            // Drop oldest data when saturated; favor newest audio to keep transcripts current.
            ring_tail_ = (ring_tail_ + 1) % raw_ring_.size();
            --ring_size_;
            dropped_callback_chunks_.fetch_add(1, std::memory_order_relaxed);
        }
        RawChunk& slot = raw_ring_[ring_head_];
        if (slot.samples.size() != frames) {
            try {
                slot.samples.resize(frames);
            } catch (...) {
                dropped_callback_chunks_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        std::memcpy(slot.samples.data(), data, frames * sizeof(int16_t));
        slot.frames = frames;
        ring_head_ = (ring_head_ + 1) % raw_ring_.size();
        ++ring_size_;
        lock.unlock();
        ring_cv_.notify_one();
    }

    bool pop_raw_chunk(RawChunk& out) {
        std::unique_lock<std::mutex> lock(ring_mutex_);
        ring_cv_.wait(lock, [&] {
            return ring_size_ > 0 || !worker_running_.load(std::memory_order_acquire);
        });
        if (ring_size_ == 0) {
            return false;
        }
        out = std::move(raw_ring_[ring_tail_]);
        // The moved-from vector remains in the ring slot and will be resized or
        // overwritten by the producer on its next use.
        ring_tail_ = (ring_tail_ + 1) % raw_ring_.size();
        --ring_size_;
        return true;
    }

    void processing_worker() {
        try {
            RawChunk chunk;
            // Drain queued chunks even after stop request so we do not lose buffered speech.
            while (worker_running_.load(std::memory_order_acquire) || ring_size_snapshot() > 0) {
                if (!pop_raw_chunk(chunk)) {
                    continue;
                }
                if (!chunk.samples.empty()) {
                    process_chunk(chunk.samples);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Recorder worker error: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Recorder worker error: unknown exception" << std::endl;
        }
    }

    size_t ring_size_snapshot() const {
        std::lock_guard<std::mutex> lock(ring_mutex_);
        return ring_size_;
    }

    void process_chunk(const std::vector<int16_t>& current_chunk) {
        auto callback = get_callback_copy();
        if (!callback) {
            return;
        }
        if (bypass_vad_.load(std::memory_order_acquire)) {
            // Used only for calibration, where every frame is signal for the
            // ambient-noise estimator.
            callback(current_chunk);
            return;
        }

        vad_.processChunk(current_chunk,
                          [&](const std::vector<int16_t>& completed, int) {
                              callback(completed);
                          });
    }

    void reset_ring() {
        std::lock_guard<std::mutex> lock(ring_mutex_);
        ring_head_ = 0;
        ring_tail_ = 0;
        ring_size_ = 0;
        for (auto& chunk : raw_ring_) {
            chunk.frames = 0;
        }
    }

    static int pick_input_device(const std::string& preferred_name) {
        const int default_device = Pa_GetDefaultInputDevice();
        if (preferred_name.empty()) {
            return default_device;
        }
        const int device_count = Pa_GetDeviceCount();
        if (device_count < 0) {
            return default_device;
        }
        auto lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        };
        const std::string needle = lower(preferred_name);
        // Match by substring to tolerate host API prefixes and localized device
        // names reported by PortAudio.
        for (int i = 0; i < device_count; ++i) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
            if (info == nullptr || info->maxInputChannels <= 0) {
                continue;
            }
            const std::string device_name = info->name != nullptr ? info->name : "";
            if (lower(device_name).find(needle) != std::string::npos) {
                return i;
            }
        }
        return default_device;
    }

    std::function<void(const std::vector<int16_t>&)> get_callback_copy() {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        return audio_callback_;
    }

    void set_callback(std::function<void(const std::vector<int16_t>&)> callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        audio_callback_ = std::move(callback);
    }

    PortAudioStream stream_;
    std::function<void(const std::vector<int16_t>&)> audio_callback_;
    std::mutex callback_mutex_;
    std::atomic<bool> recording_active_{false};
    std::atomic<bool> worker_running_{false};
    std::thread worker_thread_;
    mutable std::mutex ring_mutex_;
    std::condition_variable ring_cv_;
    std::vector<RawChunk> raw_ring_;
    size_t ring_head_ = 0;
    size_t ring_tail_ = 0;
    size_t ring_size_ = 0;
    std::atomic<size_t> dropped_callback_chunks_{0};
    EnergyVadProcessor vad_;
    std::atomic<bool> bypass_vad_{false};
    std::string preferred_device_name_;
    int sample_rate_ = Constants::SAMPLE_RATE;
    double record_timeout_ = 2.0;
    double phrase_timeout_ = 3.0;
};

std::vector<std::string> AudioRecorder::listMicrophoneNames() {
    // Enumeration also initializes PortAudio so --list_microphones works even
    // when no recorder instance is created.
    PortAudioRuntime::ensure_initialized();
    std::vector<std::string> names;
    const int device_count = Pa_GetDeviceCount();
    if (device_count < 0) {
        throw AudioException(std::string("PortAudio error (device count): ") +
                             Pa_GetErrorText(device_count));
    }
    for (int i = 0; i < device_count; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info != nullptr && info->maxInputChannels > 0) {
            names.emplace_back(info->name != nullptr ? info->name : "(unnamed)");
        }
    }
    return names;
}

class WhisperModel {
public:
    // Whisper contexts are not shared across model files; this wrapper serializes
    // all calls through AudioTranscriber so the context is used by one worker.
    explicit WhisperModel(const std::string& model_path) : model_path_(model_path) {
        if (!std::filesystem::exists(model_path_)) {
            throw AudioException("Model file does not exist: " + model_path_);
        }
        std::cout << "Loading Whisper model from: " << model_path_ << std::endl;
        const whisper_context_params cparams = whisper_context_default_params();
        ctx_ = whisper_init_from_file_with_params(model_path_.c_str(), cparams);
        if (ctx_ == nullptr) {
            throw AudioException("Failed to load Whisper model from " + model_path_);
        }
        is_multilingual_ = whisper_is_multilingual(ctx_) != 0;
    }

    ~WhisperModel() {
        if (ctx_ != nullptr) {
            whisper_free(ctx_);
        }
    }

    WhisperModel(const WhisperModel&) = delete;
    WhisperModel& operator=(const WhisperModel&) = delete;

    void validate_language(const std::string& language, bool translate = false) const {
        const std::string normalized = normalize_whisper_language(language);
        // English-only models ignore non-English language requests, which looks
        // like translation from the user's perspective. Fail early instead.
        if (!is_multilingual_ && (translate || normalized != "en")) {
            if (translate) {
                throw AudioException("Whisper translation requires a multilingual model, but the loaded model is English-only.");
            }
            throw AudioException("Requested Whisper language '" + normalized +
                                 "' requires a multilingual model, but the loaded model is English-only.");
        }
    }

    std::string transcribe(const std::vector<float>& audio,
                           const std::string& language,
                           bool translate = false) {
        if (ctx_ == nullptr || audio.empty()) {
            return {};
        }

        const std::string effective_language = normalize_whisper_language(language);
        const bool auto_language = effective_language == "auto";

        // WebSocket sessions can provide a per-session language, so keep the
        // model/language compatibility check on every transcription request.
        if (!is_multilingual_ && (translate || effective_language != "en")) {
            if (translate) {
                throw AudioException("Whisper translation requires a multilingual model, but the loaded model is English-only.");
            }
            throw AudioException("Requested Whisper language '" + effective_language +
                                 "' requires a multilingual model, but the loaded model is English-only.");
        }

        const std::vector<float>* input = &audio;
        std::vector<float> padded;
        // Whisper's hard floor is 100 ms; pad to 200 ms so we're safely above it
        // regardless of integer-rounding differences across whisper.cpp versions.
        constexpr size_t kWhisperPadSamples =
            static_cast<size_t>(Constants::SAMPLE_RATE * 200 / 1000);
        if (audio.size() < kWhisperPadSamples) {
            padded = audio;
            padded.resize(kWhisperPadSamples, 0.0f);
            input = &padded;
        }

        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        // Current whisper.cpp contract, as of the v1.8.x header:
        //   - params.language == nullptr, "", or "auto" means auto-detect.
        //   - params.detect_language asks whisper.cpp to run language detection.
        //   - params.translate controls translation to English.
        //
        // For --language it we must *not* enable detection and we must *not*
        // rely on the prompt to steer language. This mirrors whisper-cli more
        // closely and avoids a class of regressions where short chunks decode as
        // plausible English even though translate=false.
        params.language = auto_language ? nullptr : effective_language.c_str();
        params.detect_language = auto_language;
        params.translate = translate;

        // Keep text context enabled. For file mode this is important because the
        // file is processed as consecutive record_timeout-sized chunks; disabling
        // context makes every chunk look like an isolated utterance and increases
        // hallucinations. Live mode still has VAD phrase boundaries and the
        // secondary silence gate to reduce cross-phrase contamination.
        params.no_context = false;
        params.no_timestamps = true;
        params.single_segment = true;

        // Own printing in this program rather than letting whisper.cpp print
        // progress or partial text from inside the library.
        params.print_special = false;
        params.print_realtime = false;
        params.print_progress = false;
        params.print_timestamps = false;

        params.suppress_blank = true;
        params.suppress_nst = true;
        params.temperature = 0.0f;
        params.temperature_inc = 0.0f;

        // Do not set an initial prompt by default. The language/task tokens are
        // the authoritative interface. A prompt can be useful for domain terms,
        // but it should be opt-in because it can bias short chunks unpredictably.
        const std::string initial_prompt = transcription_context_for_language(effective_language);
        if (!initial_prompt.empty()) {
            params.initial_prompt = initial_prompt.c_str();
            params.carry_initial_prompt = false;
        }

        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (hw <= 0) {
            hw = 1;
        }
        params.n_threads = std::min(Constants::WHISPER_MAX_THREADS, hw);

        {
            // Log once per effective language so runtime output proves which
            // Whisper task flags reached the library. Include whisper.cpp's
            // version string when available so field reports identify the exact
            // ABI/API behaviour under test.
            std::lock_guard<std::mutex> lock(log_mutex_);
            const std::string log_key = effective_language + (translate ? ":translate" : ":transcribe");
            if (logged_languages_.insert(log_key).second) {
                std::cerr << "[Whisper] version=" << whisper_version()
                          << " language=" << (params.language ? params.language : "auto")
                          << " detect_language=" << (params.detect_language ? 1 : 0)
                          << " translate=" << (params.translate ? 1 : 0)
                          << " no_timestamps=" << (params.no_timestamps ? 1 : 0)
                          << " no_context=" << (params.no_context ? 1 : 0)
                          << " single_segment=" << (params.single_segment ? 1 : 0)
                          << " temperature_inc=" << params.temperature_inc
                          << " prompt=" << (params.initial_prompt ? 1 : 0)
                          << " multilingual_model=" << (is_multilingual_ ? 1 : 0)
                          << std::endl;
            }
        }

        if (whisper_full(ctx_,
                         params,
                         input->data(),
                         static_cast<int>(input->size())) != 0) {
            return {};
        }

        const int result_lang_id = whisper_full_lang_id(ctx_);
        const char* result_lang = result_lang_id >= 0 ? whisper_lang_str(result_lang_id) : nullptr;
        if (verbose_) {
            std::cerr << "[Whisper result] requested_language=" << effective_language
                      << " result_lang_id=" << result_lang_id
                      << " result_language=" << (result_lang ? result_lang : "unknown")
                      << std::endl;
        }
        if (!auto_language && result_lang != nullptr && effective_language != result_lang) {
            // Do not discard text here: this diagnostic is deliberately non-fatal
            // because whisper.cpp may report detection metadata even when a fixed
            // language token was supplied. The message is a strong signal when
            // comparing this wrapper against whisper-cli.
            std::lock_guard<std::mutex> lock(log_mutex_);
            const std::string key = effective_language + "->" + result_lang;
            if (logged_language_mismatches_.insert(key).second) {
                std::cerr << "[Whisper warning] requested fixed language '" << effective_language
                          << "' but whisper_full_lang_id() reported '" << result_lang
                          << "'. If transcription is in the wrong language, compare with "
                          << "whisper-cli using the same model and audio."
                          << std::endl;
            }
        }

        std::string out;
        const int segments = whisper_full_n_segments(ctx_);
        for (int i = 0; i < segments; ++i) {
            const float no_speech_prob = whisper_full_get_segment_no_speech_prob(ctx_, i);
            if (verbose_) {
                const char* raw = whisper_full_get_segment_text(ctx_, i);
                std::cerr << "[whisper] seg=" << i
                          << " no_speech_prob=" << std::fixed << std::setprecision(3) << no_speech_prob
                          << " thold=" << params.no_speech_thold
                          << " text=\"" << (raw ? raw : "") << "\"" << std::endl;
            }
            // Skip segments where Whisper is confident there is no speech.
            if (no_speech_prob > params.no_speech_thold) {
                continue;
            }
            const char* text = whisper_full_get_segment_text(ctx_, i);
            if (text != nullptr) {
                out += text;
            }
        }
        return out;
    }

    void set_verbose(bool v) { verbose_ = v; }

private:
    whisper_context* ctx_ = nullptr;
    std::string model_path_;
    bool verbose_ = false;
    bool is_multilingual_ = false;
    std::mutex log_mutex_;
    std::unordered_set<std::string> logged_languages_;
    std::unordered_set<std::string> logged_language_mismatches_;
};

class AudioTranscriber {
public:
    // A single Whisper worker preserves order and avoids concurrent use of the
    // same whisper_context, which whisper.cpp documents as not thread-safe.
    AudioTranscriber(WhisperModel& model, std::string language, bool translate)
        : model_(model), default_language_(std::move(language)), default_translate_(translate) {
        running_.store(true, std::memory_order_release);
        worker_ = std::thread(&AudioTranscriber::worker_loop, this);
    }

    ~AudioTranscriber() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            running_.store(false, std::memory_order_release);
        }
        queue_cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    AudioTranscriber(const AudioTranscriber&) = delete;
    AudioTranscriber& operator=(const AudioTranscriber&) = delete;

    std::future<std::string> transcribe_async(std::vector<float> audio,
                                              std::string language = {},
                                              std::optional<bool> translate = std::nullopt) {
        std::promise<std::string> promise;
        auto future = promise.get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            // Ordered queue preserves temporal ordering for phrase reconstruction.
            transcription_queue_.push(Task{std::move(audio),
                                           std::move(language),
                                           translate,
                                           std::move(promise)});
        }
        queue_cv_.notify_one();
        return future;
    }

private:
    struct Task {
        // Audio is already normalized to float PCM before it reaches the worker.
        std::vector<float> audio;
        std::string language;
        std::optional<bool> translate;
        std::promise<std::string> promise;
    };

    void worker_loop() {
        for (;;) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [&] {
                    return !transcription_queue_.empty() ||
                           !running_.load(std::memory_order_acquire);
                });
                if (!running_.load(std::memory_order_acquire) && transcription_queue_.empty()) {
                    break;
                }
                task = std::move(transcription_queue_.front());
                transcription_queue_.pop();
            }
            try {
                // Fulfill every promise, including empty chunks, so the main
                // loop never waits indefinitely on a queued job.
                if (task.audio.empty()) {
                    task.promise.set_value({});
                } else {
                    const std::string& language =
                        task.language.empty() ? default_language_ : task.language;
                    const bool translate = task.translate.value_or(default_translate_);
                    task.promise.set_value(model_.transcribe(task.audio, language, translate));
                }
            } catch (...) {
                task.promise.set_exception(std::current_exception());
            }
        }
    }

    WhisperModel& model_;
    std::string default_language_;
    bool default_translate_ = false;
    std::thread worker_;
    std::queue<Task> transcription_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{false};
};

namespace {
// The WebSocket protocol uses small flat JSON control messages. These helpers
// intentionally cover only the subset needed for those messages.
// Minimal JSON escaping for small control payloads without pulling a full JSON library.
std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (char c : input) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

size_t skip_json_whitespace(const std::string& json, size_t pos) {
    while (pos < json.size()) {
        const unsigned char c = static_cast<unsigned char>(json[pos]);
        if (!std::isspace(c)) {
            break;
        }
        ++pos;
    }
    return pos;
}

std::optional<std::string> parse_json_quoted_string(const std::string& json, size_t& pos) {
    if (pos >= json.size() || json[pos] != '"') {
        return std::nullopt;
    }
    ++pos;
    std::string out;
    out.reserve(16);
    while (pos < json.size()) {
        const char c = json[pos++];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            if (pos >= json.size()) {
                return std::nullopt;
            }
            const char esc = json[pos++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                // Keep unicode escapes literal: protocol routing fields are
                // ASCII names and ids, so full Unicode decoding is unnecessary.
                case 'u': out += "\\u"; break;
                default: out.push_back(esc); break;
            }
            continue;
        }
        out.push_back(c);
    }
    return std::nullopt;
}

std::string extract_json_string_field(const std::string& json, const std::string& field) {
    // This parser is intentionally shallow because start/stop frames are flat
    // JSON objects controlled by our mobile client.
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return {};
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return {};
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    auto parsed = parse_json_quoted_string(json, value_pos);
    return parsed.has_value() ? *parsed : std::string{};
}

int extract_json_int_field(const std::string& json, const std::string& field, int fallback) {
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return fallback;
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return fallback;
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    size_t end = value_pos;
    if (end < json.size() && (json[end] == '-' || json[end] == '+')) {
        ++end;
    }
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end]))) {
        ++end;
    }
    if (end == value_pos || (end == value_pos + 1 && (json[value_pos] == '-' || json[value_pos] == '+'))) {
        return fallback;
    }
    try {
        return std::stoi(json.substr(value_pos, end - value_pos));
    } catch (...) {
        return fallback;
    }
}

std::string ascii_lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string trim_ascii_whitespace(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    const auto first = std::find_if(s.begin(), s.end(), not_space);
    if (first == s.end()) {
        return {};
    }
    const auto last = std::find_if(s.rbegin(), s.rend(), not_space).base();
    return std::string(first, last);
}

bool json_value_delimiter(char c) {
    return c == ',' || c == '}' || c == ']' || std::isspace(static_cast<unsigned char>(c));
}

bool extract_json_bool_field(const std::string& json, const std::string& field, bool fallback) {
    // Accept both real booleans and string-ish form values to keep the protocol
    // tolerant of clients that serialize UI settings as strings.
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return fallback;
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return fallback;
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    if (value_pos >= json.size()) {
        return fallback;
    }

    if (json[value_pos] == '"') {
        auto parsed = parse_json_quoted_string(json, value_pos);
        if (!parsed.has_value()) {
            return fallback;
        }
        const std::string value = ascii_lower_copy(trim_ascii_whitespace(*parsed));
        if (value == "true" || value == "1" || value == "on" || value == "yes") {
            return true;
        }
        if (value == "false" || value == "0" || value == "off" || value == "no") {
            return false;
        }
        return fallback;
    }

    const size_t remaining = json.size() - value_pos;
    if (remaining >= 4 &&
        json.compare(value_pos, 4, "true") == 0 &&
        (value_pos + 4 == json.size() || json_value_delimiter(json[value_pos + 4]))) {
        return true;
    }
    if (remaining >= 5 &&
        json.compare(value_pos, 5, "false") == 0 &&
        (value_pos + 5 == json.size() || json_value_delimiter(json[value_pos + 5]))) {
        return false;
    }
    if (json[value_pos] == '1' &&
        (value_pos + 1 == json.size() || json_value_delimiter(json[value_pos + 1]))) {
        return true;
    }
    if (json[value_pos] == '0' &&
        (value_pos + 1 == json.size() || json_value_delimiter(json[value_pos + 1]))) {
        return false;
    }
    return fallback;
}

bool json_message_type_is(const std::string& json, const char* expected) {
    return extract_json_string_field(json, "type") == expected;
}
}

class WebSocketPcmServer {
public:
    // Per-connection state. The WebSocket server receives arbitrary client frame
    // sizes, then converts them into fixed-size chunks for the shared VAD.
    struct ClientContext {
        std::string session_id;
        std::string language = "en";
        int sample_rate = Constants::SAMPLE_RATE;
        int channels = 1;
        int bits_per_sample = 16;
        std::string format = "pcm_s16le";
        int frame_ms = Constants::MIN_AUDIO_LENGTH_MS;
        size_t target_frame_samples = Constants::FRAMES_PER_BUFFER;
        std::vector<int16_t> pending_samples;
        EnergyVadProcessor vad;
        bool started = false;
        bool include_timestamps = false;
    };

    using AudioCallback = std::function<void(const std::vector<int16_t>&,
                                             const std::string& session_id,
                                             int energy_threshold,
                                             const std::string& language)>;

    WebSocketPcmServer(std::string bind_address,
                       int port,
                       bool send_transcripts,
                       int idle_timeout_seconds,
                       bool include_transcript_timestamps,
                       std::string default_language,
                       int energy_threshold,
                       bool adaptive_energy,
                       double vad_pre_roll_seconds,
                       int adaptive_hangover_chunks,
                       bool verbose,
                       double phrase_timeout_seconds,
                       double record_timeout_seconds = 2.0)
        : bind_address_(std::move(bind_address)),
          port_(port),
          send_transcripts_(send_transcripts),
          idle_timeout_seconds_(idle_timeout_seconds > 0
                                    ? idle_timeout_seconds
                                    : Constants::WS_IDLE_TIMEOUT_SECONDS_DEFAULT),
          include_transcript_timestamps_(include_transcript_timestamps),
          default_language_(std::move(default_language)),
          initial_energy_threshold_(std::max(energy_threshold, Constants::ADAPTIVE_THRESHOLD_MIN)),
          adaptive_energy_enabled_(adaptive_energy),
          vad_pre_roll_seconds_(vad_pre_roll_seconds),
          adaptive_hangover_chunks_(std::max(1, adaptive_hangover_chunks)),
          verbose_(verbose),
          phrase_timeout_seconds_(phrase_timeout_seconds > 0.0 ? phrase_timeout_seconds : 3.0),
          record_timeout_seconds_(record_timeout_seconds > 0.0 ? record_timeout_seconds : 2.0) {
    }

    ~WebSocketPcmServer() {
        stop();
    }

    bool start(AudioCallback callback) {
        callback_ = std::move(callback);
        running_.store(true, std::memory_order_release);
        try {
            // Keep accept loop isolated from the transcription/main thread.
            server_thread_ = std::thread(&WebSocketPcmServer::server_loop, this);
        } catch (const std::exception& e) {
            std::cerr << "Failed to start WebSocket server thread: " << e.what() << std::endl;
            running_.store(false, std::memory_order_release);
            return false;
        }
        return true;
    }

    void stop() {
        if (!running_.exchange(false, std::memory_order_acq_rel)) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(acceptor_mutex_);
            if (acceptor_) {
                // Closing the acceptor breaks the blocking accept() in the
                // server thread during shutdown.
                beast::error_code ec;
                acceptor_->cancel(ec);
                acceptor_->close(ec);
            }
        }

        if (server_thread_.joinable()) {
            server_thread_.join();
        }

        std::vector<std::shared_ptr<Session>> sessions_to_close;
        {
            std::lock_guard<std::mutex> lock(live_sessions_mutex_);
            sessions_to_close = live_sessions_;
        }
        // Force sockets closed before joining session threads so blocked read()
        // calls wake promptly.
        for (const auto& session : sessions_to_close) {
            if (session) {
                session->force_close();
            }
        }

        std::vector<std::thread> threads_to_join;
        {
            std::lock_guard<std::mutex> lock(client_threads_mutex_);
            threads_to_join.swap(client_threads_);
        }
        for (auto& t : threads_to_join) {
            if (t.joinable()) {
                t.join();
            }
        }
        {
            std::lock_guard<std::mutex> lock(live_sessions_mutex_);
            live_sessions_.clear();
        }
    }

    void send_transcript_to_session(const std::string& session_id,
                                    const std::string& text,
                                    const std::optional<std::string>& timestamp) {
        if (!send_transcripts_ || session_id.empty()) {
            return;
        }

        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(session_map_mutex_);
            // The map holds weak_ptrs so finished session threads are not kept
            // alive solely because a transcript completes late.
            auto it = sessions_.find(session_id);
            if (it == sessions_.end()) {
                return;
            }
            session = it->second.lock();
        }
        if (!session) {
            return;
        }

        std::string payload =
            std::string("{\"type\":\"transcript\",\"sessionId\":\"") +
            json_escape(session_id) +
            "\"";
        const bool include_timestamp = timestamp.has_value() &&
            (include_transcript_timestamps_ || session->ctx.include_timestamps);
        if (include_timestamp) {
            payload +=
                std::string(",\"timestamp\":\"") +
                json_escape(*timestamp) +
                "\"";
        }
        payload +=
            std::string(",\"text\":\"") +
            json_escape(text) +
            "\"}";

        session->send_text(payload);
    }

private:
    // Format helpers accept a few aliases from mobile/platform clients while
    // preserving a single normalized PCM16 path after decode.
    static std::string to_ascii_lower(std::string s) {
        for (char& c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return s;
    }

    static bool ws_format_is_float32(const std::string& format) {
        const std::string f = to_ascii_lower(format);
        return f == "pcm_f32le" || f == "f32le" || f == "float32" ||
               f == "float32le" || f == "pcm_float32le";
    }

    static bool ws_format_is_s16(const std::string& format) {
        const std::string f = to_ascii_lower(format);
        return f.empty() || f == "pcm_s16le" || f == "s16le" || f == "pcm16";
    }

    static bool ws_format_is_supported(const std::string& format) {
        return ws_format_is_s16(format) || ws_format_is_float32(format);
    }

    static size_t ws_bytes_per_sample(const ClientContext& ctx) {
        return ws_format_is_float32(ctx.format) ? sizeof(float) : sizeof(int16_t);
    }

    static std::vector<int16_t> decode_ws_payload(const uint8_t* data,
                                                  size_t byte_count,
                                                  const ClientContext& ctx) {
        if (data == nullptr || byte_count == 0) {
            return {};
        }

        if (ws_format_is_float32(ctx.format)) {
            if ((byte_count % sizeof(float)) != 0u) {
                return {};
            }
            const size_t sample_count = byte_count / sizeof(float);
            std::vector<int16_t> out;
            out.reserve(sample_count);
            for (size_t i = 0; i < sample_count; ++i) {
                float v = 0.0f;
                std::memcpy(&v, data + i * sizeof(float), sizeof(float));
                if (!std::isfinite(v)) {
                    v = 0.0f;
                }
                // Clamp client floats to valid PCM range before scaling.
                v = std::max(-1.0f, std::min(1.0f, v));
                out.push_back(static_cast<int16_t>(std::lround(v * 32767.0f)));
            }
            return out;
        }

        if ((byte_count % sizeof(int16_t)) != 0u) {
            return {};
        }
        const size_t sample_count = byte_count / sizeof(int16_t);
        std::vector<int16_t> out(sample_count);
        for (size_t i = 0; i < sample_count; ++i) {
            // The wire format is little-endian regardless of host endianness.
            const uint16_t lo = static_cast<uint16_t>(data[i * 2]);
            const uint16_t hi = static_cast<uint16_t>(data[i * 2 + 1]);
            out[i] = static_cast<int16_t>((hi << 8) | lo);
        }
        return out;
    }

    static std::vector<int16_t> normalize_ws_audio_to_mono_16k(const std::vector<int16_t>& in,
                                                                int source_sample_rate,
                                                                int source_channels) {
        if (in.empty()) {
            return {};
        }
        if (source_sample_rate <= 0) {
            source_sample_rate = Constants::SAMPLE_RATE;
        }
        if (source_channels <= 0) {
            source_channels = 1;
        }

        std::vector<int16_t> mono;
        if (source_channels == 1) {
            mono = in;
        } else {
            // Downmix by arithmetic mean; this is adequate for speech PCM and
            // avoids adding a DSP dependency for the WebSocket path.
            const size_t frame_count = in.size() / static_cast<size_t>(source_channels);
            mono.resize(frame_count);
            for (size_t i = 0; i < frame_count; ++i) {
                int32_t sum = 0;
                for (int ch = 0; ch < source_channels; ++ch) {
                    sum += in[i * static_cast<size_t>(source_channels) + static_cast<size_t>(ch)];
                }
                mono[i] = static_cast<int16_t>(sum / source_channels);
            }
        }

        if (mono.empty() || source_sample_rate == Constants::SAMPLE_RATE) {
            return mono;
        }

        const double ratio =
            static_cast<double>(Constants::SAMPLE_RATE) / static_cast<double>(source_sample_rate);
        size_t out_count = static_cast<size_t>(std::llround(static_cast<double>(mono.size()) * ratio));
        out_count = std::max<size_t>(out_count, 1);
        std::vector<int16_t> out(out_count);
        for (size_t i = 0; i < out_count; ++i) {
            // Linear interpolation keeps latency low and is sufficient because
            // clients are expected to send speech, not high-fidelity audio.
            const double src_pos =
                static_cast<double>(i) * static_cast<double>(source_sample_rate) /
                static_cast<double>(Constants::SAMPLE_RATE);
            const size_t i0 = std::min(static_cast<size_t>(src_pos), mono.size() - 1);
            const size_t i1 = std::min(i0 + 1, mono.size() - 1);
            const double frac = src_pos - static_cast<double>(i0);
            const double a = static_cast<double>(mono[i0]);
            const double b = static_cast<double>(mono[i1]);
            const double interpolated = a + (b - a) * frac;
            out[i] = static_cast<int16_t>(std::llround(interpolated));
        }
        return out;
    }

    static std::string format_ws_url(const std::string& host, int port) {
        // IPv6 literals in URLs must be enclosed in brackets.
        const bool needs_brackets =
            host.find(':') != std::string::npos &&
            !(host.size() >= 2 && host.front() == '[' && host.back() == ']');
        const std::string url_host = needs_brackets ? ("[" + host + "]") : host;
        return "ws://" + url_host + ":" + std::to_string(port) + "/";
    }

    struct Session : public std::enable_shared_from_this<Session> {
        Session(tcp::socket socket, WebSocketPcmServer& owner)
            : ws(std::move(socket)), owner(owner) {}

        websocket::stream<tcp::socket> ws;
        std::mutex write_mutex;
        WebSocketPcmServer& owner;
        ClientContext ctx;

        void run() {
            beast::error_code ec;
            auto timeout = websocket::stream_base::timeout::suggested(beast::role_type::server);
            // Keep connections alive across short client pauses between start/stop cycles.
            timeout.idle_timeout = std::chrono::seconds(owner.idle_timeout_seconds_);
            ws.set_option(timeout);
            ws.accept(ec);
            if (ec) {
                std::cerr << "WebSocket handshake failed: " << ec.message() << std::endl;
                return;
            }

            while (owner.running_.load(std::memory_order_acquire)) {
                beast::flat_buffer buffer;
                ws.read(buffer, ec);
                if (ec) {
                    if (ec == beast::error::timeout) {
                        ec = {};
                        continue;
                    }
                    break;
                }

                if (ws.got_text()) {
                    // Control frames (start/stop) are text.
                    owner.handle_text_message(shared_from_this(), beast::buffers_to_string(buffer.data()));
                } else {
                    // Audio frames are binary PCM16 payloads.
                    owner.handle_binary_message(shared_from_this(), buffer);
                }
            }

            owner.flush_pending_audio(ctx);
            owner.unregister_session(ctx.session_id);
            owner.unregister_live_session(this);
        }

        void send_text(const std::string& payload) {
            std::lock_guard<std::mutex> lock(write_mutex);
            // Beast websocket streams require serialized writes per connection.
            beast::error_code ec;
            ws.text(true);
            ws.write(net::buffer(payload), ec);
        }

        void force_close() {
            beast::error_code ec;
            // Best-effort shutdown; errors are ignored because this is only used
            // during server teardown.
            beast::get_lowest_layer(ws).cancel(ec);
            beast::get_lowest_layer(ws).shutdown(tcp::socket::shutdown_both, ec);
            beast::get_lowest_layer(ws).close(ec);
        }
    };

    void register_session(const std::string& session_id, const std::shared_ptr<Session>& session) {
        if (session_id.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(session_map_mutex_);
        sessions_[session_id] = session;
    }

    void unregister_session(const std::string& session_id) {
        if (session_id.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(session_map_mutex_);
        sessions_.erase(session_id);
    }

    void register_live_session(const std::shared_ptr<Session>& session) {
        if (!session) {
            return;
        }
        std::lock_guard<std::mutex> lock(live_sessions_mutex_);
        live_sessions_.push_back(session);
    }

    void unregister_live_session(const Session* session_ptr) {
        if (session_ptr == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(live_sessions_mutex_);
        live_sessions_.erase(
            std::remove_if(
                live_sessions_.begin(),
                live_sessions_.end(),
                [&](const std::shared_ptr<Session>& s) { return s.get() == session_ptr; }),
            live_sessions_.end());
    }

    void handle_text_message(const std::shared_ptr<Session>& session, const std::string& msg) {
        // Protocol is intentionally lightweight: start/stop messages only.
        if (json_message_type_is(msg, "start")) {
            session->ctx.started = true;
            session->ctx.session_id = extract_json_string_field(msg, "sessionId");
            session->ctx.language = default_language_.empty() ? "en" : default_language_;
            const std::string language = extract_json_string_field(msg, "language");
            if (!language.empty()) {
                try {
                    // Normalize client-provided language before queueing audio
                    // so Whisper sees the same values as CLI transcription.
                    session->ctx.language = normalize_whisper_language(language);
                } catch (const std::exception& e) {
                    session->ctx.started = false;
                    session->send_text(
                        std::string("{\"type\":\"error\",\"message\":\"") +
                        json_escape(e.what()) + "\"}");
                    return;
                }
            }
            session->ctx.sample_rate = extract_json_int_field(msg, "sampleRate", Constants::SAMPLE_RATE);
            session->ctx.channels = extract_json_int_field(msg, "channels", 1);
            session->ctx.bits_per_sample = extract_json_int_field(msg, "bitsPerSample", 16);
            session->ctx.format = extract_json_string_field(msg, "format");
            session->ctx.frame_ms = extract_json_int_field(msg, "frameMs", Constants::MIN_AUDIO_LENGTH_MS);
            session->ctx.include_timestamps =
                extract_json_bool_field(msg, "timestamp",
                    extract_json_bool_field(msg, "timestamps", false));
            if (session->ctx.frame_ms < Constants::MIN_AUDIO_LENGTH_MS) {
                session->ctx.frame_ms = Constants::MIN_AUDIO_LENGTH_MS;
            }
            if (session->ctx.sample_rate <= 0) {
                session->ctx.sample_rate = Constants::SAMPLE_RATE;
            }
            if (session->ctx.channels <= 0) {
                session->ctx.channels = 1;
            }
            if (session->ctx.bits_per_sample <= 0) {
                session->ctx.bits_per_sample = 16;
            }
            if (session->ctx.format.empty()) {
                session->ctx.format = (session->ctx.bits_per_sample == 32) ? "pcm_f32le" : "pcm_s16le";
            }
            if (!ws_format_is_supported(session->ctx.format)) {
                session->ctx.started = false;
                session->send_text(
                    std::string("{\"type\":\"error\",\"message\":\"unsupported format: ") +
                    json_escape(session->ctx.format) +
                    "\"}");
                return;
            }
            // VAD runs on small fixed frames; record_timeout remains the maximum
            // emitted speech chunk duration inside the shared energy VAD.
            session->ctx.target_frame_samples = Constants::FRAMES_PER_BUFFER;
            session->ctx.pending_samples.clear();
            session->ctx.pending_samples.reserve(session->ctx.target_frame_samples * 2);
            // Each session has its own VAD so mobile clients do not share noise
            // floor estimates or phrase state.
            session->ctx.vad.setEnergyThreshold(initial_energy_threshold_);
            session->ctx.vad.setVadPreRollSeconds(vad_pre_roll_seconds_);
            session->ctx.vad.setAdaptiveHangoverChunks(adaptive_hangover_chunks_);
            session->ctx.vad.setVerbose(verbose_);
            session->ctx.vad.setAdaptiveEnergyEnabled(adaptive_energy_enabled_);
            session->ctx.vad.configure(Constants::SAMPLE_RATE,
                                       record_timeout_seconds_,
                                       phrase_timeout_seconds_);
            register_session(session->ctx.session_id, session);
            session->send_text(
                std::string("{\"type\":\"ack\",\"sessionId\":\"") +
                json_escape(session->ctx.session_id) + "\"}");
            return;
        }

        if (json_message_type_is(msg, "stop")) {
            flush_pending_audio(session->ctx);
            unregister_session(session->ctx.session_id);
            session->ctx.started = false;
            session->send_text("{\"type\":\"stopped\"}");
            return;
        }
    }

    void handle_binary_message(const std::shared_ptr<Session>& session,
                               beast::flat_buffer& buffer) {
        if (!session->ctx.started) {
            session->send_text("{\"type\":\"error\",\"message\":\"send start message first\"}");
            return;
        }

        const size_t byte_count = buffer.size();
        const size_t bytes_per_sample = ws_bytes_per_sample(session->ctx);
        const size_t max_frame_bytes = std::max(
            Constants::MAX_WS_FRAME_BYTES,
            static_cast<size_t>(std::max(session->ctx.sample_rate, Constants::SAMPLE_RATE)) *
                static_cast<size_t>(std::max(session->ctx.channels, 1)) *
                bytes_per_sample);
        if (byte_count == 0 || byte_count > max_frame_bytes) {
            // Bound frame size to protect memory and latency under malformed input.
            session->send_text("{\"type\":\"error\",\"message\":\"invalid frame size\"}");
            return;
        }
        if ((byte_count % bytes_per_sample) != 0u) {
            session->send_text("{\"type\":\"error\",\"message\":\"binary frame size not aligned with format\"}");
            return;
        }

        std::vector<uint8_t> raw(byte_count);
        net::buffer_copy(net::buffer(raw.data(), byte_count), buffer.data());
        std::vector<int16_t> samples = decode_ws_payload(raw.data(), byte_count, session->ctx);
        if (samples.empty()) {
            session->send_text("{\"type\":\"error\",\"message\":\"failed to decode audio payload\"}");
            return;
        }
        if (!callback_) {
            return;
        }
        std::vector<int16_t> normalized = normalize_ws_audio_to_mono_16k(samples,
                                                                          session->ctx.sample_rate,
                                                                          session->ctx.channels);
        if (normalized.empty()) {
            return;
        }
        session->ctx.pending_samples.insert(session->ctx.pending_samples.end(),
                                            normalized.begin(),
                                            normalized.end());
        const size_t target = session->ctx.target_frame_samples;
        size_t consumed = 0;
        // Consume fixed VAD frames and keep any remainder for the next socket
        // frame, since WebSocket payload boundaries do not imply audio frames.
        while ((session->ctx.pending_samples.size() - consumed) >= target) {
            std::vector<int16_t> chunk(target);
            std::memcpy(chunk.data(),
                        session->ctx.pending_samples.data() + consumed,
                        target * sizeof(int16_t));
            session->ctx.vad.processChunk(
                chunk,
                [&](const std::vector<int16_t>& speech_chunk, int energy_threshold) {
                    callback_(speech_chunk,
                              session->ctx.session_id,
                              energy_threshold,
                              session->ctx.language);
                });
            consumed += target;
        }
        if (consumed > 0) {
            session->ctx.pending_samples.erase(session->ctx.pending_samples.begin(),
                                               session->ctx.pending_samples.begin() +
                                                   static_cast<std::ptrdiff_t>(consumed));
        }
    }

    void flush_pending_audio(ClientContext& ctx) {
        if (!callback_) {
            return;
        }
        if (!ctx.pending_samples.empty()) {
            // On stop/disconnect, push the final partial frame through VAD so
            // the end of an utterance is not stranded in pending_samples.
            std::vector<int16_t> chunk = std::move(ctx.pending_samples);
            ctx.pending_samples.clear();
            ctx.vad.processChunk(
                chunk,
                [&](const std::vector<int16_t>& speech_chunk, int energy_threshold) {
                    callback_(speech_chunk, ctx.session_id, energy_threshold, ctx.language);
                });
        }
        ctx.vad.flush(
            [&](const std::vector<int16_t>& speech_chunk, int energy_threshold) {
                callback_(speech_chunk, ctx.session_id, energy_threshold, ctx.language);
            });
    }

    void server_loop() {
        try {
            net::io_context ioc(1);
            auto endpoint = tcp::endpoint(net::ip::make_address(bind_address_),
                                          static_cast<unsigned short>(port_));

            auto local_acceptor = std::make_unique<tcp::acceptor>(ioc);
            beast::error_code ec;
            local_acceptor->open(endpoint.protocol(), ec);
            if (ec) {
                throw AudioException(std::string("WebSocket open failed: ") + ec.message());
            }
            local_acceptor->set_option(net::socket_base::reuse_address(true), ec);
            local_acceptor->bind(endpoint, ec);
            if (ec) {
                throw AudioException(std::string("WebSocket bind failed: ") + ec.message());
            }
            local_acceptor->listen(net::socket_base::max_listen_connections, ec);
            if (ec) {
                throw AudioException(std::string("WebSocket listen failed: ") + ec.message());
            }

            {
                std::lock_guard<std::mutex> lock(acceptor_mutex_);
                acceptor_ = std::move(local_acceptor);
            }

            std::cout << "WebSocket server listening on " << format_ws_url(bind_address_, port_)
                      << std::endl;

            while (running_.load(std::memory_order_acquire)) {
                tcp::socket socket(ioc);
                beast::error_code accept_ec;
                tcp::acceptor* acceptor_ptr = nullptr;
                {
                    std::lock_guard<std::mutex> lock(acceptor_mutex_);
                    if (!acceptor_) {
                        break;
                    }
                    acceptor_ptr = acceptor_.get();
                }
                if (!acceptor_ptr) {
                    break;
                }
                acceptor_ptr->accept(socket, accept_ec);
                if (accept_ec) {
                    if (running_.load(std::memory_order_acquire)) {
                        std::cerr << "WebSocket accept failed: " << accept_ec.message() << std::endl;
                    }
                    continue;
                }

                auto session = std::make_shared<Session>(std::move(socket), *this);
                register_live_session(session);
                std::lock_guard<std::mutex> lock(client_threads_mutex_);
                // One blocking thread per session keeps control flow straightforward.
                client_threads_.emplace_back([session]() { session->run(); });
            }
        } catch (const std::exception& e) {
            std::cerr << "WebSocket server error: " << e.what() << std::endl;
        }
    }

    std::string bind_address_;
    int port_;
    // Server lifetime and client session bookkeeping. Threads are joined in
    // stop(); weak session map entries allow late transcript delivery attempts
    // to fail safely after disconnect.
    bool send_transcripts_ = true;
    int idle_timeout_seconds_ = Constants::WS_IDLE_TIMEOUT_SECONDS_DEFAULT;
    bool include_transcript_timestamps_ = false;
    double record_timeout_seconds_ = 2.0;
    AudioCallback callback_;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    std::mutex acceptor_mutex_;
    std::unique_ptr<tcp::acceptor> acceptor_;
    std::mutex client_threads_mutex_;
    std::vector<std::thread> client_threads_;
    std::mutex live_sessions_mutex_;
    std::vector<std::shared_ptr<Session>> live_sessions_;
    std::mutex session_map_mutex_;
    std::map<std::string, std::weak_ptr<Session>> sessions_;
    std::string default_language_ = "en";
    int initial_energy_threshold_ = Constants::DEFAULT_ENERGY_THRESHOLD;
    bool adaptive_energy_enabled_ = false;
    double vad_pre_roll_seconds_ = Constants::VAD_PRE_ROLL_SECONDS;
    int adaptive_hangover_chunks_ = Constants::ADAPTIVE_HANGOVER_CHUNKS;
    bool verbose_ = false;
    double phrase_timeout_seconds_ = 3.0;
};

Args parse_arguments(int argc, char* argv[]) {
    Args args;
    // Maintain an allowlist so misspelled options fail loudly instead of being
    // treated as positional values for a previous flag.
    const std::unordered_set<std::string> valid_args = {
        "--energy_threshold", "--record_timeout", "--phrase_timeout", "--language",
        "--translate", "--pipe", "--timestamp", "--default_microphone", "--whisper_model_path",
        "--help", "-h", "--list_microphones", "--adaptive_energy", "--audio_file",
        "--predefined_start_time", "--vad_pre_roll", "--adaptive_silence_fraction",
        "--adaptive_hangover_chunks", "--verbose",
        "--input_source", "--websocket_server", "--websocket_bind", "--websocket_port",
        "--websocket_send_transcripts", "--websocket_idle_timeout"
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (valid_args.find(arg) == valid_args.end()) {
            // Fail fast on typos to avoid silently ignored options.
            std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
            std::exit(1);
        }
        if (arg == "--energy_threshold" && i + 1 < argc) {
            try {
                args.energy_threshold = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid energy threshold value" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--record_timeout" && i + 1 < argc) {
            try {
                args.record_timeout = std::stod(argv[++i]);
                if (args.record_timeout <= 0.0) {
                    throw std::runtime_error("invalid");
                }
            } catch (...) {
                std::cerr << "Error: record_timeout must be a positive number" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--phrase_timeout" && i + 1 < argc) {
            try {
                args.phrase_timeout = std::stod(argv[++i]);
                if (args.phrase_timeout <= 0.0) {
                    throw std::runtime_error("invalid");
                }
            } catch (...) {
                std::cerr << "Error: phrase_timeout must be a positive number" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--language" && i + 1 < argc) {
            try {
                args.language = normalize_whisper_language(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                std::exit(1);
            }
        } else if (arg == "--translate") {
            args.translate = true;
        } else if (arg == "--pipe") {
            args.pipe = true;
        } else if (arg == "--timestamp") {
            args.timestamp = true;
        } else if (arg == "--default_microphone" && i + 1 < argc) {
            args.default_microphone = argv[++i];
        } else if (arg == "--whisper_model_path" && i + 1 < argc) {
            args.whisper_model_path = argv[++i];
        } else if (arg == "--list_microphones") {
            args.list_microphones = true;
        } else if (arg == "--adaptive_energy") {
            args.adaptive_energy = true;
        } else if (arg == "--audio_file" && i + 1 < argc) {
            args.audio_file_path = argv[++i];
        } else if (arg == "--vad_pre_roll" && i + 1 < argc) {
            try {
                args.vad_pre_roll = std::stod(argv[++i]);
                if (args.vad_pre_roll < 0.0 || args.vad_pre_roll > 2.0) {
                    throw std::runtime_error("invalid");
                }
            } catch (...) {
                std::cerr << "Error: vad_pre_roll must be in [0.0, 2.0]" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--adaptive_silence_fraction" && i + 1 < argc) {
            try {
                args.adaptive_silence_fraction = std::stod(argv[++i]);
                if (args.adaptive_silence_fraction <= 0.0 || args.adaptive_silence_fraction >= 1.0) {
                    throw std::runtime_error("invalid");
                }
            } catch (...) {
                std::cerr << "Error: adaptive_silence_fraction must be in (0.0, 1.0)" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--adaptive_hangover_chunks" && i + 1 < argc) {
            try {
                args.adaptive_hangover_chunks = std::stoi(argv[++i]);
                if (args.adaptive_hangover_chunks < 1) {
                    throw std::runtime_error("invalid");
                }
            } catch (...) {
                std::cerr << "Error: adaptive_hangover_chunks must be a positive integer" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--input_source" && i + 1 < argc) {
            args.input_source = argv[++i];
            // Keep source modes closed-ended because each mode owns a different
            // capture lifecycle and resource cleanup path.
            if (args.input_source != "microphone" &&
                args.input_source != "file" &&
                args.input_source != "websocket") {
                std::cerr << "Error: --input_source must be one of: microphone, file, websocket" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--websocket_server") {
            args.websocket_server = true;
        } else if (arg == "--websocket_bind" && i + 1 < argc) {
            args.websocket_bind = argv[++i];
        } else if (arg == "--websocket_port" && i + 1 < argc) {
            try {
                args.websocket_port = std::stoi(argv[++i]);
                if (args.websocket_port <= 0 || args.websocket_port > 65535) {
                    throw std::runtime_error("invalid");
                }
            } catch (...) {
                std::cerr << "Error: websocket_port must be an integer in 1..65535" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--websocket_send_transcripts") {
            args.websocket_send_transcripts = true;
        } else if (arg == "--websocket_idle_timeout" && i + 1 < argc) {
            try {
                args.websocket_idle_timeout = std::stoi(argv[++i]);
                if (args.websocket_idle_timeout < 1 || args.websocket_idle_timeout > 3600) {
                    throw std::runtime_error("invalid");
                }
            } catch (...) {
                std::cerr << "Error: websocket_idle_timeout must be an integer in 1..3600 seconds" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--predefined_start_time" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (!parse_predefined_datetime(value, args.predefined_start_time)) {
                std::cerr << "Error: Invalid --predefined_start_time format. Expected \"YYYY-mm-dd HH:MM:SS\"."
                          << std::endl;
                std::exit(1);
            }
            args.has_predefined_start_time = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << " --energy_threshold <int>          Energy threshold for speech detection. Default: auto-adjust\n"
                << " --adaptive_energy                 Continuously adapt the energy threshold based on silence\n"
                << " --adaptive_silence_fraction <float> Silence RMS fraction used before sending to Whisper. Default: 0.35\n"
                << " --adaptive_hangover_chunks <int>  Consecutive silence chunks before noise-floor update\n"
                << " --vad_pre_roll <float>            Seconds of pre-speech audio kept before VAD trigger. Default: 0.30\n"
                << " --record_timeout <float>          Max duration for audio chunks in seconds. Default: 2.0\n"
                << " --phrase_timeout <float>          Silence duration to end a phrase in seconds. Default: 3.0\n"
                << " --language <lang>                 Whisper language code. Default: en\n"
                << " --translate                       Translate source speech to English instead of transcribing it\n"
                << " --pipe                            Enable pipe mode for continuous streaming\n"
                << " --timestamp                       Print timestamps in pipe mode and all WebSocket transcripts\n"
                << " --whisper_model_path <path>       REQUIRED: Path to the ggml Whisper model\n"
                << " --list_microphones                List available microphones and exit\n"
                << " --audio_file <path>               Transcribe a media file. Audio is extracted via ffmpeg if needed\n"
                << " --predefined_start_time \"YYYY-mm-dd HH:MM:SS\" Override transcript start time for all input sources\n"
                << " --verbose                         Print adaptive threshold changes and diagnostics\n"
                << " --input_source <mode>             microphone | file | websocket. Default: microphone\n"
                << " --websocket_server                Enable WebSocket PCM audio server\n"
                << " --websocket_bind <ip>             WebSocket bind address. Default: 0.0.0.0\n"
                << " --websocket_port <port>           WebSocket port. Default: 8080\n"
                << " --websocket_send_transcripts      Send transcript JSON messages back to clients\n"
                << " --websocket_idle_timeout <sec>    Idle timeout before closing socket. Default: 30\n";
#ifdef __linux__
            std::cout << " --default_microphone <name>       Preferred microphone name. Use --list_microphones to inspect devices\n";
#endif
            std::exit(0);
        }
    }

    if (args.input_source == "file" && args.audio_file_path.empty()) {
        std::cerr << "Error: --audio_file is required when --input_source file is used." << std::endl;
        std::exit(1);
    }
    if (args.input_source == "websocket") {
        // Preserve compatibility with the older --websocket_server flag while
        // allowing --input_source websocket to be the primary selector.
        args.websocket_server = true;
    }
    if (args.whisper_model_path.empty() && !args.list_microphones) {
        std::cerr << "Error: --whisper_model_path is required." << std::endl;
        std::exit(1);
    }
    return args;
}

namespace FileAudio {
namespace {
    // File-mode helpers are kept local so live microphone/WebSocket code remains
    // independent from ffmpeg/ffprobe details.
    std::string to_lower_ascii(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    bool is_explicit_raw_pcm_path(const std::string& path) {
        // Allow bypassing container parsing for known raw PCM extensions.
        const std::string ext = to_lower_ascii(std::filesystem::path(path).extension().string());
        return ext == ".raw" || ext == ".pcm" || ext == ".s16" || ext == ".s16le";
    }

    uint32_t read_u32_le(std::ifstream& f) {
        std::array<unsigned char, 4> b{};
        f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
        if (!f) {
            throw AudioException("Unexpected end of file while reading 32-bit value.");
        }
        return static_cast<uint32_t>(b[0]) |
               (static_cast<uint32_t>(b[1]) << 8u) |
               (static_cast<uint32_t>(b[2]) << 16u) |
               (static_cast<uint32_t>(b[3]) << 24u);
    }

    uint16_t read_u16_le(std::ifstream& f) {
        std::array<unsigned char, 2> b{};
        f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
        if (!f) {
            throw AudioException("Unexpected end of file while reading 16-bit value.");
        }
        return static_cast<uint16_t>(b[0]) |
               static_cast<uint16_t>(static_cast<uint16_t>(b[1]) << 8u);
    }

#ifdef _WIN32
    std::string quote_windows_arg(const std::string& arg) {
        std::string out = "\"";
        for (char c : arg) {
            if (c == '"') {
                out += "\\\"";
            } else {
                out += c;
            }
        }
        out += "\"";
        return out;
    }
#endif

    void run_ffmpeg_extract(const std::string& input, const std::string& output, int target_sample_rate) {
        // Emit a temporary mono 16 kHz WAV because the parser below only accepts
        // the exact PCM format Whisper receives downstream.
#ifdef _WIN32
        const std::string cmd =
            "ffmpeg -y -hide_banner -loglevel error -i " + quote_windows_arg(input) +
            " -ac 1 -ar " + std::to_string(target_sample_rate) +
            " -f wav " + quote_windows_arg(output);
#else
        const std::string cmd =
            "ffmpeg -y -hide_banner -loglevel error -i \"" + input +
            "\" -ac 1 -ar " + std::to_string(target_sample_rate) +
            " -f wav \"" + output + "\"";
#endif
        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            throw AudioException("ffmpeg failed while transcoding media to WAV.");
        }
    }
}

bool load_wav_mono_16(const std::string& path, int& sample_rate_out, std::vector<int16_t>& samples_out) {
    // Minimal RIFF/WAV parser for PCM16 mono. Complex containers are handled by
    // transcoding first, so this function can stay strict.
    sample_rate_out = 0;
    samples_out.clear();

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    std::array<char, 4> riff{};
    std::array<char, 4> wave{};
    f.read(riff.data(), static_cast<std::streamsize>(riff.size()));
    const uint32_t riff_size = read_u32_le(f);
    (void)riff_size;
    f.read(wave.data(), static_cast<std::streamsize>(wave.size()));
    if (!f || std::strncmp(riff.data(), "RIFF", 4) != 0 || std::strncmp(wave.data(), "WAVE", 4) != 0) {
        return false;
    }

    bool fmt_found = false;
    bool data_found = false;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint16_t bits_per_sample = 0;
    uint32_t sample_rate = 0;
    std::vector<int16_t> parsed_samples;

    while (f && !(fmt_found && data_found)) {
        std::array<char, 4> chunk_id{};
        f.read(chunk_id.data(), static_cast<std::streamsize>(chunk_id.size()));
        if (!f) {
            break;
        }

        const uint32_t chunk_size = read_u32_le(f);
        if (std::strncmp(chunk_id.data(), "fmt ", 4) == 0) {
            audio_format = read_u16_le(f);
            channels = read_u16_le(f);
            sample_rate = read_u32_le(f);
            (void)read_u32_le(f);
            (void)read_u16_le(f);
            bits_per_sample = read_u16_le(f);
            if (chunk_size > 16u) {
                f.seekg(static_cast<std::streamoff>(chunk_size - 16u), std::ios::cur);
            }
            fmt_found = f.good();
        } else if (std::strncmp(chunk_id.data(), "data", 4) == 0) {
            const size_t byte_count = static_cast<size_t>(chunk_size);
            if ((byte_count % sizeof(int16_t)) != 0u) {
                return false;
            }
            parsed_samples.resize(byte_count / sizeof(int16_t));
            if (!parsed_samples.empty()) {
                f.read(reinterpret_cast<char*>(parsed_samples.data()),
                       static_cast<std::streamsize>(byte_count));
                if (!f) {
                    return false;
                }
            }
            data_found = true;
        } else {
            // Skip unknown chunks (LIST/JUNK/etc.) per RIFF spec.
            f.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }

        if ((chunk_size % 2u) != 0u) {
            // RIFF chunks are word-aligned; odd-sized chunks carry one pad byte.
            f.seekg(1, std::ios::cur);
        }
    }

    if (!fmt_found || !data_found) {
        return false;
    }
    if (audio_format != 1u || channels != 1u || bits_per_sample != 16u) {
        return false;
    }

    sample_rate_out = static_cast<int>(sample_rate);
    samples_out = std::move(parsed_samples);
    return true;
}

std::vector<int16_t> load_raw_pcm_16(const std::string& path) {
    // Raw mode assumes signed little-endian PCM16 at 16 kHz mono; use only for
    // explicit .raw/.pcm/.s16 inputs.
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return {};
    }
    const std::streamsize size = f.tellg();
    if (size <= 0 || (size % static_cast<std::streamsize>(sizeof(int16_t))) != 0) {
        return {};
    }
    f.seekg(0, std::ios::beg);
    std::vector<int16_t> out(static_cast<size_t>(size) / sizeof(int16_t));
    f.read(reinterpret_cast<char*>(out.data()), size);
    if (!f) {
        return {};
    }
    return out;
}

std::vector<float> to_float(const std::vector<int16_t>& samples) {
    std::vector<float> out;
    out.reserve(samples.size());
    constexpr float kScale = 1.0f / 32768.0f;
    for (int16_t s : samples) {
        out.push_back(static_cast<float>(s) * kScale);
    }
    return out;
}

bool parse_metadata_datetime(std::string value,
                             std::chrono::system_clock::time_point& out) {
    // ffprobe may return ISO timestamps with T/Z, fractions, or offsets. Strip
    // those decorations before using the platform local-time converter.
    const auto trim_copy = [](const std::string& input) -> std::string {
        const size_t start = input.find_first_not_of(" \t\n\r\f\v");
        if (start == std::string::npos) {
            return {};
        }
        const size_t end = input.find_last_not_of(" \t\n\r\f\v");
        return input.substr(start, end - start + 1);
    };

    value = trim_copy(value);
    if (value.empty()) {
        return false;
    }

    std::replace(value.begin(), value.end(), 'T', ' ');
    if (!value.empty() && (value.back() == 'Z' || value.back() == 'z')) {
        value.pop_back();
    }

    const size_t frac_pos = value.find('.');
    if (frac_pos != std::string::npos) {
        value.erase(frac_pos);
    }

    const size_t tz_pos = value.find_first_of("+-", 19);
    if (tz_pos != std::string::npos) {
        // Preserve historical behavior: ignore explicit offsets after seconds
        // and let mktime interpret the normalized timestamp locally.
        value.erase(tz_pos);
    }

    value = trim_copy(value);
    if (value.size() < 19) {
        return false;
    }

    value = value.substr(0, 19);
    std::tm tm{};
    std::istringstream ss(value);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) {
        return false;
    }

    const std::time_t tt = std::mktime(&tm);
    if (tt == static_cast<std::time_t>(-1)) {
        return false;
    }

    out = std::chrono::system_clock::from_time_t(tt);
    return true;
}

std::optional<std::chrono::system_clock::time_point>
probe_file_encoded_timeline_start(const std::string& media_path) {
    // Prefer ffprobe's realtime timeline when present (microseconds since epoch).
    const auto trim_copy = [](const std::string& input) -> std::string {
        const size_t start = input.find_first_not_of(" \t\n\r\f\v");
        if (start == std::string::npos) {
            return {};
        }
        const size_t end = input.find_last_not_of(" \t\n\r\f\v");
        return input.substr(start, end - start + 1);
    };

#ifdef _WIN32
    const std::string cmd =
        "ffprobe -v error -show_entries format=start_time_realtime:stream=start_time_realtime "
        "-of default=noprint_wrappers=1:nokey=1 " +
        quote_windows_arg(media_path) + " 2>nul";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    const std::string cmd =
        "ffprobe -v error -show_entries format=start_time_realtime:stream=start_time_realtime "
        "-of default=noprint_wrappers=1:nokey=1 \"" + media_path + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (pipe == nullptr) {
        return std::nullopt;
    }

    std::string output;
    char buffer[256];
    while (std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr) {
        output += buffer;
    }

#ifdef _WIN32
    const int rc = _pclose(pipe);
#else
    const int rc = pclose(pipe);
#endif
    if (rc != 0 || output.empty()) {
        return std::nullopt;
    }

    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        line = trim_copy(line);
        if (line.empty()) {
            continue;
        }
        try {
            // start_time_realtime is microseconds since Unix epoch.
            const long long micros = std::stoll(line);
            if (micros <= 0) {
                continue;
            }
            const auto tp = std::chrono::system_clock::time_point{
                std::chrono::microseconds(micros)};
            return tp;
        } catch (...) {
        }
    }
    return std::nullopt;
}

std::optional<std::chrono::system_clock::time_point>
probe_file_encoded_start_time(const std::string& media_path) {
    // Fallback to creation_time metadata when timeline start is unavailable.
#ifdef _WIN32
    const std::string cmd =
        "ffprobe -v error -show_entries "
        "format_tags=creation_time:stream_tags=creation_time "
        "-of default=noprint_wrappers=1:nokey=1 " +
        quote_windows_arg(media_path) + " 2>nul";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    const std::string cmd =
        "ffprobe -v error -show_entries "
        "format_tags=creation_time:stream_tags=creation_time "
        "-of default=noprint_wrappers=1:nokey=1 \"" + media_path + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (pipe == nullptr) {
        return std::nullopt;
    }

    std::string output;
    char buffer[256];
    while (std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr) {
        output += buffer;
    }

#ifdef _WIN32
    const int rc = _pclose(pipe);
#else
    const int rc = pclose(pipe);
#endif
    if (rc != 0 || output.empty()) {
        return std::nullopt;
    }

    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        std::chrono::system_clock::time_point parsed{};
        if (parse_metadata_datetime(line, parsed)) {
            return parsed;
        }
    }
    return std::nullopt;
}

std::chrono::system_clock::time_point resolve_file_start_time(
    const Args& args,
    const std::chrono::system_clock::time_point& application_start_time) {
    // Priority: explicit CLI override > encoded timeline > metadata tag > app start.
    if (args.has_predefined_start_time) {
        return args.predefined_start_time;
    }
    if (!args.audio_file_path.empty()) {
        const auto timeline = probe_file_encoded_timeline_start(args.audio_file_path);
        if (timeline.has_value()) {
            return *timeline;
        }
        const auto probed = probe_file_encoded_start_time(args.audio_file_path);
        if (probed.has_value()) {
            return *probed;
        }
    }
    return application_start_time;
}

std::string transcode_media_to_wav(const std::string& media_path, int target_sample_rate) {
    // Include PID and wall-clock milliseconds to avoid collisions when multiple
    // transcriptions run from the same working directory.
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
#ifdef _WIN32
    const auto pid = static_cast<long long>(GetCurrentProcessId());
#else
    const auto pid = static_cast<long long>(getpid());
#endif
    const auto tmp_path =
        std::filesystem::temp_directory_path() /
        ("stt_" + std::to_string(pid) + "_" + std::to_string(now_ms) + ".wav");
    run_ffmpeg_extract(media_path, tmp_path.string(), target_sample_rate);
    return tmp_path.string();
}
} // namespace FileAudio

void clear_console() {
    // Interactive display is terminal-only; pipe mode avoids clearing output.
#ifdef _WIN32
    std::system("cls");
#else
    std::cout << "\033[2J\033[H";
#endif
}

std::string trim(const std::string& str) {
    const size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return {};
    }
    const size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

// Whisper emits these tokens when it detects no speech; suppress them so they
// are never written to the transcript or forwarded to WebSocket clients.
bool is_whisper_noise_token(const std::string& text) {
    // Exact-match bracket/paren pseudo-tokens.
    static const std::vector<std::string> kExactTokens = {
        "[BLANK_AUDIO]", "[ Silence]", "[silence]", "(silence)",
        "[Music]", "[ Music]", "(Music)", "(music)", "[music]",
        "[Applause]", "[ Applause]", "(Applause)",
        "[MUSIC]", "[APPLAUSE]",
    };
    for (const auto& tok : kExactTokens) {
        if (text == tok) {
            return true;
        }
    }

    // Normalize: lowercase + strip trailing punctuation/whitespace, then check
    // against known hallucination phrases that Whisper emits on silence/noise.
    std::string norm = text;
    while (!norm.empty() &&
           (std::ispunct(static_cast<unsigned char>(norm.back())) ||
            std::isspace(static_cast<unsigned char>(norm.back())))) {
        norm.pop_back();
    }
    std::transform(norm.begin(), norm.end(), norm.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    static const std::vector<std::string> kHallucinationPhrases = {
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "thank you very much",
        "thank you so much",
        "thanks for listening",
        "thank you for listening",
        "thanks",
        "you",                  // single-token noise on some models
        "bye",
        "bye bye",
        "goodbye",
        // Large-model hallucinations on silence/background noise
        "i believe in the lord",
        "i believe in god",
        "and the door",
        "the door",
        "subscribe",
        "subtitles by",
        "subtitled by",
        "transcribed by",
        "www",
    };
    for (const auto& phrase : kHallucinationPhrases) {
        if (norm == phrase) {
            return true;
        }
    }
    return false;
}

std::string format_datetime(const std::chrono::time_point<std::chrono::system_clock>& tp) {
    const std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "[%Y-%m-%d %H:%M:%S]");
    return oss.str();
}

void list_and_exit() {
    try {
        const auto microphones = AudioRecorder::listMicrophoneNames();
        if (microphones.empty()) {
            std::cout << "No microphones found." << std::endl;
        } else {
            std::cout << "Available microphones:" << std::endl;
            for (size_t i = 0; i < microphones.size(); ++i) {
                std::cout << " " << (i + 1) << ". " << microphones[i] << std::endl;
            }
        }
        std::exit(0);
    } catch (const std::exception& e) {
        std::cerr << "Error listing microphones: " << e.what() << std::endl;
        std::exit(1);
    }
}

bool is_silent_chunk(const std::vector<int16_t>& samples,
                     int energy_threshold,
                     long double threshold_fraction) {
    // Secondary silence gate after VAD. This filters very quiet fragments before
    // spending Whisper time on them.
    if (samples.empty()) {
        return true;
    }
    long double sum_squares = 0.0L;
    for (int16_t s : samples) {
        const long double v = static_cast<long double>(s);
        sum_squares += v * v;
    }
    const long double rms = std::sqrt(sum_squares / static_cast<long double>(samples.size()));
    const long double min_rms = static_cast<long double>(energy_threshold) * threshold_fraction;
    return rms < min_rms;
}

std::vector<float> load_audio_from_file(const std::string& path) {
    // Normalize all file inputs into the same float PCM format used by live
    // capture so WhisperModel has only one audio representation to handle.
    int sample_rate = 0;
    std::vector<int16_t> pcm;
    if (FileAudio::load_wav_mono_16(path, sample_rate, pcm)) {
        if (sample_rate != Constants::SAMPLE_RATE) {
            throw AudioException("Unsupported WAV sample rate. Expected 16000 Hz mono 16-bit PCM.");
        }
    } else {
        if (FileAudio::is_explicit_raw_pcm_path(path)) {
            pcm = FileAudio::load_raw_pcm_16(path);
            sample_rate = Constants::SAMPLE_RATE;
            if (pcm.empty()) {
                throw AudioException("Failed to decode raw PCM16 file.");
            }
        } else {
            // Generic media path: transcode to WAV first, then parse PCM16.
            const std::string tmp_wav = FileAudio::transcode_media_to_wav(path, Constants::SAMPLE_RATE);
            const bool ok = FileAudio::load_wav_mono_16(tmp_wav, sample_rate, pcm);
            std::error_code ec;
            std::filesystem::remove(tmp_wav, ec);
            if (!ok || sample_rate != Constants::SAMPLE_RATE) {
                // Last-resort fallback for unconventional but already-PCM payloads.
                pcm = FileAudio::load_raw_pcm_16(path);
                sample_rate = Constants::SAMPLE_RATE;
            }
            if (pcm.empty()) {
                throw AudioException("Failed to decode media file. Ensure ffmpeg is installed and input is valid.");
            }
        }
    }
    if (pcm.size() < Constants::MIN_AUDIO_SAMPLES) {
        pcm.resize(Constants::MIN_AUDIO_SAMPLES, 0);
    }
    return FileAudio::to_float(pcm);
}

std::atomic<bool> g_quit{false};

void on_sigint(int) {
    // Keep the signal handler async-signal-safe: only store an atomic flag.
    g_quit.store(true, std::memory_order_release);
}

struct QueuedAudioChunk {
    // Shared work item for microphone and WebSocket capture paths.
    std::vector<int16_t> samples;
    std::string session_id;
    std::chrono::system_clock::time_point submitted;
    int energy_threshold = Constants::DEFAULT_ENERGY_THRESHOLD;
    std::string language;
    bool translate = false;
};

int main(int argc, char* argv[]) {
    try {
        std::signal(SIGINT, on_sigint);
        std::cerr << "[ServerBuild] exe=" << (argc > 0 && argv[0] ? argv[0] : "(unknown)")
                  << " fingerprint=" << Constants::BUILD_FINGERPRINT << std::endl;
        const auto application_start_time = std::chrono::system_clock::now();
        Args args = parse_arguments(argc, argv);

        if (args.list_microphones) {
            list_and_exit();
        }

        const auto live_transcript_start_time =
            args.has_predefined_start_time ? args.predefined_start_time : application_start_time;
        auto live_transcript_timestamp_for =
            [&](const std::chrono::system_clock::time_point& captured_at) {
                // Preserve relative capture timing while shifting the whole live
                // transcript onto a caller-provided absolute start time.
                return live_transcript_start_time +
                       std::chrono::duration_cast<std::chrono::system_clock::duration>(
                           captured_at - application_start_time);
            };

        if (args.input_source == "file") {
            // Batch mode: deterministic chunking over already available media.
            WhisperModel audio_model(args.whisper_model_path);
            // Validate before decoding the whole file so configuration errors
            // fail quickly and consistently with live modes.
            audio_model.validate_language(args.language, args.translate);
            auto audio_data = load_audio_from_file(args.audio_file_path);
            std::cout << "Transcribing media file: " << args.audio_file_path << std::endl;
            size_t chunk_samples = static_cast<size_t>(Constants::SAMPLE_RATE * args.record_timeout);
            chunk_samples = std::max(chunk_samples, Constants::MIN_AUDIO_SAMPLES);
            const auto transcript_start = FileAudio::resolve_file_start_time(args, application_start_time);
            size_t offset = 0;
            while (offset < audio_data.size()) {
                if (g_quit.load(std::memory_order_acquire)) {
                    break;
                }
                const auto loop_start = std::chrono::steady_clock::now();
                // Chunking mirrors live record_timeout behavior so file and live
                // transcripts have similar segment sizes.
                const size_t end = std::min(offset + chunk_samples, audio_data.size());
                std::vector<float> chunk(audio_data.begin() + static_cast<std::ptrdiff_t>(offset),
                                         audio_data.begin() + static_cast<std::ptrdiff_t>(end));
                if (chunk.size() < Constants::MIN_AUDIO_SAMPLES) {
                    chunk.resize(Constants::MIN_AUDIO_SAMPLES, 0.0f);
                }
                const double end_sec = static_cast<double>(end) / Constants::SAMPLE_RATE;
                const auto end_offset = std::chrono::duration_cast<std::chrono::system_clock::duration>(
                    std::chrono::duration<double>(end_sec));
                const auto end_time = transcript_start + end_offset;
                std::string text = trim(audio_model.transcribe(chunk, args.language, args.translate));
                if (!text.empty()) {
                    const std::string timestamp_text = args.has_predefined_start_time
                        ? format_datetime_no_conversion(end_time)
                        : format_datetime(end_time);
                    std::cout << timestamp_text << " " << text << std::endl;
                    std::cout.flush();
                }
                offset = end;
                const auto elapsed = std::chrono::steady_clock::now() - loop_start;
                const auto target = std::chrono::duration<double>(args.record_timeout);
                if (elapsed < target) {
                    // Pace file mode like live capture; this keeps downstream
                    // consumers from receiving an entire file burst at once.
                    std::this_thread::sleep_for(target - elapsed);
                }
            }
            PortAudioRuntime::terminate_if_initialized();
            return 0;
        }

        std::chrono::time_point<std::chrono::system_clock> last_phrase_end_time{};
        bool phrase_time_set = false;
        // Single queue decouples capture ingress from transcription throughput.
        std::queue<QueuedAudioChunk> data_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;

        std::unique_ptr<PortAudioRecorder> recorder;
        std::unique_ptr<WebSocketPcmServer> websocket_server;

        WhisperModel audio_model(args.whisper_model_path);
        audio_model.set_verbose(args.verbose);
        // Validate the CLI default language before opening microphone or
        // WebSocket capture. Per-session WebSocket languages are checked later.
        audio_model.validate_language(args.language, args.translate);
        AudioTranscriber transcriber(audio_model, args.language, args.translate);
        std::vector<std::string> transcription{""};

        if (args.input_source == "microphone") {
            recorder = std::make_unique<PortAudioRecorder>();
            recorder->setPreferredDeviceName(args.default_microphone);
            if (args.energy_threshold == -1) {
                std::cout << "Calibrating microphone..." << std::endl;
                recorder->adjustForAmbientNoise(args.energy_threshold);
            } else {
                recorder->setEnergyThreshold(args.energy_threshold);
                std::cout << "Using energy threshold: " << args.energy_threshold << std::endl;
            }
            recorder->setAdaptiveEnergyEnabled(args.adaptive_energy);
            recorder->setVadPreRollSeconds(args.vad_pre_roll);
            recorder->setAdaptiveHangoverChunks(args.adaptive_hangover_chunks);
            recorder->setVerbose(args.verbose);
            if (args.adaptive_energy) {
                std::cout << "Adaptive energy threshold enabled (EMA + hangover="
                          << args.adaptive_hangover_chunks << " chunks)." << std::endl;
                std::cout << "Adaptive silence fraction: " << args.adaptive_silence_fraction << std::endl;
            }
            std::cout << "VAD pre-roll seconds: " << args.vad_pre_roll << std::endl;

            auto record_callback = [&](const std::vector<int16_t>& audio_chunk) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (data_queue.size() >= Constants::MAX_QUEUED_AUDIO_CHUNKS) {
                    // Back-pressure policy: bounded memory, drop oldest pending audio.
                    data_queue.pop();
                }
                data_queue.push(QueuedAudioChunk{
                    audio_chunk,
                    "",
                    std::chrono::system_clock::now(),
                    recorder ? recorder->getEnergyThreshold() : Constants::DEFAULT_ENERGY_THRESHOLD,
                    args.language,
                    args.translate});
                queue_cv.notify_one();
            };

            if (!recorder->startRecording(record_callback,
                                          Constants::SAMPLE_RATE,
                                          args.record_timeout,
                                          args.phrase_timeout)) {
                std::cerr << "Failed to start continuous recording." << std::endl;
                PortAudioRuntime::terminate_if_initialized();
                return 1;
            }
            if (!args.pipe) {
                std::cout << "Model loaded and recording started.\n" << std::endl;
            }
        } else if (args.input_source == "websocket") {
            const int websocket_initial_threshold = std::max(
                args.energy_threshold > 0 ? args.energy_threshold : Constants::DEFAULT_ENERGY_THRESHOLD,
                Constants::ADAPTIVE_THRESHOLD_MIN);
            websocket_server = std::make_unique<WebSocketPcmServer>(
                args.websocket_bind,
                args.websocket_port,
                args.websocket_send_transcripts,
                args.websocket_idle_timeout,
                args.timestamp,
                args.language,
                websocket_initial_threshold,
                args.adaptive_energy,
                args.vad_pre_roll,
                args.adaptive_hangover_chunks,
                args.verbose,
                args.phrase_timeout,
                args.record_timeout);

            auto websocket_callback = [&](const std::vector<int16_t>& audio_chunk,
                                          const std::string& session_id,
                                          int energy_threshold,
                                          const std::string& language) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (data_queue.size() >= Constants::MAX_QUEUED_AUDIO_CHUNKS) {
                    // Same back-pressure policy as microphone mode: keep memory
                    // bounded and prefer newest speech when transcription lags.
                    data_queue.pop();
                }
                data_queue.push(QueuedAudioChunk{
                    audio_chunk,
                    session_id,
                    std::chrono::system_clock::now(),
                    std::max(energy_threshold, Constants::ADAPTIVE_THRESHOLD_MIN),
                    language.empty() ? args.language : language,
                    args.translate});
                queue_cv.notify_one();
            };

            if (!websocket_server->start(websocket_callback)) {
                std::cerr << "Failed to start WebSocket audio server." << std::endl;
                PortAudioRuntime::terminate_if_initialized();
                return 1;
            }
            std::cout << "Using WebSocket energy threshold: " << websocket_initial_threshold << std::endl;
            if (args.adaptive_energy) {
                std::cout << "Adaptive energy threshold enabled (EMA + hangover="
                          << args.adaptive_hangover_chunks << " chunks)." << std::endl;
                std::cout << "Adaptive silence fraction: " << args.adaptive_silence_fraction << std::endl;
            }
            std::cout << "VAD pre-roll seconds: " << args.vad_pre_roll << std::endl;
            std::cout << "Model loaded and WebSocket server started.\n" << std::endl;
        } else {
            throw AudioException("Unsupported input_source.");
        }

        struct Pending {
            std::future<std::string> fut;
            std::chrono::system_clock::time_point submitted;
            bool starts_new_phrase = false;
            std::string session_id;
        };
        // Pending futures preserve ordering while allowing async Whisper execution.
        std::deque<Pending> pending;

        while (!g_quit.load(std::memory_order_acquire)) {
            QueuedAudioChunk queued_chunk;
            bool has_chunk = false;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait_for(lock,
                                  std::chrono::milliseconds(Constants::MAIN_LOOP_TIMEOUT_MS),
                                  [&] { return !data_queue.empty() || g_quit.load(); });
                if (!data_queue.empty()) {
                    queued_chunk = std::move(data_queue.front());
                    data_queue.pop();
                    has_chunk = true;
                }
            }

            const auto now = std::chrono::system_clock::now();
            bool phrase_complete = false;
            if (phrase_time_set &&
                (now - last_phrase_end_time) > std::chrono::duration<double>(args.phrase_timeout)) {
                // Mark the next completed transcript as a new displayed phrase.
                phrase_complete = true;
            }

            for (auto it = pending.begin(); it != pending.end();) {
                using namespace std::chrono_literals;
                if (it->fut.wait_for(0ms) == std::future_status::ready) {
                    // Futures are checked in queue order, so output order follows
                    // capture order even if a later job would finish first.
                    std::string text;
                    try {
                        text = trim(it->fut.get());
                    } catch (const std::exception& e) {
                        std::cerr << "Transcription error: " << e.what() << std::endl;
                    }
                    if (!text.empty() && !is_whisper_noise_token(text)) {
                        const auto transcript_timestamp =
                            live_transcript_timestamp_for(it->submitted);
                        const std::string timestamp_text =
                            args.has_predefined_start_time
                                ? format_datetime_no_conversion(transcript_timestamp)
                                : format_datetime(transcript_timestamp);
                        if (args.pipe) {
                            if (args.timestamp) {
                                std::cout << timestamp_text << " " << text << std::endl;
                            } else {
                                std::cout << text << std::endl;
                            }
                        } else {
                            // Interactive mode redraws current transcript state each update.
                            if (it->starts_new_phrase) {
                                transcription.push_back(text);
                            } else {
                                transcription.back() = text;
                            }
                            clear_console();
                            for (const auto& line : transcription) {
                                if (!line.empty()) {
                                    std::cout << line << std::endl;
                                }
                            }
                            std::cout << std::flush;
                        }
                        if (websocket_server && !it->session_id.empty()) {
                            // WebSocket clients receive the same text accepted
                            // for console output after noise filtering.
                            websocket_server->send_transcript_to_session(
                                it->session_id,
                                text,
                                std::optional<std::string>(timestamp_text));
                        }
                    }
                    it = pending.erase(it);
                } else {
                    ++it;
                }
            }

            if (has_chunk && !queued_chunk.samples.empty()) {
                const int current_energy_threshold = std::max(
                    queued_chunk.energy_threshold > 0
                        ? queued_chunk.energy_threshold
                        : (args.energy_threshold > 0 ? args.energy_threshold : Constants::DEFAULT_ENERGY_THRESHOLD),
                    Constants::ADAPTIVE_THRESHOLD_MIN);
                const long double silent_fraction = args.adaptive_energy
                                                        ? static_cast<long double>(args.adaptive_silence_fraction)
                                                        : 0.5L;
                if (args.verbose) {
                    long double ss = 0.0L;
                    int16_t mn = std::numeric_limits<int16_t>::max();
                    int16_t mx = std::numeric_limits<int16_t>::min();
                    for (int16_t s : queued_chunk.samples) {
                        ss += static_cast<long double>(s) * s;
                        mn = std::min(mn, s);
                        mx = std::max(mx, s);
                    }
                    const long double rms = std::sqrt(ss / static_cast<long double>(queued_chunk.samples.size()));
                    std::cerr << "[chunk] samples=" << queued_chunk.samples.size()
                              << " rms=" << std::fixed << std::setprecision(1) << static_cast<double>(rms)
                              << " min=" << mn << " max=" << mx
                              << " threshold=" << current_energy_threshold
                              << std::endl;
                }
                if (is_silent_chunk(queued_chunk.samples, current_energy_threshold, silent_fraction)) {
                    if (args.verbose) {
                        std::cerr << "[chunk] dropped (silent)" << std::endl;
                    }
                    continue;
                }
                last_phrase_end_time = queued_chunk.submitted;
                phrase_time_set = true;
                if (queued_chunk.samples.size() < Constants::MIN_AUDIO_SAMPLES) {
                    queued_chunk.samples.resize(Constants::MIN_AUDIO_SAMPLES, 0);
                }
                std::vector<float> audio_float(queued_chunk.samples.size());
                for (size_t i = 0; i < queued_chunk.samples.size(); ++i) {
                    audio_float[i] = static_cast<float>(queued_chunk.samples[i]) / 32768.0f;
                }
                Pending p;
                // Whisper runs asynchronously so capture can continue while the
                // model is processing the previous utterance.
                p.fut = transcriber.transcribe_async(std::move(audio_float),
                                                     queued_chunk.language,
                                                     queued_chunk.translate);
                p.submitted = queued_chunk.submitted;
                p.starts_new_phrase = phrase_complete;
                p.session_id = queued_chunk.session_id;
                if (p.starts_new_phrase && !args.pipe && !transcription.back().empty()) {
                    transcription.push_back("");
                }
                pending.emplace_back(std::move(p));
            } else {
                if (phrase_time_set &&
                    (now - last_phrase_end_time) >
                        std::chrono::duration<double>(args.phrase_timeout *
                                                      Constants::PHRASE_TIMEOUT_MULTIPLIER)) {
                    // Hard phrase boundary after extended silence in live mode.
                    if (!args.pipe && !transcription.back().empty()) {
                        transcription.push_back("");
                    }
                    phrase_time_set = false;
                }
            }
        }

        if (websocket_server) {
            websocket_server->stop();
        }
        if (recorder) {
            recorder->stopRecording();
        }
        PortAudioRuntime::terminate_if_initialized();
        return 0;
    } catch (const AudioException& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }
    PortAudioRuntime::terminate_if_initialized();
    return 1;
}
