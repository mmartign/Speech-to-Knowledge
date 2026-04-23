// -*- coding: utf-8 -*-
// SPDX-License-Identifier: AGPL-3.0-or-later
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
//   "source": "maui-mobile"
// }
//
// Copyright (C) 2025 Spazio IT

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
    constexpr double ADAPTIVE_UPDATE_MAX_RMS_FRACTION = 0.20;
    constexpr double ADAPTIVE_THRESHOLD_STEP_FRACTION = 0.05;
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
}

class AudioException : public std::runtime_error {
public:
    explicit AudioException(const std::string& message) : std::runtime_error(message) {}
};

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

struct Args {
    int energy_threshold = -1;
    bool adaptive_energy = false;
    double record_timeout = 2.0;
    double phrase_timeout = 3.0;
    std::string language = "en";
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
};

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
        adaptive_hangover_chunks_.store(std::max(1, chunks), std::memory_order_relaxed);
    }

    void setVerbose(bool v) override {
        verbose_ = v;
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
            audio_callback_ = std::move(callback);
        }

        sample_rate_ = sample_rate;
        record_timeout_ = record_timeout;
        phrase_timeout_ = phrase_timeout;
        max_buffer_samples_ = static_cast<size_t>(std::ceil(sample_rate_ * record_timeout_));
        max_silence_chunks_ = static_cast<size_t>(
            std::ceil(phrase_timeout_ * sample_rate_ / static_cast<double>(Constants::FRAMES_PER_BUFFER)));
        max_pre_roll_chunks_ = std::max<size_t>(
            1,
            static_cast<size_t>(std::ceil(
                vad_pre_roll_seconds_.load(std::memory_order_relaxed) * sample_rate_ /
                static_cast<double>(Constants::FRAMES_PER_BUFFER))));
        // Runtime limits are computed once per start() so hot callbacks stay lightweight.

        clear_processing_state();
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
        stream_.stop();
        stream_.close();
        worker_running_.store(false, std::memory_order_release);
        ring_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        if (was_active) {
            clear_processing_state();
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
        if (adaptive_energy_enabled_.load(std::memory_order_relaxed)) {
            prime_noise_floor_estimate(static_cast<double>(rms));
        }
        std::cout << "Adjusted energy threshold to: " << getEnergyThreshold() << std::endl;
    }

    void setEnergyThreshold(int threshold) override {
        threshold = std::max(threshold, 1);
        bool expected = false;
        if (base_threshold_initialized_.compare_exchange_strong(expected,
                                                                true,
                                                                std::memory_order_acq_rel,
                                                                std::memory_order_acquire)) {
            base_energy_threshold_.store(threshold, std::memory_order_relaxed);
        }
        energy_threshold_.store(threshold, std::memory_order_relaxed);
        energy_threshold_squared_.store(static_cast<int64_t>(threshold) * static_cast<int64_t>(threshold),
                                        std::memory_order_relaxed);
    }

    int getEnergyThreshold() const override {
        return energy_threshold_.load(std::memory_order_relaxed);
    }

    void setAdaptiveEnergyEnabled(bool enabled) override {
        adaptive_energy_enabled_.store(enabled, std::memory_order_release);
        if (!enabled) {
            silence_floor_initialized_.store(false, std::memory_order_release);
            return;
        }
        double estimate = static_cast<double>(getEnergyThreshold()) /
                          Constants::ENERGY_THRESHOLD_MULTIPLIER;
        if (estimate <= 0.0) {
            estimate = static_cast<double>(Constants::ADAPTIVE_THRESHOLD_MIN) /
                       Constants::ENERGY_THRESHOLD_MULTIPLIER;
        }
        prime_noise_floor_estimate(estimate);
    }

    void setVadPreRollSeconds(double seconds) override {
        if (seconds < 0.0) {
            seconds = 0.0;
        }
        vad_pre_roll_seconds_.store(seconds, std::memory_order_relaxed);
        max_pre_roll_chunks_ = std::max<size_t>(
            1,
            static_cast<size_t>(std::ceil(
                seconds * sample_rate_ / static_cast<double>(Constants::FRAMES_PER_BUFFER))));
    }

private:
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
            callback(current_chunk);
            return;
        }

        const double sum_squares = calculate_audio_energy(current_chunk.data(), current_chunk.size());
        const double threshold_squared =
            static_cast<double>(energy_threshold_squared_.load(std::memory_order_relaxed));
        const bool adaptive_enabled = adaptive_energy_enabled_.load(std::memory_order_relaxed);
        bool is_speech = false;
        bool should_update_adaptive = false;
        std::vector<int16_t> completed;

        {
            std::lock_guard<std::mutex> lock(vad_mutex_);
            const bool speech_active = !vad_buffer_.empty();
            double speech_trigger = threshold_squared;
            if (adaptive_enabled && !speech_active) {
                // When idle, slightly lower onset threshold to reduce missed speech starts.
                const double ratio = Constants::ADAPTIVE_ONSET_TRIGGER_RATIO;
                speech_trigger *= (ratio * ratio);
            }
            is_speech = sum_squares > (speech_trigger * current_chunk.size());
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
                pre_roll_chunks_.push_back(current_chunk);
                while (pre_roll_chunks_.size() > max_pre_roll_chunks_) {
                    pre_roll_chunks_.pop_front();
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
                completed = std::move(vad_buffer_);
                vad_buffer_.clear();
                consecutive_silence_chunks_ = 0;
                consecutive_silence_for_adaptation_ = 0;
            }
        }

        if (should_update_adaptive) {
            update_adaptive_threshold(sum_squares, current_chunk.size());
        }
        if (!completed.empty()) {
            callback(completed);
        }
    }

    static double calculate_audio_energy(const int16_t* data, size_t count) {
        double sum_squares = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const double s = static_cast<double>(data[i]);
            sum_squares += s * s;
        }
        return sum_squares;
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
            // Loud frames are likely speech/transients: do not treat them as background noise.
            return;
        }
        if (!silence_floor_initialized_.load(std::memory_order_acquire)) {
            silence_rms_ema_.store(rms, std::memory_order_release);
            silence_floor_initialized_.store(true, std::memory_order_release);
            return;
        }
        const double prev = silence_rms_ema_.load(std::memory_order_relaxed);
        const double updated = (1.0 - Constants::ADAPTIVE_NOISE_ALPHA) * prev +
                               Constants::ADAPTIVE_NOISE_ALPHA * rms;
        silence_rms_ema_.store(updated, std::memory_order_release);
        double desired = updated * Constants::ENERGY_THRESHOLD_MULTIPLIER;
        desired = std::max(desired, static_cast<double>(Constants::ADAPTIVE_THRESHOLD_MIN));
        const int base = base_energy_threshold_.load(std::memory_order_relaxed);
        desired = std::min(desired, static_cast<double>(base));
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
            setEnergyThreshold(target);
            if (verbose_) {
                std::cout << "[Adaptive VAD] RMS=" << std::fixed << std::setprecision(1) << rms
                          << " EMA=" << updated
                          << " threshold -> " << target << std::endl;
            }
        }
    }

    void prime_noise_floor_estimate(double rms) {
        if (rms <= 0.0) {
            return;
        }
        silence_rms_ema_.store(rms, std::memory_order_release);
        silence_floor_initialized_.store(true, std::memory_order_release);
    }

    void clear_processing_state() {
        std::lock_guard<std::mutex> lock(vad_mutex_);
        vad_buffer_.clear();
        pre_roll_chunks_.clear();
        consecutive_silence_chunks_ = 0;
        consecutive_silence_for_adaptation_ = 0;
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
    std::atomic<int> energy_threshold_{1000};
    std::atomic<int64_t> energy_threshold_squared_{1000LL * 1000LL};
    std::atomic<int> base_energy_threshold_{1000};
    std::atomic<bool> base_threshold_initialized_{false};
    std::atomic<bool> adaptive_energy_enabled_{false};
    std::atomic<bool> bypass_vad_{false};
    std::atomic<double> vad_pre_roll_seconds_{Constants::VAD_PRE_ROLL_SECONDS};
    std::atomic<double> silence_rms_ema_{0.0};
    std::atomic<bool> silence_floor_initialized_{false};
    std::mutex vad_mutex_;
    std::vector<int16_t> vad_buffer_;
    std::deque<std::vector<int16_t>> pre_roll_chunks_;
    size_t consecutive_silence_chunks_ = 0;
    size_t consecutive_silence_for_adaptation_ = 0;
    std::atomic<int> adaptive_hangover_chunks_{Constants::ADAPTIVE_HANGOVER_CHUNKS};
    bool verbose_ = false;
    std::string preferred_device_name_;
    int sample_rate_ = Constants::SAMPLE_RATE;
    double record_timeout_ = 2.0;
    double phrase_timeout_ = 3.0;
    size_t max_buffer_samples_ = static_cast<size_t>(Constants::SAMPLE_RATE * 2.0);
    size_t max_silence_chunks_ = 1;
    size_t max_pre_roll_chunks_ = 1;
};

std::vector<std::string> AudioRecorder::listMicrophoneNames() {
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
    }

    ~WhisperModel() {
        if (ctx_ != nullptr) {
            whisper_free(ctx_);
        }
    }

    WhisperModel(const WhisperModel&) = delete;
    WhisperModel& operator=(const WhisperModel&) = delete;

    std::string transcribe(const std::vector<float>& audio, const std::string& language) {
        if (ctx_ == nullptr || audio.empty()) {
            return {};
        }
        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.language = language.c_str();
        params.print_realtime = false;
        params.print_progress = false;
        params.print_timestamps = false;
        params.single_segment = true;
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (hw <= 0) {
            hw = 1;
        }
        params.n_threads = std::min(Constants::WHISPER_MAX_THREADS, hw);
        if (whisper_full(ctx_, params, audio.data(), static_cast<int>(audio.size())) != 0) {
            return {};
        }
        std::string out;
        const int segments = whisper_full_n_segments(ctx_);
        for (int i = 0; i < segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx_, i);
            if (text != nullptr) {
                out += text;
            }
        }
        return out;
    }

private:
    whisper_context* ctx_ = nullptr;
    std::string model_path_;
};

class AudioTranscriber {
public:
    AudioTranscriber(WhisperModel& model, std::string language)
        : model_(model), language_(std::move(language)) {
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

    std::future<std::string> transcribe_async(std::vector<float> audio) {
        std::promise<std::string> promise;
        auto future = promise.get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            // Ordered queue preserves temporal ordering for phrase reconstruction.
            transcription_queue_.push(Task{std::move(audio), std::move(promise)});
        }
        queue_cv_.notify_one();
        return future;
    }

private:
    struct Task {
        std::vector<float> audio;
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
                if (task.audio.empty()) {
                    task.promise.set_value({});
                } else {
                    task.promise.set_value(model_.transcribe(task.audio, language_));
                }
            } catch (...) {
                task.promise.set_exception(std::current_exception());
            }
        }
    }

    WhisperModel& model_;
    std::string language_;
    std::thread worker_;
    std::queue<Task> transcription_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{false};
};

namespace {
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

std::string extract_json_string_field(const std::string& json, const std::string& field) {
    // Intentionally simple parser for controlled protocol messages.
    const std::string token = std::string("\"") + field + "\":\"";
    const size_t pos = json.find(token);
    if (pos == std::string::npos) {
        return {};
    }
    const size_t start = pos + token.size();
    const size_t end = json.find('"', start);
    if (end == std::string::npos) {
        return {};
    }
    return json.substr(start, end - start);
}
}

class WebSocketPcmServer {
public:
    struct ClientContext {
        std::string session_id;
        std::string language = "en";
        int sample_rate = Constants::SAMPLE_RATE;
        int channels = 1;
        bool started = false;
    };

    using AudioCallback = std::function<void(const std::vector<int16_t>&, const std::string& session_id)>;

    WebSocketPcmServer(std::string bind_address,
                       int port,
                       bool send_transcripts)
        : bind_address_(std::move(bind_address)),
          port_(port),
          send_transcripts_(send_transcripts) {
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
                                    const std::string& timestamp) {
        if (!send_transcripts_ || session_id.empty()) {
            return;
        }

        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(session_map_mutex_);
            auto it = sessions_.find(session_id);
            if (it == sessions_.end()) {
                return;
            }
            session = it->second.lock();
        }
        if (!session) {
            return;
        }

        const std::string payload =
            std::string("{\"type\":\"transcript\",\"sessionId\":\"") +
            json_escape(session_id) +
            "\",\"timestamp\":\"" +
            json_escape(timestamp) +
            "\",\"text\":\"" +
            json_escape(text) +
            "\"}";

        session->send_text(payload);
    }

private:
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
            // Keep idle reads short so stop()/Ctrl-C can terminate session threads promptly.
            timeout.idle_timeout = std::chrono::seconds(1);
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

            owner.unregister_session(ctx.session_id);
            owner.unregister_live_session(this);
        }

        void send_text(const std::string& payload) {
            std::lock_guard<std::mutex> lock(write_mutex);
            beast::error_code ec;
            ws.text(true);
            ws.write(net::buffer(payload), ec);
        }

        void force_close() {
            beast::error_code ec;
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
        if (msg.find("\"type\":\"start\"") != std::string::npos) {
            session->ctx.started = true;
            session->ctx.session_id = extract_json_string_field(msg, "sessionId");
            const std::string language = extract_json_string_field(msg, "language");
            if (!language.empty()) {
                session->ctx.language = language;
            }
            register_session(session->ctx.session_id, session);
            session->send_text(
                std::string("{\"type\":\"ack\",\"sessionId\":\"") +
                json_escape(session->ctx.session_id) + "\"}");
            return;
        }

        if (msg.find("\"type\":\"stop\"") != std::string::npos) {
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
        if (byte_count == 0 || byte_count > Constants::MAX_WS_FRAME_BYTES) {
            // Bound frame size to protect memory and latency under malformed input.
            session->send_text("{\"type\":\"error\",\"message\":\"invalid frame size\"}");
            return;
        }
        if ((byte_count % sizeof(int16_t)) != 0u) {
            session->send_text("{\"type\":\"error\",\"message\":\"binary frame must be PCM16LE\"}");
            return;
        }

        std::vector<int16_t> samples(byte_count / sizeof(int16_t));
        net::buffer_copy(net::buffer(samples.data(), byte_count), buffer.data());
        if (callback_) {
            callback_(samples, session->ctx.session_id);
        }
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
    bool send_transcripts_ = true;
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
};

Args parse_arguments(int argc, char* argv[]) {
    Args args;
    const std::unordered_set<std::string> valid_args = {
        "--energy_threshold", "--record_timeout", "--phrase_timeout", "--language",
        "--pipe", "--timestamp", "--default_microphone", "--whisper_model_path",
        "--help", "-h", "--list_microphones", "--adaptive_energy", "--audio_file",
        "--predefined_start_time", "--vad_pre_roll", "--adaptive_silence_fraction",
        "--adaptive_hangover_chunks", "--verbose",
        "--input_source", "--websocket_server", "--websocket_bind", "--websocket_port",
        "--websocket_send_transcripts"
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
            args.language = argv[++i];
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
        } else if (arg == "--predefined_start_time" && i + 1 < argc) {
            std::tm tm{};
            std::istringstream ss(argv[++i]);
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
            if (ss.fail()) {
                std::cerr << "Error: Invalid --predefined_start_time format. Expected \"YYYY-mm-dd HH:MM:SS\"."
                          << std::endl;
                std::exit(1);
            }
            const std::time_t tt = std::mktime(&tm);
            if (tt == static_cast<std::time_t>(-1)) {
                std::cerr << "Error: Failed to convert --predefined_start_time to time." << std::endl;
                std::exit(1);
            }
            args.predefined_start_time = std::chrono::system_clock::from_time_t(tt);
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
                << " --pipe                            Enable pipe mode for continuous streaming\n"
                << " --timestamp                       Print timestamps in pipe mode\n"
                << " --whisper_model_path <path>       REQUIRED: Path to the ggml Whisper model\n"
                << " --list_microphones                List available microphones and exit\n"
                << " --audio_file <path>               Transcribe a media file. Audio is extracted via ffmpeg if needed\n"
                << " --predefined_start_time \"YYYY-mm-dd HH:MM:SS\" Override transcript start time for --audio_file mode\n"
                << " --verbose                         Print adaptive threshold changes and diagnostics\n"
                << " --input_source <mode>             microphone | file | websocket. Default: microphone\n"
                << " --websocket_server                Enable WebSocket PCM audio server\n"
                << " --websocket_bind <ip>             WebSocket bind address. Default: 0.0.0.0\n"
                << " --websocket_port <port>           WebSocket port. Default: 8080\n"
                << " --websocket_send_transcripts      Send transcript JSON messages back to clients\n";
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
    g_quit.store(true, std::memory_order_release);
}

struct QueuedAudioChunk {
    std::vector<int16_t> samples;
    std::string session_id;
    std::chrono::system_clock::time_point submitted;
};

int main(int argc, char* argv[]) {
    try {
        std::signal(SIGINT, on_sigint);
        const auto application_start_time = std::chrono::system_clock::now();
        Args args = parse_arguments(argc, argv);

        if (args.list_microphones) {
            list_and_exit();
        }

        if (args.input_source == "file") {
            // Batch mode: deterministic chunking over already available media.
            WhisperModel audio_model(args.whisper_model_path);
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
                std::string text = trim(audio_model.transcribe(chunk, args.language));
                if (!text.empty()) {
                    std::cout << format_datetime(end_time) << " " << text << std::endl;
                    std::cout.flush();
                }
                offset = end;
                const auto elapsed = std::chrono::steady_clock::now() - loop_start;
                const auto target = std::chrono::duration<double>(args.record_timeout);
                if (elapsed < target) {
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
        AudioTranscriber transcriber(audio_model, args.language);
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
                data_queue.push(QueuedAudioChunk{audio_chunk, "", std::chrono::system_clock::now()});
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
            websocket_server = std::make_unique<WebSocketPcmServer>(
                args.websocket_bind,
                args.websocket_port,
                args.websocket_send_transcripts);

            auto websocket_callback = [&](const std::vector<int16_t>& audio_chunk,
                                          const std::string& session_id) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (data_queue.size() >= Constants::MAX_QUEUED_AUDIO_CHUNKS) {
                    data_queue.pop();
                }
                data_queue.push(QueuedAudioChunk{audio_chunk, session_id, std::chrono::system_clock::now()});
                queue_cv.notify_one();
            };

            if (!websocket_server->start(websocket_callback)) {
                std::cerr << "Failed to start WebSocket audio server." << std::endl;
                PortAudioRuntime::terminate_if_initialized();
                return 1;
            }
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
                phrase_complete = true;
            }

            for (auto it = pending.begin(); it != pending.end();) {
                using namespace std::chrono_literals;
                if (it->fut.wait_for(0ms) == std::future_status::ready) {
                    std::string text;
                    try {
                        text = trim(it->fut.get());
                    } catch (const std::exception& e) {
                        std::cerr << "Transcription error: " << e.what() << std::endl;
                    }
                    if (!text.empty()) {
                        if (args.pipe) {
                            if (args.timestamp) {
                                std::cout << format_datetime(it->submitted)
                                          << " " << text << std::endl;
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
                            websocket_server->send_transcript_to_session(
                                it->session_id,
                                text,
                                format_datetime(it->submitted));
                        }
                    }
                    it = pending.erase(it);
                } else {
                    ++it;
                }
            }

            if (has_chunk && !queued_chunk.samples.empty()) {
                const int current_energy_threshold = recorder
                    ? recorder->getEnergyThreshold()
                    : std::max(args.energy_threshold > 0 ? args.energy_threshold : 1000,
                               Constants::ADAPTIVE_THRESHOLD_MIN);
                const long double silent_fraction = args.adaptive_energy
                                                        ? static_cast<long double>(args.adaptive_silence_fraction)
                                                        : 0.5L;
                if (is_silent_chunk(queued_chunk.samples, current_energy_threshold, silent_fraction)) {
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
                p.fut = transcriber.transcribe_async(std::move(audio_float));
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
