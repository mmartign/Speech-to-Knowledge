// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// Copyright (C) 2025 Spazio IT 
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
// This file is based on https://github.com/davabase/whisper_real_time, with modifications
// for real-time performance, concurrency improvements, and debug tracing.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see https://www.gnu.org/licenses/.
//

#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cctype>
#include <atomic>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <ctime>
#include <memory>
#include <future>
#include <csignal>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <array>
#include <cstdlib>

// PortAudio includes
#include "portaudio.h"
// whisper.cpp includes
#include "whisper.h"

// =======================
// Constants
// =======================
namespace Constants {
    constexpr int SAMPLE_RATE = 16000;
    constexpr unsigned long FRAMES_PER_BUFFER = 1024;
    constexpr int MIN_AUDIO_LENGTH_MS = 100;
    constexpr size_t MIN_AUDIO_SAMPLES = static_cast<size_t>(SAMPLE_RATE * MIN_AUDIO_LENGTH_MS / 1000.0);
    constexpr double AMBIENT_NOISE_DURATION_SECONDS = 3.0;
    constexpr double ENERGY_THRESHOLD_MULTIPLIER = 2.5;
    constexpr double ADAPTIVE_NOISE_ALPHA = 0.05;        // smoothing factor for silence RMS
    constexpr double ADAPTIVE_THRESHOLD_STEP_FRACTION = 0.05; // max fraction change per tick
    constexpr int ADAPTIVE_THRESHOLD_MIN = 200;          // don't drop below this amplitude
    constexpr int WHISPER_MAX_THREADS = 4;
    constexpr int MAIN_LOOP_TIMEOUT_MS = 250;
    constexpr double PHRASE_TIMEOUT_MULTIPLIER = 1.5;
    constexpr size_t MAX_QUEUED_AUDIO_CHUNKS = 64; // backpressure
}

// =======================
// Exceptions
// =======================
class AudioException : public std::runtime_error {
public:
    explicit AudioException(const std::string& message) : std::runtime_error(message) {}
};

// =======================
// RAII wrapper for PortAudio stream
// =======================
class PortAudioStream {
private:
    PaStream* stream = nullptr;
    PaError last_err = paNoError;

public:
    PortAudioStream() = default;
    ~PortAudioStream() { close(); }

    bool open(PaStreamParameters* inputParameters,
              double sampleRate,
              unsigned long framesPerBuffer,
              PaStreamCallback* callback,
              void* userData) {
        close(); // Ensure any existing stream is closed

        last_err = Pa_OpenStream(&stream,
                                 inputParameters,
                                 nullptr, // no output
                                 sampleRate,
                                 framesPerBuffer,
                                 paNoFlag, // do not set paClipOff on input-only
                                 callback,
                                 userData);
        return last_err == paNoError;
    }

    bool start() {
        if (!stream) { last_err = paBadStreamPtr; return false; }
        last_err = Pa_StartStream(stream);
        return last_err == paNoError;
    }

    void stop() {
        if (stream) {
            Pa_StopStream(stream); // ignore error if not started
        }
    }

    void close() {
        if (stream) {
            Pa_StopStream(stream); // best-effort stop before close
            Pa_CloseStream(stream);
            stream = nullptr;
        }
    }

    PaError last_error() const { return last_err; }
    PaStream* get() const { return stream; }
};

// =======================
// CLI Args
// =======================
struct Args {
    std::string model = "medium";
    bool non_english = false;
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
    std::chrono::system_clock::time_point encoded_start{};
    bool has_encoded_start = false;
};

// =======================
// Audio Recorder Interface
// =======================
class AudioRecorder {
public:
    virtual ~AudioRecorder() = default;
    virtual bool startRecording(std::function<void(const std::vector<int16_t>&)> callback,
                                int sampleRate,
                                double recordTimeout,
                                double phraseTimeout) = 0;
    virtual void stopRecording() = 0;
    virtual void adjustForAmbientNoise(int energy_threshold) = 0;
    virtual void setEnergyThreshold(int threshold) = 0;
    virtual int getEnergyThreshold() const = 0;
    virtual void setAdaptiveEnergyEnabled(bool enabled) = 0;
    virtual void setPreferredDeviceName(const std::string& name) = 0;

    static std::vector<std::string> listMicrophoneNames();
};

// =======================
// PortAudio Recorder
// =======================
class PortAudioRecorder : public AudioRecorder {
private:
    PortAudioStream stream;
    std::function<void(const std::vector<int16_t>&)> audioCallback;

    std::atomic<bool> recordingActive{false};

    // Energy/VAD thresholds
    std::atomic<int>      energyThreshold{1000};
    std::atomic<int64_t>  energyThresholdSquared{1000LL * 1000LL};

    // Remember the initial calibrated/user threshold as a hard upper bound for adaptive updates
    std::atomic<int>  base_energy_threshold_{1000};
    std::atomic<bool> base_threshold_initialized_{false};

    int sampleRate_ = Constants::SAMPLE_RATE;
    double recordTimeout_ = 2.0;
    double phraseTimeout_ = 3.0;

    size_t max_buffer_samples_ = 0;
    size_t max_silence_chunks_ = 0;

    std::atomic<bool> bypass_vad_{false};
    std::atomic<bool> adaptive_energy_enabled_{false};
    std::atomic<double> silence_rms_ema_{0.0};
    std::atomic<bool> silence_floor_initialized_{false};

    // VAD state
    std::vector<int16_t> vad_buffer;
    std::mutex vad_buffer_mutex;
    size_t consecutive_silence_chunks_ = 0;

    // Callback protection
    std::mutex callback_mutex_;

    // Scratch buffer for zero-alloc callback
    std::vector<int16_t> scratch_;

    // Device selection
    std::string preferred_device_name_;

    // PortAudio init (thread-safe single-init)
    static std::atomic<bool> pa_initialized;
    static std::mutex pa_init_mutex;

    static int pa_callback(const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData) {
        (void)outputBuffer; (void)timeInfo; (void)statusFlags;
        PortAudioRecorder *recorder = static_cast<PortAudioRecorder*>(userData);
        const int16_t *in = static_cast<const int16_t*>(inputBuffer);

        if (inputBuffer == nullptr || !recorder->recordingActive.load(std::memory_order_acquire)) {
            return paContinue;
        }

        // Ensure scratch large enough (defensive)
        if (recorder->scratch_.size() < framesPerBuffer) {
            recorder->scratch_.resize(framesPerBuffer);
        }

        std::memcpy(recorder->scratch_.data(), in, framesPerBuffer * sizeof(int16_t));

        // Snapshot of callback
        std::function<void(const std::vector<int16_t>&)> cb;
        {
            std::lock_guard<std::mutex> lock(recorder->callback_mutex_);
            cb = recorder->audioCallback;
        }

        if (recorder->bypass_vad_.load(std::memory_order_acquire)) {
            if (cb) cb(recorder->scratch_);
            return paContinue;
        }

        // Process with VAD
        recorder->process_audio_with_vad(recorder->scratch_, framesPerBuffer, cb);
        return paContinue;
    }

    static void ensure_pa_initialized() {
        std::lock_guard<std::mutex> lock(pa_init_mutex);
        if (!pa_initialized) {
            PaError err = Pa_Initialize();
            if (err != paNoError) {
                throw AudioException("PortAudio init failed: " + std::string(Pa_GetErrorText(err)));
            }
            pa_initialized = true;
            std::atexit([]() {
                if (pa_initialized) {
                    Pa_Terminate();
                    pa_initialized = false;
                }
            });
        }
    }

    // Device selection helper
    static int pick_input_device(const std::string& name) {
        int def = Pa_GetDefaultInputDevice();
        if (name.empty()) return def;

        int n = Pa_GetDeviceCount();
        if (n < 0) return def;

        auto tolower = [](std::string s){ std::transform(s.begin(), s.end(), s.begin(),
                                                         [](unsigned char c){ return std::tolower(c); });
                                          return s; };
        std::string needle = tolower(name);

        for (int i=0; i<n; ++i) {
            const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
            if (info && info->maxInputChannels > 0) {
                std::string deviceName = info->name ? info->name : "";
                if (tolower(deviceName).find(needle) != std::string::npos) {
                    return i;
                }
            }
        }
        return def;
    }

    // Helper functions for VAD processing
    static double calculate_audio_energy(const int16_t* data, size_t n) {
        // sum of squares (not RMS) – compared against threshold^2 * N
        double sum_squares = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double s = static_cast<double>(data[i]);
            sum_squares += s * s;
        }
        return sum_squares;
    }

    void update_adaptive_threshold(double sum_squares, size_t sample_count) {
        if (!adaptive_energy_enabled_.load(std::memory_order_relaxed) || sample_count == 0) {
            return;
        }

        double rms = std::sqrt(sum_squares / static_cast<double>(sample_count));
        if (rms <= 0.0) {
            return;
        }

        if (!silence_floor_initialized_.load(std::memory_order_acquire)) {
            silence_rms_ema_.store(rms, std::memory_order_release);
            silence_floor_initialized_.store(true, std::memory_order_release);
            return;
        }

        double prev = silence_rms_ema_.load(std::memory_order_relaxed);
        double updated = (1.0 - Constants::ADAPTIVE_NOISE_ALPHA) * prev +
                         Constants::ADAPTIVE_NOISE_ALPHA * rms;
        silence_rms_ema_.store(updated, std::memory_order_release);

        double desired = updated * Constants::ENERGY_THRESHOLD_MULTIPLIER;

        if (desired < static_cast<double>(Constants::ADAPTIVE_THRESHOLD_MIN)) {
            desired = static_cast<double>(Constants::ADAPTIVE_THRESHOLD_MIN);
        }

        int current = getEnergyThreshold();
        int base    = base_energy_threshold_.load(std::memory_order_relaxed);

        // Never allow adaptive threshold to exceed the initial calibrated/user value
        if (desired > static_cast<double>(base)) {
            desired = static_cast<double>(base);
        }

        int target = static_cast<int>(desired);

        int max_step = std::max(5, static_cast<int>(current * Constants::ADAPTIVE_THRESHOLD_STEP_FRACTION));
        if (max_step < 1) max_step = 1;

        if (target > current + max_step) {
            target = current + max_step;
        } else if (target < current - max_step) {
            target = current - max_step;
        }

        if (target != current) {
            setEnergyThreshold(target);
        }
    }

    void prime_noise_floor_estimate(double rms) {
        if (rms <= 0.0) {
            return;
        }
        silence_rms_ema_.store(rms, std::memory_order_release);
        silence_floor_initialized_.store(true, std::memory_order_release);
    }

    void process_audio_with_vad(const std::vector<int16_t>& current_chunk,
                                size_t framesPerBuffer,
                                const std::function<void(const std::vector<int16_t>&)>& cb) {
        (void)framesPerBuffer;
        double sum_squares = calculate_audio_energy(current_chunk.data(), current_chunk.size());
        double threshold_squared = static_cast<double>(energyThresholdSquared.load(std::memory_order_relaxed));
        bool is_speech = sum_squares > (threshold_squared * current_chunk.size());

        if (!is_speech) {
            update_adaptive_threshold(sum_squares, current_chunk.size());
        }

        {
            std::lock_guard<std::mutex> lock(vad_buffer_mutex);
            if (is_speech) {
                consecutive_silence_chunks_ = 0;
                vad_buffer.insert(vad_buffer.end(), current_chunk.begin(), current_chunk.end());
            } else if (!vad_buffer.empty()) {
                consecutive_silence_chunks_++;
                vad_buffer.insert(vad_buffer.end(), current_chunk.begin(), current_chunk.end());
            }

            if (!vad_buffer.empty() &&
                (vad_buffer.size() >= max_buffer_samples_ ||
                 consecutive_silence_chunks_ >= max_silence_chunks_)) {

                if (cb) {
                    cb(vad_buffer);
                }
                vad_buffer.clear();
                consecutive_silence_chunks_ = 0;
            }
        }
    }

public:
    PortAudioRecorder() {
        ensure_pa_initialized();
    }

    ~PortAudioRecorder() override {
        stopRecording();
    }

    void setPreferredDeviceName(const std::string& name) override {
        preferred_device_name_ = name;
    }

    bool startRecording(std::function<void(const std::vector<int16_t>&)> callback,
                        int sampleRate,
                        double recordTimeout,
                        double phraseTimeout) override {
        if (recordingActive.load(std::memory_order_acquire)) {
            stopRecording();
        }

        if (sampleRate <= 0 || recordTimeout <= 0 || phraseTimeout <= 0) {
            throw AudioException("Invalid parameters: sample rate and timeouts must be positive");
        }

        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            audioCallback = std::move(callback);
        }

        sampleRate_ = sampleRate;
        recordTimeout_ = recordTimeout;
        phraseTimeout_ = phraseTimeout;

        // Calculate buffer limits
        max_buffer_samples_ = static_cast<size_t>(sampleRate_ * recordTimeout_);
        max_silence_chunks_ = static_cast<size_t>(
            std::ceil(phraseTimeout_ * sampleRate_ / Constants::FRAMES_PER_BUFFER));
        consecutive_silence_chunks_ = 0;

        // Prepare scratch
        scratch_.resize(Constants::FRAMES_PER_BUFFER);

        recordingActive.store(true, std::memory_order_release);
        ensure_pa_initialized();

        PaStreamParameters inputParameters{};
        inputParameters.device = pick_input_device(preferred_device_name_);
        if (inputParameters.device == paNoDevice) {
            recordingActive.store(false, std::memory_order_release);
            throw AudioException("Error: No input device.");
        }

        inputParameters.channelCount = 1;
        inputParameters.sampleFormat = paInt16;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = nullptr;

        if (!stream.open(&inputParameters, sampleRate_, Constants::FRAMES_PER_BUFFER, pa_callback, this)) {
            recordingActive.store(false, std::memory_order_release);
            throw AudioException(std::string("PortAudio error (open stream): ") + Pa_GetErrorText(stream.last_error()));
        }

        if (!stream.start()) {
            recordingActive.store(false, std::memory_order_release);
            stream.close();
            throw AudioException(std::string("PortAudio error (start stream): ") + Pa_GetErrorText(stream.last_error()));
        }

        std::cout << "Started recording on: " << Pa_GetDeviceInfo(inputParameters.device)->name << std::endl;
        return true;
    }

    void stopRecording() override {
        if (recordingActive.exchange(false)) {
            stream.stop();
            stream.close();
            std::lock_guard<std::mutex> lock(vad_buffer_mutex);
            vad_buffer.clear();
            consecutive_silence_chunks_ = 0;
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

        std::vector<int16_t> noise_samples;
        std::mutex noise_mutex;
        std::condition_variable noise_cv;
        bool noise_collection_done = false;

        // Temporary callback to collect raw audio (bypassing VAD)
        auto noise_callback = [&](const std::vector<int16_t>& audio_data) {
            std::lock_guard<std::mutex> lock(noise_mutex);
            noise_samples.insert(noise_samples.end(), audio_data.begin(), audio_data.end());
            if (noise_samples.size() >= static_cast<size_t>(Constants::SAMPLE_RATE * Constants::AMBIENT_NOISE_DURATION_SECONDS)) {
                noise_collection_done = true;
                noise_cv.notify_one();
            }
        };

        // Save/replace callback while we collect
        std::function<void(const std::vector<int16_t>&)> old_cb;
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            old_cb = audioCallback;
            audioCallback = noise_callback;
        }

        // Start the stream if not already running
        bool was_running = recordingActive.load(std::memory_order_acquire);
        if (!was_running) {
            if (!startRecording(noise_callback, sampleRate_, recordTimeout_, phraseTimeout_)) {
                // restore previous callback
                std::lock_guard<std::mutex> lock(callback_mutex_);
                audioCallback = old_cb;
                throw AudioException("Failed to start recording for ambient noise adjustment.");
            }
        }

        // Ensure VAD is bypassed for calibration
        bypass_vad_.store(true, std::memory_order_release);

        // Wait for noise collection to complete (with a timeout)
        {
            std::unique_lock<std::mutex> lock(noise_mutex);
            (void)noise_cv.wait_for(lock, std::chrono::seconds(4), [&]{ return noise_collection_done; });
        }

        // Stop bypass
        bypass_vad_.store(false, std::memory_order_release);

        // Restore previous state
        if (!was_running) {
            stopRecording();
        }
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            audioCallback = old_cb;
        }

        if (noise_samples.empty()) {
            throw AudioException("No noise samples collected. Using default energy threshold.");
        }

        // Calculate RMS and set threshold
        long double sum_squares = 0.0;
        for (int16_t s : noise_samples) {
            long double v = static_cast<long double>(s);
            sum_squares += v * v;
        }
        long double rms = std::sqrt(sum_squares / static_cast<long double>(noise_samples.size()));
        int threshold = static_cast<int>(rms * Constants::ENERGY_THRESHOLD_MULTIPLIER);
        setEnergyThreshold(threshold);
        if (adaptive_energy_enabled_.load(std::memory_order_relaxed)) {
            prime_noise_floor_estimate(static_cast<double>(rms));
        }
        std::cout << "Adjusted energy threshold to: " << threshold << std::endl;
    }

    void setEnergyThreshold(int threshold) override {
        // The very first call to setEnergyThreshold (from user or calibration)
        // becomes our "base" threshold. Adaptive logic will never exceed this.
        bool expected = false;
        if (base_threshold_initialized_.compare_exchange_strong(expected, true,
                                                                std::memory_order_acq_rel,
                                                                std::memory_order_acquire)) {
            base_energy_threshold_.store(threshold, std::memory_order_relaxed);
        }

        energyThreshold.store(threshold, std::memory_order_relaxed);
        energyThresholdSquared.store(static_cast<int64_t>(threshold) * static_cast<int64_t>(threshold),
                                     std::memory_order_relaxed);
    }

    int getEnergyThreshold() const override { return energyThreshold.load(std::memory_order_relaxed); }

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

    // For tests / status
    static bool isInitialized() { return pa_initialized.load(); }
};

// Initialize static members
std::atomic<bool> PortAudioRecorder::pa_initialized{false};
std::mutex PortAudioRecorder::pa_init_mutex;

// =======================
// List microphones
// =======================
std::vector<std::string> AudioRecorder::listMicrophoneNames() {
    std::vector<std::string> names;

    try {
        // ensure PortAudio initialized exactly once
        // (PortAudioRecorder constructor also ensures this, but listing works even without instance)
        {
            // inline the ensure call
            static std::once_flag init_flag;
            std::call_once(init_flag, []{
                PaError err = Pa_Initialize();
                if (err != paNoError) {
                    throw AudioException("PortAudio init failed: " + std::string(Pa_GetErrorText(err)));
                }
                std::atexit([](){ Pa_Terminate(); });
            });
        }

        int numDevices = Pa_GetDeviceCount();
        if (numDevices < 0) {
            throw AudioException("PortAudio error (device count): " + std::string(Pa_GetErrorText(numDevices)));
        }

        for (int i = 0; i < numDevices; ++i) {
            const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(i);
            if (deviceInfo && deviceInfo->maxInputChannels > 0) {
                names.emplace_back(deviceInfo->name ? deviceInfo->name : "(unnamed)");
            }
        }
    } catch (const AudioException& e) {
        std::cerr << e.what() << std::endl;
    }

    return names;
}

// =======================
// Whisper wrapper
// =======================
class WhisperModel {
private:
    whisper_context *ctx = nullptr;
    std::string model_path;

public:
    explicit WhisperModel(const std::string& modelPath) : model_path(modelPath) {
        if (!std::filesystem::exists(modelPath)) {
            throw AudioException("Model file does not exist: " + modelPath);
        }

        std::cout << "Loading Whisper model from: " << model_path << std::endl;
        ctx = whisper_init_from_file(model_path.c_str());
        if (!ctx) {
            throw AudioException("Failed to load Whisper model from " + model_path);
        }
    }

    ~WhisperModel() {
        if (ctx) whisper_free(ctx);
    }

    std::string transcribe(const std::vector<float>& audio_data_normalized, const std::string& lang) {
        if (!ctx || audio_data_normalized.empty()) {
            return "";
        }

        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.language = lang.c_str();

        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (hw <= 0) hw = 1;
        params.n_threads = std::min(Constants::WHISPER_MAX_THREADS, hw);

        params.print_realtime = false;
        params.print_progress = false;
        params.print_timestamps = false;
        params.single_segment = true;

        if (whisper_full(ctx, params, audio_data_normalized.data(), static_cast<int>(audio_data_normalized.size())) != 0) {
            return "";
        }

        std::string result_text;
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char *text = whisper_full_get_segment_text(ctx, i);
            if (text) result_text += text;
        }
        return result_text;
    }
};

// =======================
// Async transcriber
// =======================
class AudioTranscriber {
private:
    WhisperModel& model;
    std::string language;
    std::thread transcription_thread;
    std::queue<std::pair<std::vector<float>, std::promise<std::string>>> transcription_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> running{false};

    void transcription_worker() {
        while (running.load(std::memory_order_acquire) || !transcription_queue.empty()) {
            std::pair<std::vector<float>, std::promise<std::string>> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&] {
                    return !transcription_queue.empty() || !running.load(std::memory_order_acquire);
                });

                if (!running.load(std::memory_order_acquire) && transcription_queue.empty()) {
                    break;
                }

                if (!transcription_queue.empty()) {
                    task = std::move(transcription_queue.front());
                    transcription_queue.pop();
                }
            }

            if (!task.first.empty()) {
                std::string result = model.transcribe(task.first, language);
                task.second.set_value(result);
            } else {
                task.second.set_value(std::string{});
            }
        }
    }

public:
    AudioTranscriber(WhisperModel& model, const std::string& lang)
        : model(model), language(lang) {
        running.store(true, std::memory_order_release);
        transcription_thread = std::thread(&AudioTranscriber::transcription_worker, this);
    }

    ~AudioTranscriber() {
        running.store(false, std::memory_order_release);
        queue_cv.notify_one();
        if (transcription_thread.joinable()) {
            transcription_thread.join();
        }
    }

    std::future<std::string> transcribe_async(const std::vector<float>& audio_data) {
        std::promise<std::string> promise;
        auto future = promise.get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            transcription_queue.emplace(audio_data, std::move(promise));
        }
        queue_cv.notify_one();
        return future;
    }
};

// =======================
// Utilities
// =======================
Args parse_arguments(int argc, char* argv[]) {
    Args args;
    const std::unordered_set<std::string> valid_args = {
        "--model", "--non_english", "--energy_threshold", "--record_timeout",
        "--phrase_timeout", "--language", "--pipe", "--default_microphone",
        "--whisper_model_path", "--help", "-h", "--timestamp", "--list_microphones",
        "--adaptive_energy", "--audio_file", "--encoded_start"
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (valid_args.find(arg) == valid_args.end()) {
            std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
    std::exit(1);
}

        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--non_english") {
            args.non_english = true;
        } else if (arg == "--energy_threshold" && i + 1 < argc) {
            try {
                args.energy_threshold = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid energy threshold value" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--record_timeout" && i + 1 < argc) {
            try {
                args.record_timeout = std::stod(argv[++i]);
                if (args.record_timeout <= 0) {
                    std::cerr << "Error: record_timeout must be positive" << std::endl;
                    std::exit(1);
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid record_timeout value" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--phrase_timeout" && i + 1 < argc) {
            try {
                args.phrase_timeout = std::stod(argv[++i]);
                if (args.phrase_timeout <= 0) {
                    std::cerr << "Error: phrase_timeout must be positive" << std::endl;
                    std::exit(1);
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid phrase_timeout value" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--language" && i + 1 < argc) {
            args.language = argv[++i];
        } else if (arg == "--pipe") {
            args.pipe = true;
        } else if (arg == "--timestamp") {
            args.timestamp = true;
        } else if (arg == "--adaptive_energy") {
            args.adaptive_energy = true;
        } else if (arg == "--default_microphone" && i + 1 < argc) {
            args.default_microphone = argv[++i];
        } else if (arg == "--whisper_model_path" && i + 1 < argc) {
            args.whisper_model_path = argv[++i];
        } else if (arg == "--list_microphones") {
            args.list_microphones = true;
        } else if (arg == "--audio_file" && i + 1 < argc) {
            args.audio_file_path = argv[++i];
        } else if (arg == "--encoded_start" && i + 1 < argc) {
            std::string ts = argv[++i];
            std::tm tm{};
            std::istringstream ss(ts);
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
            if (ss.fail()) {
                std::cerr << "Error: Invalid --encoded_start format. Expected \"YYYY-mm-dd HH:MM:SS\"." << std::endl;
                std::exit(1);
            }
            std::time_t tt = std::mktime(&tm);
            if (tt == -1) {
                std::cerr << "Error: Failed to convert --encoded_start to time." << std::endl;
                std::exit(1);
            }
            args.encoded_start = std::chrono::system_clock::from_time_t(tt);
            args.has_encoded_start = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --model <name>            Model to use (tiny, base, small, medium, large). Default: medium\n"
                      << "  --non_english             Don't use the English-specific model variant.\n"
                      << "  --energy_threshold <int>  Energy level for mic to detect. Default: auto-adjust\n"
                      << "  --adaptive_energy         Continuously adapt the energy threshold based on silence.\n"
                      << "  --record_timeout <float>  Max duration for audio chunks (seconds). Default: 2.0\n"
                      << "  --phrase_timeout <float>  Silence duration to end a phrase (seconds). Default: 3.0\n"
                      << "  --language <lang>         Language (de, en, es, fr, he, it, sv). Default: en\n"
                      << "  --pipe                    Enable pipe mode for continuous streaming.\n"
                      << "  --timestamp               Print timestamps before each line in pipe mode.\n"
                      << "  --whisper_model_path <path> REQUIRED: Path to the ggml Whisper model file\n"
                      << "  --list_microphones        List available microphones and exit\n"
                      << "  --audio_file <path>       Transcribe a media file (audio or video). Audio is extracted via ffmpeg.\n"
                      << "  --encoded_start \"YYYY-mm-dd HH:MM:SS\"  Override transcript start time for --audio_file mode.\n"
#ifdef __linux__
                      << "  --default_microphone <name> Default microphone name. Use '--list_microphones' to see options.\n"
#endif
                      ;
            std::exit(0);
        }
    }

    if (args.whisper_model_path.empty() && !args.list_microphones) {
        std::cerr << "Error: --whisper_model_path is required." << std::endl;
        std::exit(1);
    }

    return args;
}

// =======================
// File audio loading (mono 16-bit WAV or raw PCM; media via ffmpeg)
// =======================
namespace FileAudio {
    namespace {
        uint32_t read_u32_le(std::ifstream& f) {
            uint32_t v = 0;
            f.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        }

        uint16_t read_u16_le(std::ifstream& f) {
            uint16_t v = 0;
            f.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        }
    }

    bool load_wav_mono_16(const std::string& path, int& sample_rate_out, std::vector<int16_t>& samples_out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        std::array<char, 4> tag{};
        f.read(tag.data(), 4);
        if (f.gcount() != 4 || std::string(tag.data(), 4) != "RIFF") return false;

        (void)read_u32_le(f); // chunk size
        f.read(tag.data(), 4);
        if (f.gcount() != 4 || std::string(tag.data(), 4) != "WAVE") return false;

        bool fmt_found = false;
        bool data_found = false;
        uint16_t num_channels = 0;
        uint16_t bits_per_sample = 0;
        sample_rate_out = 0;
        samples_out.clear();

        while (f && !data_found) {
            f.read(tag.data(), 4);
            if (f.gcount() != 4) break;
            uint32_t chunk_size = read_u32_le(f);
            std::string chunk_id(tag.data(), 4);

            if (chunk_id == "fmt ") {
                uint16_t audio_format = read_u16_le(f);
                num_channels = read_u16_le(f);
                sample_rate_out = static_cast<int>(read_u32_le(f));
                (void)read_u32_le(f); // byte rate
                (void)read_u16_le(f); // block align
                bits_per_sample = read_u16_le(f);

                // skip any extra fmt bytes
                if (chunk_size > 16) {
                    f.seekg(chunk_size - 16, std::ios::cur);
                }

                if (audio_format != 1) return false; // only PCM
                fmt_found = true;
            } else if (chunk_id == "data") {
                if (!fmt_found) return false;
                if (num_channels != 1 || bits_per_sample != 16) return false;
                size_t sample_count = chunk_size / sizeof(int16_t);
                samples_out.resize(sample_count);
                f.read(reinterpret_cast<char*>(samples_out.data()), static_cast<std::streamsize>(sample_count * sizeof(int16_t)));
                data_found = true;
            } else {
                f.seekg(chunk_size, std::ios::cur);
            }
        }

        return data_found && fmt_found && !samples_out.empty();
    }

    std::vector<int16_t> load_raw_pcm_16(const std::string& path, int sample_rate_expected) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return {};
        f.seekg(0, std::ios::end);
        std::streamsize sz = f.tellg();
        f.seekg(0, std::ios::beg);
        if (sz <= 0 || sz % static_cast<std::streamsize>(sizeof(int16_t)) != 0) {
            return {};
        }
        std::vector<int16_t> data(static_cast<size_t>(sz / sizeof(int16_t)));
        f.read(reinterpret_cast<char*>(data.data()), sz);
        (void)sample_rate_expected; // kept for symmetry / future resample
        return data;
    }

    std::vector<float> to_float(const std::vector<int16_t>& samples) {
        std::vector<float> out(samples.size());
        constexpr float denom = 32768.0f;
        for (size_t i = 0; i < samples.size(); ++i) {
            out[i] = static_cast<float>(samples[i]) / denom;
        }
        return out;
    }

    std::string transcode_media_to_wav(const std::string& media_path, int target_sample_rate) {
        namespace fs = std::filesystem;
        auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
        fs::path tmp = fs::temp_directory_path() / ("stkw_media_" + std::to_string(ts) + ".wav");

        std::ostringstream cmd;
        cmd << "ffmpeg -y -i \"" << media_path << "\" -vn -ac 1 -ar "
            << target_sample_rate << " -sample_fmt s16 \"" << tmp.string()
            << "\" > /dev/null 2>&1";

        int ret = std::system(cmd.str().c_str());
        if (ret != 0 || !fs::exists(tmp)) {
            throw AudioException("Failed to extract audio from media file (ffmpeg required).");
        }
        return tmp.string();
    }
}

void clear_console() {
#ifdef _WIN32
    std::system("cls");
#else
    std::cout << "\033[2J\033[H";
#endif
}

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm{};
#if defined(_WIN32)
    localtime_s(&now_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &now_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "[%Y-%m-%d %H:%M:%S]");
    return oss.str();
}

template <typename Duration>
std::string format_datetime(const std::chrono::time_point<std::chrono::system_clock, Duration>& tp) {
    // Cast to system_clock::duration so to_time_t accepts it even if the incoming duration is fractional
    auto tp_sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(tp);
    std::time_t tt = std::chrono::system_clock::to_time_t(tp_sys);
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
        std::vector<std::string> mics = AudioRecorder::listMicrophoneNames();
        if (mics.empty()) {
            std::cout << "No microphones found." << std::endl;
        } else {
            std::cout << "Available microphones:" << std::endl;
            for (size_t i = 0; i < mics.size(); ++i) {
                std::cout << "  " << (i + 1) << ". " << mics[i] << std::endl;
            }
        }
        std::exit(0);
    } catch (const std::exception& e) {
        std::cerr << "Error listing microphones: " << e.what() << std::endl;
        std::exit(1);
    }
}

// Simple RMS-based silence detector on int16 chunks.
// We compare the RMS against a fraction of the current energy threshold.
bool is_silent_chunk(const std::vector<int16_t>& samples, int energy_threshold) {
    if (samples.empty()) {
        return true;
    }

    long double sum_squares = 0.0L;
    for (int16_t s : samples) {
        long double v = static_cast<long double>(s);
        sum_squares += v * v;
    }

    long double rms = std::sqrt(sum_squares / static_cast<long double>(samples.size()));

    // Energy threshold is an amplitude; here we treat as "no speech"
    // anything well below it. The factor (e.g. 0.5L) is tunable.
    const long double kFraction = 0.5L;
    long double min_rms = static_cast<long double>(energy_threshold) * kFraction;

    return rms < min_rms;
}

std::vector<float> load_audio_from_file(const std::string& path) {
    int sample_rate = 0;
    std::vector<int16_t> pcm;

    if (FileAudio::load_wav_mono_16(path, sample_rate, pcm)) {
        if (sample_rate != Constants::SAMPLE_RATE) {
            throw AudioException("Unsupported WAV sample rate. Expected 16000 Hz mono 16-bit PCM.");
        }
    } else { // not a WAV we recognize
        // Try raw PCM first
        pcm = FileAudio::load_raw_pcm_16(path, Constants::SAMPLE_RATE);
        sample_rate = Constants::SAMPLE_RATE;

        if (pcm.empty()) {
            // Last resort: treat as media (e.g., video) and extract audio with ffmpeg
            std::string tmp_wav = FileAudio::transcode_media_to_wav(path, Constants::SAMPLE_RATE);
            bool ok = FileAudio::load_wav_mono_16(tmp_wav, sample_rate, pcm);
            std::filesystem::remove(tmp_wav);

            if (!ok || sample_rate != Constants::SAMPLE_RATE) {
                throw AudioException("Failed to decode media file. Ensure ffmpeg is installed and input is valid.");
            }
        }
    }

    if (pcm.size() < Constants::MIN_AUDIO_SAMPLES) {
        pcm.resize(Constants::MIN_AUDIO_SAMPLES, 0);
    }

    return FileAudio::to_float(pcm);
}

// =======================
// Graceful shutdown handling
// =======================
std::atomic<bool> g_quit{false};
void on_sigint(int) { g_quit.store(true, std::memory_order_release); }

// =======================
// Main
// =======================
int main(int argc, char* argv[]) {
    try {
        std::signal(SIGINT, on_sigint);

        Args args = parse_arguments(argc, argv);

        if (args.list_microphones) {
            list_and_exit();
        }

        if (!args.audio_file_path.empty()) {
            WhisperModel audio_model(args.whisper_model_path);
            auto audio_np_full = load_audio_from_file(args.audio_file_path);
            std::cout << "Transcribing media file: " << args.audio_file_path << std::endl;

            size_t chunk_samples = static_cast<size_t>(Constants::SAMPLE_RATE * args.record_timeout);
            if (chunk_samples < Constants::MIN_AUDIO_SAMPLES) {
                chunk_samples = Constants::MIN_AUDIO_SAMPLES;
            }

            auto transcript_start = args.has_encoded_start
                                        ? args.encoded_start
                                        : std::chrono::system_clock::now();

            size_t offset = 0;
            while (offset < audio_np_full.size()) {
                if (g_quit.load(std::memory_order_acquire)) {
                    break;
                }

                auto loop_start = std::chrono::steady_clock::now();

                size_t end = std::min(offset + chunk_samples, audio_np_full.size());
                std::vector<float> chunk(audio_np_full.begin() + static_cast<std::ptrdiff_t>(offset),
                                         audio_np_full.begin() + static_cast<std::ptrdiff_t>(end));
                if (chunk.size() < Constants::MIN_AUDIO_SAMPLES) {
                    chunk.resize(Constants::MIN_AUDIO_SAMPLES, 0.0f);
                }

                double end_sec   = static_cast<double>(end) / Constants::SAMPLE_RATE;
                auto end_time = transcript_start + std::chrono::duration<double>(end_sec);

                std::string text = audio_model.transcribe(chunk, args.language);
                text = trim(text);
                if (!text.empty()) {
                    std::cout << format_datetime(end_time) << " " << text << std::endl;
                    std::cout.flush();
                }

                offset = end;

                if (g_quit.load(std::memory_order_acquire)) {
                    break;
                }

                // Pace processing so each loop spans approximately record_timeout seconds
                auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
                auto target = std::chrono::duration<double>(args.record_timeout);
                if (loop_elapsed < target) {
                    std::this_thread::sleep_for(target - loop_elapsed);
                }
            }

            return 0;
        }

        std::chrono::time_point<std::chrono::system_clock> last_phrase_end_time{};
        bool phrase_time_set = false;

        // Audio queue from recorder -> main thread
        std::queue<std::vector<int16_t>> data_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;

        // Recorder
        std::unique_ptr<PortAudioRecorder> recorder;
        try {
            recorder = std::make_unique<PortAudioRecorder>();
            recorder->setPreferredDeviceName(args.default_microphone);
        } catch (const AudioException& e) {
            std::cerr << "Failed to initialize recorder: " << e.what() << std::endl;
            return 1;
        }

        // Whisper model + async transcriber
        WhisperModel audio_model(args.whisper_model_path);
        AudioTranscriber transcriber(audio_model, args.language);

        // Buffer of displayed lines (non-pipe)
        std::vector<std::string> transcription = {""};

        // Calibrate microphone if energy threshold not set
        if (args.energy_threshold == -1) {
            std::cout << "Calibrating microphone..." << std::endl;
            recorder->adjustForAmbientNoise(args.energy_threshold);
        } else {
            recorder->setEnergyThreshold(args.energy_threshold);
            std::cout << "Using energy threshold: " << args.energy_threshold << std::endl;
        }

        recorder->setAdaptiveEnergyEnabled(args.adaptive_energy);
        if (args.adaptive_energy) {
            std::cout << "Adaptive energy threshold enabled (EMA over silence)." << std::endl;
        }

        // Start continuous recording
        auto record_callback = [&](const std::vector<int16_t>& audio_data) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (data_queue.size() >= Constants::MAX_QUEUED_AUDIO_CHUNKS) {
                // drop the oldest to apply backpressure
                data_queue.pop();
            }
            data_queue.push(audio_data);
            queue_cv.notify_one();
        };

        if (!recorder->startRecording(record_callback, Constants::SAMPLE_RATE, args.record_timeout, args.phrase_timeout)) {
            std::cerr << "Failed to start continuous recording." << std::endl;
            return 1;
        }

        if (!args.pipe) {
            std::cout << "Model loaded and recording started.\n" << std::endl;
        }

        struct Pending {
            std::future<std::string> fut;
            std::chrono::system_clock::time_point submitted;
            bool starts_new_phrase;
        };
        std::deque<Pending> pending;

        while (!g_quit.load(std::memory_order_acquire)) {
            std::vector<int16_t> audio_data;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait_for(lock,
                                  std::chrono::milliseconds(Constants::MAIN_LOOP_TIMEOUT_MS),
                                  [&]{ return !data_queue.empty() || g_quit.load(); });

                if (!data_queue.empty()) {
                    audio_data = std::move(data_queue.front());
                    data_queue.pop();
                }
            }

            auto now = std::chrono::system_clock::now();

            bool phrase_complete = false;
            if (phrase_time_set &&
                (now - last_phrase_end_time) > std::chrono::duration<double>(args.phrase_timeout)) {
                phrase_complete = true;
            }

            // Harvest finished transcriptions (non-blocking)
            for (auto it = pending.begin(); it != pending.end();) {
                using namespace std::chrono_literals;
                if (it->fut.wait_for(0ms) == std::future_status::ready) {
                    std::string text = trim(it->fut.get());
                    if (!text.empty()) {
                        if (args.pipe) {
                            if (args.timestamp) {
                                std::cout << get_current_timestamp() << " " << text << std::endl;
                            } else {
                                std::cout << text << std::endl;
                            }
                        } else {
                            // For simple UX: if it was submitted when a phrase was considered complete,
                            // append as a new line; otherwise, update the current last line.
                            if (it->starts_new_phrase) {
                                transcription.push_back(text);
                            } else {
                                transcription.back() = text;
                            }
                            clear_console();
                            for (const auto& line : transcription) {
                                if (!line.empty()) std::cout << line << std::endl;
                            }
                            std::cout << std::flush;
                        }
                    }
                    it = pending.erase(it);
                } else {
                    ++it;
                }
            }

            if (!audio_data.empty()) {
                // Skip near-silent chunks to avoid Whisper hallucinating "Thank you" etc. ---
                int current_energy_threshold = recorder->getEnergyThreshold();
                if (is_silent_chunk(audio_data, current_energy_threshold)) {
                    // Treat as no speech: do not update phrase timing and do not send to Whisper.
                    // This prevents low-energy/no-speech buffers from producing fake text.
                    // (phrase_complete logic above still works off the last real speech.)
                    continue;
            }

                last_phrase_end_time = now;
                phrase_time_set = true;

                // Pad to minimum length if needed
                if (audio_data.size() < Constants::MIN_AUDIO_SAMPLES) {
                    audio_data.resize(Constants::MIN_AUDIO_SAMPLES, 0);
                }

                // Convert to float [-1, 1]
                std::vector<float> audio_np(audio_data.size());
                for (size_t i = 0; i < audio_data.size(); ++i) {
                    audio_np[i] = static_cast<float>(audio_data[i]) / 32768.0f;
                }

                // Submit asynchronous transcription without blocking
                Pending p;
                p.fut = transcriber.transcribe_async(audio_np);
                p.submitted = now;
                p.starts_new_phrase = phrase_complete; // snapshot decision
                if (p.starts_new_phrase && !args.pipe && !transcription.back().empty()) {
                    // We'll append when result returns; optionally reserve a placeholder
                    transcription.push_back("");
                }
                pending.emplace_back(std::move(p));
            } else {
                // idle tick: if enough time passed with no audio, ensure a new line boundary next time
                if (phrase_time_set &&
                    (now - last_phrase_end_time) >
                        std::chrono::duration<double>(args.phrase_timeout * Constants::PHRASE_TIMEOUT_MULTIPLIER)) {
                    if (!args.pipe && !transcription.back().empty()) {
                        transcription.push_back("");
                    }
                    phrase_time_set = false;
                }
            }
        }

        // Graceful shutdown
        recorder->stopRecording();
        // pending futures will be resolved eventually as transcriber drains on destruction

    } catch (const AudioException& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}
