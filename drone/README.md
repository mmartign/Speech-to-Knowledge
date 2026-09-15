<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Drone Speech-to-Knowledge

An implementation of the **Speech-to-knowledge** pipeline from the
companion-computer architecture described in the project's Innovation
Report (airborne edge triage for earthquake mass-casualty incidents):

> Speech-to-knowledge captures undercarriage audio with a ReSpeaker 4-Mic
> Array (Seeed Studio, Shenzhen, China; 38 g). Rotor noise is rejected in
> three stages: adaptive notch filtering synchronized to motor telemetry,
> spectral subtraction of broadband downwash, and voice-activity gating to
> the vocal band. Cleaned audio feeds a quantized speech recognizer for
> short clinical replies and a convolutional classifier for non-verbal
> distress. [...] Completing the assessment loop requires the aircraft to
> issue an audible instruction [...] via a Zenmuse V1 loudspeaker (DJI,
> Shenzhen, China) [...] in text-to-speech mode with a short set of
> pre-scripted phrases.

This directory implements exactly that. It does **not** implement
video-to-knowledge, the cross-pipeline rule-based triage aggregator
(decision tree), packet signing, or mesh radio telemetry — those are
separate parts of the full architecture and weren't in scope here.

## Scope and what's real vs. a documented stand-in

Everything below is real, working code unless flagged otherwise — no
silent stubs. Where the report specifies commercial hardware or a
trained model this environment doesn't have, that's called out
explicitly rather than faked.

**The three-stage rotor-noise rejection, matching the report's own
stage names:**

| Stage | Report language | Implementation |
|---|---|---|
| 1 | "adaptive notch filtering synchronized to motor telemetry" | `speech_core/notch_filter.h` — a bank of IIR notch filters at the blade-pass frequency (`RPM/60 × blade count`) and its harmonics. There's no live RPM feed in this offline/replay environment, so the caller supplies it (`--rotor-rpm`); a live integration would read it from the flight controller telemetry the report mentions. |
| 2 | "spectral subtraction of broadband downwash" | `speech_core/spectral_subtraction.h` — classic magnitude spectral subtraction (Boll, 1979) over a short-time Fourier transform, via a small self-contained FFT (`speech_core/fft.h`; not a general-purpose DSP library, just enough for this). |
| 3 | "voice-activity gating to the vocal band" | Band-limiting half: `speech_core/vocal_band_filter.h` (300–3400 Hz band-pass). Gating half: `speech_core/vad.h` (energy + zero-crossing-rate VAD), applied after it. |

**Quantized speech recognition** — not run in-process. Consistent with
how the video-to-knowledge pipeline elsewhere in this repo calls a local
OpenAI-compatible vision model server rather than embedding one, this is
a thin client (`io/transcription_http_client.h`) against a local
`/v1/audio/transcriptions` endpoint (the shape used by whisper.cpp's
server example, faster-whisper servers, and OpenAI's own API) —
presumably a quantized GGML/GGUF Whisper model served locally on the
Jetson. See `config.example.ini`.

**"a convolutional classifier for non-verbal distress"** —
`speech_core/non_verbal_distress.h` is explicitly **not** that. A CNN
over acoustic features (e.g. a mel-spectrogram) needs training data and
a training pipeline this environment has neither of. What's there
instead is a documented placeholder — simple vocal-intensity and
pitch-irregularity heuristics — sitting behind the same interface a
trained model would, so one can be dropped in later without touching
callers. Treat its output as indicative at best.

**"short clinical replies"** — `speech_core/verbal_response.h` reads
the ASR transcript for two signals: whether the casualty said anything
intelligible at all (proxy for responsiveness), and whether recognized
distress/pain keywords (English + Italian) are present. This is
deliberately *not* called a "distress classifier" — the report reserves
that term for the CNN above — and is a plain rule-based reading of
recognized words, not a claim about matching the report's clinical
reply handling in the full aggregator.

**The Zenmuse V1 loudspeaker** — the report specifies the companion
computer drives it "in text-to-speech mode with a short set of
pre-scripted phrases" over DJI's Payload SDK: text goes up, the V1's own
firmware synthesizes and broadcasts it. There's no DJI Payload SDK or V1
hardware here to integrate against. `speech_core/dispersal_commands.h`
still defines the pre-scripted phrase catalog
(`dispersalCommandPhraseText()`) that production code would send over
that SDK; `io/loudspeaker_output.h`'s `playDispersalCommand()` is an
offline/simulation stand-in that plays a pre-recorded WAV via `aplay`
instead, useful for testing the pipeline end-to-end without DJI
hardware, but **not** the production path.

**The ReSpeaker 4-Mic Array** — `io/mic_array_capture.h` defines the
capture interface; no live hardware backend is implemented (no Jetson
or physical array here to build and verify one against).
`io/wav_replay_capture.h` implements the same interface against a WAV
file instead, so the rest of the pipeline is real and testable
end-to-end today. One integration detail worth flagging for whoever
builds the live backend: the ReSpeaker 4-Mic Array is sold as a
Raspberry Pi HAT (I2S, 40-pin header) with its own `seeed-voicecard`
ALSA driver; wiring it to a Jetson Orin NX's I2S header is a different
(if similar-shaped) integration than that driver assumes, and worth
verifying against Seeed's own guidance before committing to this exact
part for a Jetson build.

**Channel combination** — the report doesn't describe how the four
microphone channels get combined before the noise-rejection stages
(no beamforming or array geometry is mentioned), so
`main_speech_to_knowledge.cpp` does a plain unweighted average across
channels, not delay-and-sum beamforming or similar spatial processing.

## Architecture

```
speech_core/          Pure, dependency-light logic — unit tested.
  audio_frame.h           MultiChannelFrame / MonoFrame buffer types
  biquad.{h,cpp}           shared IIR biquad primitive (notch/lowpass/highpass)
  notch_filter.{h,cpp}     stage 1: adaptive notch filtering
  fft.{h,cpp}              small radix-2 FFT (STFT building block)
  spectral_subtraction.{h,cpp}  stage 2: spectral subtraction
  vocal_band_filter.{h,cpp}     stage 3 (band-limiting half)
  vad.{h,cpp}              stage 3 (gating half): energy + ZCR VAD
  transcription_client.{h,cpp}  ASR request/response shape (pure)
  non_verbal_distress.{h,cpp}   CNN placeholder (see above)
  verbal_response.{h,cpp}       transcript keyword/response reading
  dispersal_commands.{h,cpp}    phrase catalog + selection rule
  ini_config.{h,cpp}      [asr] config.ini parsing

io/                    Hardware/network-facing — not unit tested (same
                       role as realtime_video_pipeline.cpp in the
                       parent project), except wav_file.{h,cpp} and
                       wav_replay_capture.{h,cpp}, which need no
                       hardware and are tested.
  wav_file.{h,cpp}                PCM16 WAV read/write
  mic_array_capture.h             capture interface (no live backend)
  wav_replay_capture.{h,cpp}      WAV-file capture backend
  transcription_http_client.{h,cpp}  libcurl multipart POST to the ASR server
  loudspeaker_output.{h,cpp}      offline WAV-playback stand-in (see above)

main_speech_to_knowledge.cpp   Wires it all together over a WAV file.
```

Because there's no live microphone-array backend, `drone_speech_to_knowledge`
runs in **offline/replay mode**: it takes a pre-recorded 4-channel WAV file,
not a live stream, and prints one JSON result per detected speech segment.

## Building and testing

Self-contained CMake project, independent of the root
`Video-to-Knowledge` build (different target platform, no OpenCV
dependency):

```sh
cmake -S drone -B drone/build
cmake --build drone/build
ctest --test-dir drone/build --output-on-failure
```

Requires libcurl on the host; nlohmann/json and Catch2 are fetched
automatically. Set `-DDRONE_BUILD_TESTS=OFF` to skip building tests.

## Running

```sh
cmake --build drone/build
./drone/build/drone_speech_to_knowledge <4ch_input.wav> drone/config.ini \
    [--rotor-rpm <rpm>] [--rotor-blades <n>] [--audio-dir ./drone/audio] [--dispatch]
```

`config.ini` needs an `[asr]` section pointing at a running
OpenAI-compatible transcription server (see `config.example.ini`).
`--rotor-rpm` enables stage 1's notch filtering (omit it, and it's
skipped, since there's no live motor telemetry feed in offline mode).
`--dispatch` actually plays the selected dispersal command's WAV
fallback (via `aplay`); omit it to just see the JSON output.

Each detected speech segment prints one JSON line to stdout, e.g.:

```json
{"segment_start_sample":15872,"segment_end_sample":32000,"sample_rate_hz":16000,"transcript":"help, I'm trapped","peak_energy_ratio":3.2,"responded":true,"verbal_distress_language":true,"matched_keywords":["trapped"],"non_verbal_distress_level":"high","non_verbal_distress_confidence":0.8,"dispersal_command":"do_not_move","dispersal_phrase":"Do not move. Rescue is coming to you."}
```
