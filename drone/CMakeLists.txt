# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of the Spazio IT Speech-to-Knowledge project.
#
# Self-contained build for the drone speech-to-knowledge subsystem.
# Deliberately its own CMake project (not part of the root
# CMakeLists.txt): this targets a Jetson/aarch64 companion computer, has
# no OpenCV dependency at all (no video pipeline here), and is meant to
# be independently buildable/deployable. Build with:
#   cmake -S drone -B drone/build && cmake --build drone/build
#   ctest --test-dir drone/build

cmake_minimum_required(VERSION 3.16)

project(DroneSpeechToKnowledge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DRONE_BUILD_TESTS "Build the speech_core unit test suite" ON)

include(FetchContent)

# Pinned to the same release used by the root project's CMakeLists.txt,
# for the same reason: nlohmann/json's ABI inline namespace is derived
# from its version, and this subsystem is built independently so there's
# no shared-copy concern here, but pinning still avoids drift.
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(CURL REQUIRED)

# Dependency-light logic (the report's three-stage rotor-noise
# rejection, VAD, verbal/non-verbal distress signals, dispersal command
# mapping, INI/response parsing) covered by unit tests. No CURL/hardware
# dependency.
add_library(speech_core STATIC
  speech_core/audio_frame.h
  speech_core/biquad.cpp
  speech_core/biquad.h
  speech_core/dispersal_commands.cpp
  speech_core/dispersal_commands.h
  speech_core/fft.cpp
  speech_core/fft.h
  speech_core/ini_config.cpp
  speech_core/ini_config.h
  speech_core/non_verbal_distress.cpp
  speech_core/non_verbal_distress.h
  speech_core/notch_filter.cpp
  speech_core/notch_filter.h
  speech_core/spectral_subtraction.cpp
  speech_core/spectral_subtraction.h
  speech_core/transcription_client.cpp
  speech_core/transcription_client.h
  speech_core/vad.cpp
  speech_core/vad.h
  speech_core/verbal_response.cpp
  speech_core/verbal_response.h
  speech_core/vocal_band_filter.cpp
  speech_core/vocal_band_filter.h
)
target_include_directories(speech_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(speech_core PUBLIC nlohmann_json::nlohmann_json)

# WAV file I/O and the WAV-replay capture backend: hardware-adjacent but
# not network/subprocess-dependent, so still fully unit-testable.
add_library(drone_io STATIC
  io/mic_array_capture.h
  io/wav_file.cpp
  io/wav_file.h
  io/wav_replay_capture.cpp
  io/wav_replay_capture.h
)
target_link_libraries(drone_io PUBLIC speech_core)

add_executable(drone_speech_to_knowledge
  main_speech_to_knowledge.cpp
  io/loudspeaker_output.cpp
  io/loudspeaker_output.h
  io/transcription_http_client.cpp
  io/transcription_http_client.h
)
target_link_libraries(drone_speech_to_knowledge PRIVATE
  drone_io
  speech_core
  CURL::libcurl
)

if(DRONE_BUILD_TESTS)
  enable_testing()

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(speech_core_tests
    tests/test_dispersal_commands.cpp
    tests/test_fft.cpp
    tests/test_ini_config.cpp
    tests/test_non_verbal_distress.cpp
    tests/test_notch_filter.cpp
    tests/test_spectral_subtraction.cpp
    tests/test_transcription_client.cpp
    tests/test_vad.cpp
    tests/test_verbal_response.cpp
    tests/test_vocal_band_filter.cpp
    tests/test_wav_file.cpp
    tests/test_wav_replay_capture.cpp
  )
  target_link_libraries(speech_core_tests PRIVATE
    drone_io
    speech_core
    Catch2::Catch2WithMain
  )

  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  include(Catch)
  catch_discover_tests(speech_core_tests)
endif()
