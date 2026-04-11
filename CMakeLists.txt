# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.16)

project(SpeechToKnowledge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Paths kept configurable to mirror the original Makefile defaults.
set(OPENAI_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../openai-cpp" CACHE PATH "Path to openai-cpp repository")
set(WHISPER_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../whisper.cpp" CACHE PATH "Path to whisper.cpp repository")
set(PORTAUDIO_ROOT "/opt/local" CACHE PATH "PortAudio install prefix")
set(WHISPER_LIBRARY "${WHISPER_CPP_ROOT}/build/src/libwhisper.dylib" CACHE FILEPATH "Path to libwhisper shared library")

option(BUILD_TRANSCRIBE_AUDIO_ADAPTIVE "Build transcribe_audio_adaptive_energy executable" OFF)

find_package(CURL REQUIRED)
find_path(NLOHMANN_JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    HINTS
        "/usr/local/include"
        "/opt/homebrew/include"
        "${PORTAUDIO_ROOT}/include"
)

find_library(PORTAUDIO_LIBRARY
    NAMES portaudio
    HINTS "${PORTAUDIO_ROOT}/lib"
)

if(NOT PORTAUDIO_LIBRARY)
    message(FATAL_ERROR "PortAudio library not found. Set PORTAUDIO_ROOT or PORTAUDIO_LIBRARY.")
endif()

if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR "nlohmann/json.hpp not found. Install nlohmann-json and/or set NLOHMANN_JSON_INCLUDE_DIR.")
endif()

if(NOT EXISTS "${WHISPER_LIBRARY}")
    find_library(_WHISPER_FALLBACK NAMES whisper HINTS "${WHISPER_CPP_ROOT}/build/src")
    if(_WHISPER_FALLBACK)
        set(WHISPER_LIBRARY "${_WHISPER_FALLBACK}")
    else()
        message(FATAL_ERROR "Whisper library not found at ${WHISPER_LIBRARY}. Set WHISPER_LIBRARY.")
    endif()
endif()

add_executable(analyze_text analyze_text.cpp)
set_target_properties(analyze_text PROPERTIES OUTPUT_NAME "analyze_text.exe")

target_include_directories(analyze_text PRIVATE
    "${OPENAI_CPP_ROOT}/include/openai"
)

target_link_libraries(analyze_text PRIVATE CURL::libcurl)

add_executable(transcribe_audio transcribe_audio.cpp)
set_target_properties(transcribe_audio PROPERTIES OUTPUT_NAME "transcribe_audio.exe")

target_include_directories(transcribe_audio PRIVATE
    "${PORTAUDIO_ROOT}/include"
    "${WHISPER_CPP_ROOT}/include"
    "${WHISPER_CPP_ROOT}/ggml/include"
)

target_link_libraries(transcribe_audio PRIVATE
    "${PORTAUDIO_LIBRARY}"
    "${WHISPER_LIBRARY}"
)

set_target_properties(transcribe_audio PROPERTIES
    BUILD_RPATH "/usr/local/lib"
    INSTALL_RPATH "/usr/local/lib"
)

add_executable(deterministic_fhir_mapper deterministic_fhir_mapper.cpp)
set_target_properties(deterministic_fhir_mapper PROPERTIES OUTPUT_NAME "deterministic_fhir_mapper.exe")

target_include_directories(deterministic_fhir_mapper PRIVATE
    "${NLOHMANN_JSON_INCLUDE_DIR}"
)

if(BUILD_TRANSCRIBE_AUDIO_ADAPTIVE)
    add_executable(transcribe_audio_adaptive_energy transcribe_audio_adaptive_energy.cpp)
    set_target_properties(transcribe_audio_adaptive_energy PROPERTIES OUTPUT_NAME "transcribe_audio_adaptive_energy.exe")

    target_include_directories(transcribe_audio_adaptive_energy PRIVATE
        "${PORTAUDIO_ROOT}/include"
        "${WHISPER_CPP_ROOT}/include"
        "${WHISPER_CPP_ROOT}/ggml/include"
    )

    target_link_libraries(transcribe_audio_adaptive_energy PRIVATE
        "${PORTAUDIO_LIBRARY}"
        "${WHISPER_LIBRARY}"
    )

    set_target_properties(transcribe_audio_adaptive_energy PROPERTIES
        BUILD_RPATH "/usr/local/lib"
        INSTALL_RPATH "/usr/local/lib"
    )
endif()
