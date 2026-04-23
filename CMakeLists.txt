## SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)
project(OpenAIWhisperTools LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# Global C++ settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer Homebrew curl when available.
set(CURL_ROOT "/opt/homebrew/opt/curl" CACHE PATH "Homebrew curl prefix")
if(EXISTS "${CURL_ROOT}")
  list(PREPEND CMAKE_PREFIX_PATH "${CURL_ROOT}")
endif()

# ===== analyze_text.exe =====
add_executable(analyze_text analyze_text.cpp)

# Include path from your Makefile: -I ../openai-cpp/include/openai
target_include_directories(analyze_text PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../openai-cpp/include/openai
)

# Link against libcurl (prefer config if available, fallback to -lcurl)
find_package(CURL QUIET)
if(CURL_FOUND)
  target_link_libraries(analyze_text PRIVATE CURL::libcurl)
else()
  message(STATUS "CURL package not found; falling back to plain '-lcurl'")
  if(EXISTS "${CURL_ROOT}/include")
    target_include_directories(analyze_text PRIVATE "${CURL_ROOT}/include")
  endif()
  if(EXISTS "${CURL_ROOT}/lib")
    target_link_directories(analyze_text PRIVATE "${CURL_ROOT}/lib")
  endif()
  target_link_libraries(analyze_text PRIVATE curl)
endif()

# ===== transcribe_audio.exe =====
add_executable(transcribe_audio transcribe_audio.cpp)

# ===== server_transcribe_audio.exe =====
add_executable(server_transcribe_audio server_transcribe_audio.cpp)

# Cache variables let you adjust paths from cmake -D...=...
set(PORTAUDIO_INCLUDE "/opt/local/include" CACHE PATH "PortAudio include dir")
set(PORTAUDIO_LIB "/opt/local/lib/libportaudio.dylib" CACHE FILEPATH "PortAudio library")

set(WHISPER_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../whisper.cpp/include" CACHE PATH "whisper.cpp include dir")
set(GGML_INCLUDE    "${CMAKE_CURRENT_SOURCE_DIR}/../whisper.cpp/ggml/include" CACHE PATH "ggml include dir")
set(WHISPER_LIB     "${CMAKE_CURRENT_SOURCE_DIR}/../whisper.cpp/build/src/libwhisper.dylib" CACHE FILEPATH "whisper.cpp library")

target_include_directories(transcribe_audio PRIVATE
    "${PORTAUDIO_INCLUDE}"
    "${WHISPER_INCLUDE}"
    "${GGML_INCLUDE}"
)
target_include_directories(server_transcribe_audio PRIVATE
    "${PORTAUDIO_INCLUDE}"
    "${WHISPER_INCLUDE}"
    "${GGML_INCLUDE}"
)

# Link exactly the dylibs you used in the Makefile
target_link_libraries(transcribe_audio PRIVATE
    "${PORTAUDIO_LIB}"
    "${WHISPER_LIB}"
)
target_link_libraries(server_transcribe_audio PRIVATE
    "${PORTAUDIO_LIB}"
    "${WHISPER_LIB}"
)

# server_transcribe_audio also needs Boost.System + pthread.
find_package(Boost CONFIG QUIET COMPONENTS system)
if(NOT Boost_FOUND)
  find_package(Boost QUIET)
endif()
find_package(Threads QUIET)
if(Boost_FOUND)
  target_include_directories(server_transcribe_audio PRIVATE ${Boost_INCLUDE_DIRS})
endif()
if(TARGET Boost::system)
  target_link_libraries(server_transcribe_audio PRIVATE Boost::system)
elseif(Boost_SYSTEM_LIBRARY)
  target_link_libraries(server_transcribe_audio PRIVATE ${Boost_SYSTEM_LIBRARY})
elseif(Boost_FOUND)
  message(STATUS "Boost::system not found; building server_transcribe_audio without explicit boost_system link")
else()
  message(STATUS "Boost not found via CMake; ensure Boost headers are in compiler include paths")
endif()
if(Threads_FOUND)
  target_link_libraries(server_transcribe_audio PRIVATE Threads::Threads)
else()
  target_link_libraries(server_transcribe_audio PRIVATE pthread)
endif()

# Match your -rpath /usr/local/lib
# (On macOS this sets the runtime search path for the built binary.)
set_target_properties(transcribe_audio PROPERTIES
    BUILD_RPATH "/usr/local/lib"
)
set_target_properties(server_transcribe_audio PROPERTIES
    BUILD_RPATH "/usr/local/lib"
)

# ===== (optional) tidy outputs =====
# Place binaries in build/bin/
set_target_properties(analyze_text   PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_target_properties(transcribe_audio PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_target_properties(server_transcribe_audio PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ===== data_aggregator_analyzer.exe =====
add_executable(data_aggregator_analyzer data_aggregator_analyzer.cpp)
target_include_directories(data_aggregator_analyzer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../openai-cpp/include/openai
)
if(CURL_FOUND)
  target_link_libraries(data_aggregator_analyzer PRIVATE CURL::libcurl)
else()
  if(EXISTS "${CURL_ROOT}/include")
    target_include_directories(data_aggregator_analyzer PRIVATE "${CURL_ROOT}/include")
  endif()
  if(EXISTS "${CURL_ROOT}/lib")
    target_link_directories(data_aggregator_analyzer PRIVATE "${CURL_ROOT}/lib")
  endif()
  target_link_libraries(data_aggregator_analyzer PRIVATE curl)
endif()
set_target_properties(data_aggregator_analyzer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ===== deterministic_fhir_mapper.exe =====
add_executable(deterministic_fhir_mapper deterministic_fhir_mapper.cpp)
if(CURL_FOUND)
  target_link_libraries(deterministic_fhir_mapper PRIVATE CURL::libcurl)
else()
  if(EXISTS "${CURL_ROOT}/include")
    target_include_directories(deterministic_fhir_mapper PRIVATE "${CURL_ROOT}/include")
  endif()
  if(EXISTS "${CURL_ROOT}/lib")
    target_link_directories(deterministic_fhir_mapper PRIVATE "${CURL_ROOT}/lib")
  endif()
  target_link_libraries(deterministic_fhir_mapper PRIVATE curl)
endif()
set_target_properties(deterministic_fhir_mapper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
