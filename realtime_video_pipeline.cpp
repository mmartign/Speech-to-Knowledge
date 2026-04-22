// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Copyright (c) 2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
#include <opencv2/opencv.hpp>
#include <openai.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

//------------------------------------------------------------------------------
// Configuration and runtime option structures
//------------------------------------------------------------------------------

// OpenAI-related configuration loaded from the INI file.
struct OpenAIConfig {
    std::string baseUrl;
    std::string apiKey;
    std::string vmodelName;
};

// Command-line options with defaults chosen to match the original behavior.
struct ProgramOptions {
    std::string src;
    std::string configPath = "config.ini";

    double intervalSec = 10.0;
    int maxDim = 1024;
    int jpegQuality = 85;
    std::string prompt = "Analyze this frame.";
    bool guiEnabled = true;
    int reconnectSec = 5;

    // Optional explicit base datetime for media files.
    // If absent, we try media metadata creation_time, then application start.
    bool hasPredefinedStartTime = false;
    std::chrono::system_clock::time_point predefinedStartTime{};
};

// Single-slot job exchanged between the main thread and the inference worker.
//
// Design choice:
// - We keep only one pending job.
// - Newer frames overwrite older pending work.
// - This keeps the pipeline responsive for live streams and prevents latency
//   from growing without bound under slow inference.
struct PendingJob {
    cv::Mat frame;
    double wallTimeSec = 0.0;   // elapsed program time when the trigger fired
    double mediaPosSec = 0.0;   // position in the media timeline, if known
    int triggerIdx = 0;
    bool has = false;
    bool stop = false;
};

//------------------------------------------------------------------------------
// Utility helpers
//------------------------------------------------------------------------------

// Simple RAII joiner to ensure threads are joined on all exit paths.
class ThreadJoiner {
public:
    explicit ThreadJoiner(std::thread& t) noexcept : thread_(t) {}
    ThreadJoiner(const ThreadJoiner&) = delete;
    ThreadJoiner& operator=(const ThreadJoiner&) = delete;

    ~ThreadJoiner() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread& thread_;
};

// Print CLI help.
static void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " <video_or_stream_uri> [config.ini] [options]\n"
        << "Options:\n"
        << "  --interval <sec>        Prompt repetition interval in seconds (default 10)\n"
        << "  --max-dim <px>          Resize frames so max(width,height)<=px (default 1024)\n"
        << "  --jpeg-quality <1-100>  JPEG quality (default 85)\n"
        << "  --prompt <text>         Prompt prefix (default: \"Analyze this frame.\")\n"
        << "  --no-gui                Disable OpenCV imshow/waitKey\n"
        << "  --reconnect-sec <sec>   Reconnect window for live streams (default 5)\n"
        << "  --predefined_start_time \"YYYY-mm-dd HH:MM:SS\"\n"
        << "                          Override base datetime for media files\n";
}

// Trim leading and trailing whitespace in place.
// Returns false if the resulting string is empty.
static bool trimInPlace(std::string& s)
{
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        s.clear();
        return false;
    }

    const auto end = s.find_last_not_of(" \t\r\n");
    s = s.substr(start, end - start + 1);
    return true;
}

//------------------------------------------------------------------------------
// INI/config parsing
//------------------------------------------------------------------------------

// Small INI parser.
//
// Supported features:
// - [section] headers
// - key=value pairs
// - full-line comments starting with ';' or '#'
//
// Deliberately not supported:
// - quoted escaping rules
// - inline comments
// - multi-line values
//
// This keeps the parser predictable and avoids corrupting values containing
// '#' or ';', which the previous version could truncate accidentally.
static std::map<std::string, std::string> parseIni(const std::string& filename)
{
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    if (!file.is_open()) {
        return config;
    }

    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        if (!trimInPlace(line)) {
            continue;
        }

        if (line.empty()) {
            continue;
        }

        // Only treat comment markers as comments when they begin the line.
        // This is simpler and safer than trying to strip inline comments.
        if (line[0] == ';' || line[0] == '#') {
            continue;
        }

        // Section header: [openai]
        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            trimInPlace(section);
            continue;
        }

        const size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, eqPos);
        std::string value = line.substr(eqPos + 1);

        if (!trimInPlace(key)) {
            continue;
        }
        trimInPlace(value);

        // Flatten sections into "section.key".
        if (!section.empty()) {
            key = section + "." + key;
        }

        config[key] = value;
    }

    return config;
}

// Ensure base URLs end with '/' so downstream URL construction is consistent.
static std::string ensureTrailingSlash(const std::string& url)
{
    if (url.empty() || url.back() == '/') {
        return url;
    }
    return url + "/";
}

// Load and validate the required OpenAI config values.
static bool loadOpenAIConfig(const std::string& path, OpenAIConfig& cfg)
{
    const auto config = parseIni(path);

    auto getValue = [&](const std::string& key, std::string& out) {
        const auto it = config.find(key);
        if (it != config.end()) {
            out = it->second;
        }
    };

    getValue("openai.base_url", cfg.baseUrl);
    getValue("openai.api_key", cfg.apiKey);
    getValue("openai.vmodel_name", cfg.vmodelName);

    std::vector<std::string> missing;
    if (cfg.baseUrl.empty()) {
        missing.push_back("openai.base_url");
    }
    if (cfg.apiKey.empty()) {
        missing.push_back("openai.api_key");
    }
    if (cfg.vmodelName.empty()) {
        missing.push_back("openai.vmodel_name");
    }

    if (!missing.empty()) {
        std::cerr << "[ERROR] Missing config values in " << path << ":";
        for (const auto& key : missing) {
            std::cerr << ' ' << key;
        }
        std::cerr << "\n";
        return false;
    }

    cfg.baseUrl = ensureTrailingSlash(cfg.baseUrl);
    return true;
}

//------------------------------------------------------------------------------
// Encoding and API response parsing
//------------------------------------------------------------------------------

// Base64-encode a binary buffer.
//
// Used to embed a JPEG frame as a data URL in the API request body.
static std::string base64Encode(const std::vector<uchar>& data)
{
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < data.size()) {
        const uint32_t triple =
            (static_cast<uint32_t>(data[i]) << 16) |
            (static_cast<uint32_t>(data[i + 1]) << 8) |
            static_cast<uint32_t>(data[i + 2]);

        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(table[(triple >> 6) & 0x3F]);
        encoded.push_back(table[triple & 0x3F]);
        i += 3;
    }

    const size_t rem = data.size() - i;
    if (rem == 1) {
        const uint32_t triple = (static_cast<uint32_t>(data[i]) << 16);
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back('=');
        encoded.push_back('=');
    } else if (rem == 2) {
        const uint32_t triple =
            (static_cast<uint32_t>(data[i]) << 16) |
            (static_cast<uint32_t>(data[i + 1]) << 8);
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(table[(triple >> 6) & 0x3F]);
        encoded.push_back('=');
    }

    return encoded;
}

// Extract human-readable text from an API response.
//
// We support a few plausible shapes because deployments using "OpenAI-like"
// compatibility layers may not always return identical JSON structures.
static std::string extractMessageText(const json& response)
{
    const auto choicesIt = response.find("choices");
    if (choicesIt != response.end() && choicesIt->is_array() && !choicesIt->empty()) {
        const auto& first = (*choicesIt)[0];

        // Typical chat-completions shape:
        // choices[0].message.content
        const auto messageIt = first.find("message");
        if (messageIt != first.end()) {
            const auto contentIt = messageIt->find("content");
            if (contentIt != messageIt->end()) {
                if (contentIt->is_string()) {
                    return contentIt->get<std::string>();
                }
                if (contentIt->is_array()) {
                    std::string combined;
                    for (const auto& part : *contentIt) {
                        const auto textIt = part.find("text");
                        if (textIt != part.end() && textIt->is_string()) {
                            combined += textIt->get<std::string>();
                        }
                    }
                    if (!combined.empty()) {
                        return combined;
                    }
                }
            }
        }

        // Fallback shape sometimes seen in wrappers.
        const auto textIt = first.find("text");
        if (textIt != first.end() && textIt->is_string()) {
            return textIt->get<std::string>();
        }
    }

    // Additional defensive fallbacks.
    const auto outputTextIt = response.find("output_text");
    if (outputTextIt != response.end() && outputTextIt->is_string()) {
        return outputTextIt->get<std::string>();
    }

    const auto outputIt = response.find("output");
    if (outputIt != response.end() && outputIt->is_array()) {
        std::string combined;
        for (const auto& item : *outputIt) {
            const auto contentIt = item.find("content");
            if (contentIt == item.end() || !contentIt->is_array()) {
                continue;
            }
            for (const auto& part : *contentIt) {
                const auto textIt = part.find("text");
                if (textIt != part.end() && textIt->is_string()) {
                    combined += textIt->get<std::string>();
                }
            }
        }
        if (!combined.empty()) {
            return combined;
        }
    }

    return {};
}

//------------------------------------------------------------------------------
// Time helpers
//------------------------------------------------------------------------------

// Thread-safe localtime wrapper.
//
// std::localtime() is not thread-safe, so we use platform-specific safe forms.
static bool safeLocalTime(std::time_t t, std::tm& out)
{
#ifdef _WIN32
    return localtime_s(&out, &t) == 0;
#else
    return localtime_r(&t, &out) != nullptr;
#endif
}

// Format a system_clock time point as "[YYYY-mm-dd HH:MM:SS]".
static std::string formatDateTime(const std::chrono::system_clock::time_point& tp)
{
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    if (!safeLocalTime(t, tm)) {
        return "[invalid-local-time]";
    }

    std::ostringstream oss;
    oss << '[' << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << ']';
    return oss.str();
}

// Parse a local datetime string of the form "YYYY-mm-dd HH:MM:SS".
static bool parseDateTime(const std::string& s, std::chrono::system_clock::time_point& out)
{
    std::tm tm{};
    std::istringstream iss(s);
    iss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (iss.fail()) {
        return false;
    }

    const std::time_t tt = std::mktime(&tm);
    if (tt == static_cast<std::time_t>(-1)) {
        return false;
    }

    out = std::chrono::system_clock::from_time_t(tt);
    return true;
}

// Normalize metadata datetime variants (T/Z/fractional/tz suffixes) then parse.
static bool parseMetadataDateTime(
    std::string value,
    std::chrono::system_clock::time_point& out)
{
    if (!trimInPlace(value)) {
        return false;
    }

    std::replace(value.begin(), value.end(), 'T', ' ');
    if (!value.empty() && (value.back() == 'Z' || value.back() == 'z')) {
        value.pop_back();
    }

    const size_t fracPos = value.find('.');
    if (fracPos != std::string::npos) {
        value.erase(fracPos);
    }

    const size_t tzPos = value.find_first_of("+-", 19);
    if (tzPos != std::string::npos) {
        value.erase(tzPos);
    }

    if (!trimInPlace(value)) {
        return false;
    }
    if (value.size() < 19) {
        return false;
    }

    value = value.substr(0, 19);
    return parseDateTime(value, out);
}

// Probe ffprobe start_time_realtime (microseconds since epoch) for media files.
static std::optional<std::chrono::system_clock::time_point>
probeFileEncodedTimelineStart(const std::string& path)
{
#ifdef _WIN32
    const std::string cmd =
        "ffprobe -v error -show_entries format=start_time_realtime:stream=start_time_realtime "
        "-of default=noprint_wrappers=1:nokey=1 \"" + path + "\" 2>nul";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    const std::string cmd =
        "ffprobe -v error -show_entries format=start_time_realtime:stream=start_time_realtime "
        "-of default=noprint_wrappers=1:nokey=1 \"" + path + "\" 2>/dev/null";
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
        if (!trimInPlace(line)) {
            continue;
        }
        try {
            const long long micros = std::stoll(line);
            if (micros <= 0) {
                continue;
            }
            return std::chrono::system_clock::time_point{
                std::chrono::microseconds(micros)};
        } catch (...) {
        }
    }
    return std::nullopt;
}

// Probe ffprobe creation_time tags and parse to local system_clock time.
static std::optional<std::chrono::system_clock::time_point>
probeFileEncodedStartTime(const std::string& path)
{
#ifdef _WIN32
    const std::string cmd =
        "ffprobe -v error -show_entries "
        "format_tags=creation_time:stream_tags=creation_time "
        "-of default=noprint_wrappers=1:nokey=1 \"" + path + "\" 2>nul";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    const std::string cmd =
        "ffprobe -v error -show_entries "
        "format_tags=creation_time:stream_tags=creation_time "
        "-of default=noprint_wrappers=1:nokey=1 \"" + path + "\" 2>/dev/null";
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
        if (parseMetadataDateTime(line, parsed)) {
            return parsed;
        }
    }
    return std::nullopt;
}

// Add fractional seconds to a time point.
static std::chrono::system_clock::time_point addSecondsToTimePoint(
    const std::chrono::system_clock::time_point& base,
    double sec)
{
    return base + std::chrono::duration_cast<std::chrono::system_clock::duration>(
                      std::chrono::duration<double>(sec));
}

//------------------------------------------------------------------------------
// Frame/image helpers
//------------------------------------------------------------------------------

// Resize a frame so that max(width, height) <= maxDim, preserving aspect ratio.
// If maxDim <= 0, resizing is disabled.
static cv::Mat resizeMaxDim(const cv::Mat& frame, int maxDim)
{
    if (maxDim <= 0 || frame.empty()) {
        return frame;
    }

    const int w = frame.cols;
    const int h = frame.rows;
    const int m = (w > h) ? w : h;
    if (m <= maxDim) {
        return frame;
    }

    const double scale = static_cast<double>(maxDim) / static_cast<double>(m);
    const int nw = std::max(1, static_cast<int>(std::lround(w * scale)));
    const int nh = std::max(1, static_cast<int>(std::lround(h * scale)));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    return resized;
}

//------------------------------------------------------------------------------
// Strict numeric parsing helpers
//------------------------------------------------------------------------------

// Parse a double and require that the whole string is consumed.
static bool parseDoubleStrict(const std::string& s, double& out)
{
    try {
        size_t idx = 0;
        out = std::stod(s, &idx);
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

// Parse an int and require that the whole string is consumed.
static bool parseIntStrict(const std::string& s, int& out)
{
    try {
        size_t idx = 0;
        out = std::stoi(s, &idx);
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

// Return true if the entire string is composed of decimal digits.
static bool isUnsignedIntegerString(const std::string& s)
{
    if (s.empty()) {
        return false;
    }
    for (unsigned char ch : s) {
        if (!std::isdigit(ch)) {
            return false;
        }
    }
    return true;
}

// Treat any non-empty unsigned integer as a camera index.
//
// This is more useful than the original single-digit check because device
// indexes such as "10" should still work.
static bool isCameraIndexSource(const std::string& s)
{
    return isUnsignedIntegerString(s);
}

// Open either a camera index or a URI/file path.
static bool openCapture(cv::VideoCapture& cap, const std::string& src)
{
    if (isCameraIndexSource(src)) {
        try {
            const int index = std::stoi(src);
            return cap.open(index);
        } catch (...) {
            return false;
        }
    }
    return cap.open(src);
}

//------------------------------------------------------------------------------
// OpenAI request
//------------------------------------------------------------------------------

// Encode a frame as JPEG, wrap it into a data URL, and send it to the model.
//
// We include both:
// - wall time: how long the program has been running
// - media position: where the frame sits in the media timeline
//
// Keeping those separate matters for offline file playback, where media time
// should not drift if inference becomes slower or faster.
static bool sendFrameToOpenAI(
    const cv::Mat& frame,
    double wallTimeSec,
    double mediaPosSec,
    int triggerIdx,
    const OpenAIConfig& cfg,
    const std::string& prompt,
    int maxDim,
    int jpegQuality)
{
    cv::Mat resized = resizeMaxDim(frame, maxDim);

    std::vector<uchar> buffer;
    std::vector<int> params;
    if (jpegQuality > 0 && jpegQuality <= 100) {
        params = {cv::IMWRITE_JPEG_QUALITY, jpegQuality};
    }

    if (!cv::imencode(".jpg", resized, buffer, params)) {
        std::cerr << "[ERROR] Interval #" << triggerIdx
                  << " failed to encode frame to JPEG\n";
        return false;
    }

    const std::string dataUrl = "data:image/jpeg;base64," + base64Encode(buffer);

    std::ostringstream promptStream;
    promptStream
        << prompt
        << " Wall time: " << std::fixed << std::setprecision(3) << wallTimeSec << "s;"
        << " media position: " << std::fixed << std::setprecision(3) << mediaPosSec << "s;"
        << " interval #" << triggerIdx;

    json body = {
        {"model", cfg.vmodelName},
        {"messages", json::array({
            {
                {"role", "user"},
                {"content", json::array({
                    {{"type", "text"}, {"text", promptStream.str()}},
                    {{"type", "image_url"}, {"image_url", {{"url", dataUrl}}}}
                })}
            }
        })},
        {"stream", false}
    };

    try {
        auto chat = openai::chat().create(body);
        const std::string message = extractMessageText(chat);
        if (!message.empty()) {
            std::cout << message << std::endl;
        } else {
            std::cout << "(no text content)" << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] OpenAI request failed for interval #"
                  << triggerIdx << ": " << e.what() << "\n";
        return false;
    }
}

//------------------------------------------------------------------------------
// CLI parsing
//------------------------------------------------------------------------------

// Parse command-line arguments into ProgramOptions.
//
// Design goals:
// - preserve the original CLI shape
// - validate values with clear error messages
// - avoid uncaught exceptions from stod/stoi
static bool parseCommandLine(int argc, char** argv, ProgramOptions& opt)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return false;
    }

    opt.src = argv[1];

    // Optional positional config path:
    //   program <src> config.ini --interval 10
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
        opt.configPath = argv[2];
    }

    int argi = 2;
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
        argi = 3;
    }

    auto needValue = [&](const char* name) -> std::optional<std::string> {
        if (argi + 1 >= argc) {
            std::cerr << "[ERROR] Missing value for " << name << "\n";
            return std::nullopt;
        }
        return std::string(argv[++argi]);
    };

    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];

        if (a == "--interval") {
            auto v = needValue("--interval");
            if (!v) return false;

            double parsed = 0.0;
            if (!parseDoubleStrict(*v, parsed) || !std::isfinite(parsed) || parsed <= 0.0) {
                std::cerr << "[ERROR] --interval must be a positive number\n";
                return false;
            }
            opt.intervalSec = std::max(0.1, parsed);
        } else if (a == "--max-dim") {
            auto v = needValue("--max-dim");
            if (!v) return false;

            int parsed = 0;
            if (!parseIntStrict(*v, parsed) || parsed < 0) {
                std::cerr << "[ERROR] --max-dim must be an integer >= 0\n";
                return false;
            }
            opt.maxDim = parsed;
        } else if (a == "--jpeg-quality") {
            auto v = needValue("--jpeg-quality");
            if (!v) return false;

            int parsed = 0;
            if (!parseIntStrict(*v, parsed) || parsed < 1 || parsed > 100) {
                std::cerr << "[ERROR] --jpeg-quality must be 1..100\n";
                return false;
            }
            opt.jpegQuality = parsed;
        } else if (a == "--predefined_start_time") {
            auto v = needValue("--predefined_start_time");
            if (!v) return false;

            if (!parseDateTime(*v, opt.predefinedStartTime)) {
                std::cerr << "[ERROR] --predefined_start_time expects \"YYYY-mm-dd HH:MM:SS\"\n";
                return false;
            }
            opt.hasPredefinedStartTime = true;
        } else if (a == "--prompt") {
            auto v = needValue("--prompt");
            if (!v) return false;
            opt.prompt = *v;
        } else if (a == "--no-gui") {
            opt.guiEnabled = false;
        } else if (a == "--reconnect-sec") {
            auto v = needValue("--reconnect-sec");
            if (!v) return false;

            int parsed = 0;
            if (!parseIntStrict(*v, parsed) || parsed < 0) {
                std::cerr << "[ERROR] --reconnect-sec must be an integer >= 0\n";
                return false;
            }
            opt.reconnectSec = parsed;
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "[ERROR] Unknown option: " << a << "\n";
            printUsage(argv[0]);
            return false;
        }
    }

    return true;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

// Entry point wiring capture, scheduling, and inference worker threads.
int main(int argc, char** argv)
{
    ProgramOptions options;
    if (!parseCommandLine(argc, argv, options)) {
        return 1;
    }

    OpenAIConfig cfg;
    if (!loadOpenAIConfig(options.configPath, cfg)) {
        return 1;
    }

    // Log the effective non-secret configuration for easier troubleshooting.
    std::cerr << "[INFO] OpenAI base URL: " << cfg.baseUrl << "\n";
    std::cerr << "[INFO] Vision model: " << cfg.vmodelName << "\n";
    std::cerr << "[INFO] Source: " << options.src << "\n";

    try {
        openai::start(cfg.apiKey, "", true, cfg.baseUrl);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize OpenAI client: "
                  << e.what() << "\n";
        return 1;
    }

    cv::VideoCapture cap;
    if (!openCapture(cap, options.src) || !cap.isOpened()) {
        std::cerr << "[ERROR] Could not open source\n";
        return 1;
    }

    // Request a small internal buffer to reduce lag on live sources.
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // Heuristic:
    // - if frame count is finite and > 0, and the source is not a camera index,
    //   treat it as a media file
    // - otherwise assume a live-ish source
    const double frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
    const bool likelyFile =
        std::isfinite(frameCount) &&
        frameCount > 0.0 &&
        !isCameraIndexSource(options.src);

    const auto applicationStartTime = std::chrono::system_clock::now();

    // Base timestamp used to map media position -> absolute datetime.
    //
    // Priority:
    // 1. explicit --predefined_start_time
    // 2. encoded timeline start (start_time_realtime)
    // 3. encoded start from media metadata (creation_time)
    // 4. application start time
    std::chrono::system_clock::time_point fileBaseTime =
        applicationStartTime;

    if (likelyFile) {
        if (options.hasPredefinedStartTime) {
            fileBaseTime = options.predefinedStartTime;
        } else {
            const auto timeline = probeFileEncodedTimelineStart(options.src);
            if (timeline.has_value()) {
                fileBaseTime = *timeline;
            } else {
                const auto probed = probeFileEncodedStartTime(options.src);
                if (probed.has_value()) {
                    fileBaseTime = *probed;
                }
            }
        }
    }

    // Shared latest frame state.
    //
    // The capture thread continuously updates this.
    // The main loop periodically samples it and overwrites the single pending
    // inference job.
    std::mutex frameMtx;
    cv::Mat latestFrame;
    double latestMediaPosSec = 0.0;

    // Shared job state for the worker thread.
    std::mutex jobMtx;
    std::condition_variable jobCv;
    PendingJob pending;

    std::atomic<bool> running{true};

    //--------------------------------------------------------------------------
    // Worker thread
    //
    // Responsibilities:
    // - wait for a pending job
    // - process only the newest job
    // - send frame to the model
    //
    // Important:
    // For file playback, encoded timestamps are derived from mediaPosSec,
    // not wallTimeSec. This avoids drift when inference is slower than real
    // time or when file playback timing varies.
    //--------------------------------------------------------------------------
    std::thread worker([&] {
        while (true) {
            PendingJob job;
            {
                std::unique_lock<std::mutex> lk(jobMtx);
                jobCv.wait(lk, [&] { return pending.stop || pending.has; });

                if (pending.stop) {
                    break;
                }

                job.wallTimeSec = pending.wallTimeSec;
                job.mediaPosSec = pending.mediaPosSec;
                job.triggerIdx = pending.triggerIdx;
                pending.frame.copyTo(job.frame);

                // Clear the pending slot immediately so that newer work can be
                // scheduled while inference is running.
                pending.has = false;
            }

            if (job.frame.empty()) {
                continue;
            }

            const std::string acquisitionTag = formatDateTime(
                addSecondsToTimePoint(applicationStartTime, job.wallTimeSec));

            std::string mediaTag;
            if (likelyFile) {
                mediaTag = formatDateTime(
                    addSecondsToTimePoint(fileBaseTime, job.mediaPosSec));
            } else {
                mediaTag = acquisitionTag;
            }

            const bool useEncodedTimelineTag = likelyFile;
            const std::string& logTimestamp = useEncodedTimelineTag
                                                  ? mediaTag
                                                  : acquisitionTag;

            // For files we log encoded media timeline time; for live sources we
            // fall back to acquisition time because no stable encoded timeline exists.
            std::cout << logTimestamp
                      << " media-time=" << std::fixed << std::setprecision(3)
                      << job.mediaPosSec << "s";
            if (!useEncodedTimelineTag) {
                std::cout << " encoded-at=" << mediaTag;
            }
            std::cout << "  ";

            sendFrameToOpenAI(
                job.frame,
                job.wallTimeSec,
                job.mediaPosSec,
                job.triggerIdx,
                cfg,
                options.prompt,
                options.maxDim,
                options.jpegQuality);
        }
    });
    ThreadJoiner workerJoiner(worker);

    //--------------------------------------------------------------------------
    // Capture thread
    //
    // Responsibilities:
    // - continuously read frames from OpenCV
    // - keep only the latest frame
    // - for files, pace playback approximately in real time
    // - for streams, attempt reconnects on transient failures
    //
    // Notes:
    // - For media files, we prefer CAP_PROP_POS_MSEC as the media timeline.
    // - If POS_MSEC is unavailable, we fall back to FPS-derived progression.
    //--------------------------------------------------------------------------
    std::thread captureThread([&] {
        cv::Mat f;
        auto lastOk = std::chrono::steady_clock::now();

        const double fps = cap.get(cv::CAP_PROP_FPS);
        const bool hasValidFps = std::isfinite(fps) && fps > 1e-6;
        const double frameDuration = hasValidFps ? (1.0 / fps) : 0.0;

        const auto playbackStart = std::chrono::steady_clock::now();
        double fallbackPosSec = 0.0;
        int reconnectAttempt = 0;

        while (running.load()) {
            if (cap.read(f) && !f.empty()) {
                reconnectAttempt = 0;
                lastOk = std::chrono::steady_clock::now();

                double mediaPosSec = fallbackPosSec;

                if (likelyFile) {
                    const double posMsec = cap.get(cv::CAP_PROP_POS_MSEC);

                    if (std::isfinite(posMsec) && posMsec >= 1e-3) {
                        mediaPosSec = posMsec / 1000.0;
                        fallbackPosSec = mediaPosSec;
                    } else if (hasValidFps) {
                        fallbackPosSec += frameDuration;
                        mediaPosSec = fallbackPosSec;
                    }

                    // Throttle file reading to approximately real-time playback
                    // based on media position, not on loop speed.
                    if (mediaPosSec > 0.0) {
                        const auto targetTime =
                            playbackStart + std::chrono::duration<double>(mediaPosSec);
                        const auto now = std::chrono::steady_clock::now();
                        if (targetTime > now) {
                            std::this_thread::sleep_for(targetTime - now);
                        }
                    }
                } else {
                    // For streams/cameras, media position is less meaningful.
                    // Use POS_MSEC only if the backend provides something sane.
                    const double posMsec = cap.get(cv::CAP_PROP_POS_MSEC);
                    if (std::isfinite(posMsec) && posMsec >= 0.0) {
                        mediaPosSec = posMsec / 1000.0;
                    } else {
                        mediaPosSec = 0.0;
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(frameMtx);
                    f.copyTo(latestFrame);
                    latestMediaPosSec = mediaPosSec;
                }
                continue;
            }

            // For files, EOF/failure is expected termination.
            if (likelyFile) {
                running.store(false);
                break;
            }

            // For streams/cameras, treat failures as transient up to a limit.
            const auto now = std::chrono::steady_clock::now();
            const double downFor =
                std::chrono::duration<double>(now - lastOk).count();

            if (options.reconnectSec <= 0 ||
                downFor > static_cast<double>(options.reconnectSec)) {
                std::cerr << "[ERROR] Stream read failed for >"
                          << options.reconnectSec << "s; stopping.\n";
                running.store(false);
                break;
            }

            // Bounded exponential backoff:
            // 250, 500, 1000, 2000, 2000, ...
            ++reconnectAttempt;
            const int backoffMs =
                std::min(2000, 250 * (1 << std::min(reconnectAttempt - 1, 3)));

            std::cerr << "[WARN] Stream read failed; attempting reconnect in "
                      << backoffMs << " ms...\n";

            cap.release();
            std::this_thread::sleep_for(std::chrono::milliseconds(backoffMs));

            if (!running.load()) {
                break;
            }

            if (!openCapture(cap, options.src) || !cap.isOpened()) {
                continue;
            }

            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            lastOk = std::chrono::steady_clock::now();
        }
    });
    ThreadJoiner captureJoiner(captureThread);

    //--------------------------------------------------------------------------
    // Main scheduling loop
    //
    // Responsibilities:
    // - periodically sample the newest available frame
    // - overwrite the pending single-slot inference job
    // - optionally display the current frame in a GUI window
    //--------------------------------------------------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    double nextTrigger = 0.0;
    int triggerIdx = 0;

    while (running.load()) {
        const auto tNow = std::chrono::steady_clock::now();
        const double wallSec = std::chrono::duration<double>(tNow - t0).count();

        // Fire at fixed intervals.
        //
        // We overwrite the pending job rather than queueing indefinitely,
        // because freshness matters more than completeness for this workload.
        if (wallSec >= nextTrigger) {
            cv::Mat frameCopy;
            double mediaPosSec = 0.0;

            {
                std::lock_guard<std::mutex> lock(frameMtx);
                if (!latestFrame.empty()) {
                    latestFrame.copyTo(frameCopy);
                }
                mediaPosSec = latestMediaPosSec;
            }

            if (!frameCopy.empty()) {
                {
                    std::lock_guard<std::mutex> lk(jobMtx);
                    pending.frame = frameCopy;
                    pending.wallTimeSec = wallSec;
                    pending.mediaPosSec = mediaPosSec;
                    pending.triggerIdx = triggerIdx++;
                    pending.has = true;
                }
                jobCv.notify_one();

                // If the main loop was delayed, catch up by advancing the next
                // trigger beyond the current wall time.
                while (wallSec >= nextTrigger) {
                    nextTrigger += options.intervalSec;
                }
            }
        }

        // Optional local preview window.
        //
        // This is deliberately non-blocking: waitKey(1) allows GUI events to be
        // processed without stalling capture/inference scheduling.
        if (options.guiEnabled) {
            cv::Mat toShow;
            {
                std::lock_guard<std::mutex> lock(frameMtx);
                if (!latestFrame.empty()) {
                    latestFrame.copyTo(toShow);
                }
            }

            if (!toShow.empty()) {
                cv::imshow("Live", toShow);
                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {
                    running.store(false);
                }
            }
        }

        // Small sleep to avoid a busy-spin main loop.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    //--------------------------------------------------------------------------
    // Shutdown
    //
    // Signal the worker explicitly instead of faking a pending job.
    //--------------------------------------------------------------------------
    {
        std::lock_guard<std::mutex> lk(jobMtx);
        pending.stop = true;
        pending.has = false;
    }
    jobCv.notify_one();

    cap.release();

    if (options.guiEnabled) {
        cv::destroyAllWindows();
    }

    return 0;
}
