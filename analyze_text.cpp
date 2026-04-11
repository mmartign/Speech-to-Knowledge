// -*- coding: utf-8 -*-
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// Copyright (C) 2025 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <cctype>
#include <openai.hpp>

using json = nlohmann::json;

// Global config
std::string OPENWEBUI_URL;
std::string API_KEY;
std::string MODEL_NAME;
std::string KNOWLEDGE_BASE_IDS;
std::string PROMPT;
std::string TEMP_PROMPT;
std::string TRIGGER_START;
std::string TRIGGER_STOP;
std::string TRIGGER_TEMP_CHECK;
std::string TTS_COMMAND;

std::mutex analysis_mutex;
std::atomic<int> counter_value{0};
std::atomic<int> temp_counter_value{0};
std::atomic<int> active_analyses{0};
std::mutex tts_mutex;
std::once_flag openai_init_flag;

std::string escape_for_single_quotes(const std::string& text);

class AnalysisSession {
public:
    AnalysisSession(std::mutex& mutex, std::atomic<int>& active_count)
        : lock_(mutex), active_count_(active_count) {
        ++active_count_;
    }

    ~AnalysisSession() {
        --active_count_;
    }

private:
    std::unique_lock<std::mutex> lock_;
    std::atomic<int>& active_count_;
};

std::string strip_trailing_newlines(std::string text) {
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

std::string trim_whitespace(std::string text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

bool starts_with_unused_tag_at(const std::string& text, size_t pos) {
    constexpr const char* prefix = "<unused";
    constexpr size_t prefix_len = 7;
    if (pos + prefix_len >= text.size() || text.compare(pos, prefix_len, prefix) != 0) {
        return false;
    }
    size_t i = pos + prefix_len;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    return i < text.size() && text[i] == '>';
}

std::string strip_internal_reasoning_tags(std::string text) {
    size_t cursor = 0;
    while (cursor < text.size()) {
        size_t tag_pos = text.find("<unused", cursor);
        if (tag_pos == std::string::npos) {
            break;
        }
        if (!starts_with_unused_tag_at(text, tag_pos)) {
            cursor = tag_pos + 1;
            continue;
        }

        size_t thought_pos = text.find("thought", tag_pos);
        if (thought_pos == std::string::npos || thought_pos > tag_pos + 48) {
            cursor = tag_pos + 1;
            continue;
        }

        const size_t next_tag = text.find("<unused", thought_pos);
        if (next_tag == std::string::npos || !starts_with_unused_tag_at(text, next_tag)) {
            text.erase(tag_pos);
            break;
        }

        text.erase(tag_pos, next_tag - tag_pos);
        cursor = tag_pos;
    }

    while (true) {
        size_t tag_pos = text.find("<unused");
        if (tag_pos == std::string::npos || !starts_with_unused_tag_at(text, tag_pos)) {
            break;
        }
        size_t end = text.find('>', tag_pos);
        if (end == std::string::npos) {
            break;
        }
        text.erase(tag_pos, (end - tag_pos) + 1);
    }

    return trim_whitespace(text);
}

bool is_fhir_bundle_object(const json& candidate) {
    if (!candidate.is_object()) {
        return false;
    }
    const auto type_it = candidate.find("resourceType");
    if (type_it == candidate.end() || !type_it->is_string()) {
        return false;
    }
    return type_it->get<std::string>() == "Bundle";
}

bool extract_fhir_bundle_from_text(const std::string& text, json& bundle, size_t& start_pos, size_t& end_pos) {
    bool in_string = false;
    bool escaped = false;

    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];

        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }

        if (c != '{') {
            continue;
        }

        size_t depth = 0;
        bool local_in_string = false;
        bool local_escaped = false;
        bool completed = false;

        for (size_t j = i; j < text.size(); ++j) {
            const char cj = text[j];
            if (local_in_string) {
                if (local_escaped) {
                    local_escaped = false;
                } else if (cj == '\\') {
                    local_escaped = true;
                } else if (cj == '"') {
                    local_in_string = false;
                }
                continue;
            }

            if (cj == '"') {
                local_in_string = true;
                continue;
            }
            if (cj == '{') {
                ++depth;
            } else if (cj == '}') {
                if (depth == 0) {
                    break;
                }
                --depth;
                if (depth == 0) {
                    const std::string candidate = text.substr(i, j - i + 1);
                    try {
                        json parsed = json::parse(candidate);
                        if (is_fhir_bundle_object(parsed)) {
                            bundle = std::move(parsed);
                            start_pos = i;
                            end_pos = j + 1;
                            return true;
                        }
                    } catch (...) {
                        // Keep searching for the next candidate JSON object.
                    }
                    completed = true;
                    break;
                }
            }
        }

        if (!completed) {
            break;
        }
    }

    return false;
}

bool extract_revised_bundle(const json& mapper_output, json& revised_bundle) {
    if (is_fhir_bundle_object(mapper_output)) {
        revised_bundle = mapper_output;
        return true;
    }

    const auto accepted_it = mapper_output.find("acceptedBundle");
    if (accepted_it != mapper_output.end() && is_fhir_bundle_object(*accepted_it)) {
        revised_bundle = *accepted_it;
        return true;
    }

    return false;
}

std::string revise_fhir_bundle_in_response(std::string response_text,
                                           const std::string& analysis_label,
                                           std::ofstream& file) {
    json detected_bundle;
    size_t start_pos = 0;
    size_t end_pos = 0;
    if (!extract_fhir_bundle_from_text(response_text, detected_bundle, start_pos, end_pos)) {
        return response_text;
    }

    const std::string input_path = "tmp_mapper_input_" + analysis_label + ".json";
    const std::string output_path = "tmp_mapper_output_" + analysis_label + ".json";

    try {
        std::ofstream input_file(input_path);
        if (!input_file.is_open()) {
            file << "\n[WARN] Detected FHIR Bundle but failed to open mapper input file: "
                 << input_path << "\n";
            return response_text;
        }
        input_file << detected_bundle.dump(2) << "\n";
    } catch (const std::exception& e) {
        file << "\n[WARN] Failed to write mapper input bundle: " << e.what() << "\n";
        return response_text;
    }

    const std::string cmd = "./deterministic_fhir_mapper.exe '" + escape_for_single_quotes(input_path) +
                            "' '" + escape_for_single_quotes(output_path) + "' >/dev/null 2>&1";
    const int mapper_rc = std::system(cmd.c_str());
    if (mapper_rc != 0) {
        file << "\n[WARN] deterministic_fhir_mapper returned non-zero status (" << mapper_rc
             << "). Keeping original bundle.\n";
        std::remove(input_path.c_str());
        std::remove(output_path.c_str());
        return response_text;
    }

    try {
        std::ifstream output_file(output_path);
        if (!output_file.is_open()) {
            file << "\n[WARN] Mapper output file not found: " << output_path
                 << ". Keeping original bundle.\n";
            std::remove(input_path.c_str());
            std::remove(output_path.c_str());
            return response_text;
        }

        json mapper_output;
        output_file >> mapper_output;

        json revised_bundle;
        if (!extract_revised_bundle(mapper_output, revised_bundle)) {
            file << "\n[WARN] Mapper output did not contain a revised Bundle. Keeping original bundle.\n";
            std::remove(input_path.c_str());
            std::remove(output_path.c_str());
            return response_text;
        }

        const std::string revised_text = revised_bundle.dump(2);
        response_text.replace(start_pos, end_pos - start_pos, revised_text);
        file << "\n[INFO] FHIR Bundle detected and revised by deterministic_fhir_mapper.\n";
    } catch (const std::exception& e) {
        file << "\n[WARN] Failed to parse mapper output: " << e.what()
             << ". Keeping original bundle.\n";
    }

    std::remove(input_path.c_str());
    std::remove(output_path.c_str());
    return response_text;
}

std::string escape_for_single_quotes(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() * 2);
    for (char c : text) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(c);
        }
    }
    return escaped;
}

void speak_text(const std::string& text) {
    std::string trimmed = strip_trailing_newlines(text);
    if (trimmed.empty()) {
        return;
    }

    trimmed = "Announciator: " + trimmed;

    const std::string escaped = escape_for_single_quotes(trimmed);
    const std::string cmd = TTS_COMMAND + " '" + escaped + "' >/dev/null 2>&1 &";

    std::lock_guard<std::mutex> lock(tts_mutex);
    std::system(cmd.c_str());
}

void say_info(const std::string& message) {
    std::cout << message;
    speak_text(message);
}

void say_error(const std::string& message) {
    std::cerr << message;
    speak_text(message);
}

// Simple INI parser
std::map<std::string, std::string> parse_ini(const std::string& filename) {
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    std::string line, section;

    while (std::getline(file, line)) {
        // Remove comments
        size_t comment_pos = line.find_first_of(";#");
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
        } else {
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                key.erase(0, key.find_first_not_of(" \t\r\n"));
                key.erase(key.find_last_not_of(" \t\r\n") + 1);
                value.erase(0, value.find_first_not_of(" \t\r\n"));
                value.erase(value.find_last_not_of(" \t\r\n") + 1);
                config[section + "." + key] = value;
            }
        }
    }

    return config;
}

// Load config
bool load_config(const std::string& path) {
    std::ifstream file_check(path);
    if (!file_check.is_open()) {
        say_error("Unable to open config file: " + path + "\n");
        return false;
    }
    file_check.close();

    auto config = parse_ini(path);

    std::vector<std::string> missing_keys;
    auto require_value = [&](const std::string& key, std::string& destination) {
        auto it = config.find(key);
        if (it == config.end() || it->second.empty()) {
            missing_keys.push_back(key);
            return;
        }
        destination = it->second;
    };

    require_value("openai.base_url", OPENWEBUI_URL);
    require_value("openai.api_key", API_KEY);
    require_value("openai.model_name", MODEL_NAME);
    require_value("prompts.prompt", PROMPT);
    require_value("prompts.temp_prompt", TEMP_PROMPT);
    require_value("triggers.start", TRIGGER_START);
    require_value("triggers.stop", TRIGGER_STOP);
    require_value("triggers.temp_check", TRIGGER_TEMP_CHECK);
    require_value("tts.command", TTS_COMMAND);

    auto kb_it = config.find("analysis.knowledge_base_ids");
    KNOWLEDGE_BASE_IDS = (kb_it != config.end()) ? kb_it->second : std::string{};

    if (!missing_keys.empty()) {
        std::ostringstream oss;
        oss << "Missing required config values:";
        for (const auto& key : missing_keys) {
            oss << ' ' << key;
        }
        oss << "\n";
        say_error(oss.str());
        return false;
    }

    std::transform(TRIGGER_START.begin(), TRIGGER_START.end(), TRIGGER_START.begin(), ::tolower);
    std::transform(TRIGGER_STOP.begin(), TRIGGER_STOP.end(), TRIGGER_STOP.begin(), ::tolower);
    std::transform(TRIGGER_TEMP_CHECK.begin(), TRIGGER_TEMP_CHECK.end(), TRIGGER_TEMP_CHECK.begin(), ::tolower);

    if (KNOWLEDGE_BASE_IDS.empty()) {
        say_error("Warning: analysis.knowledge_base_ids is not set; knowledge base lookups will be skipped.\n");
    }

    return true;
}

// Substring check
bool contains_substring(const std::string& str, const std::string& sub) {
    return sub.empty() || str.find(sub) != std::string::npos;
}

// Safely extract a textual message content from an OpenAI-style response
std::string extract_message_content(const json& response) {
    const auto choices_it = response.find("choices");
    if (choices_it == response.end() || !choices_it->is_array() || choices_it->empty()) {
        return {};
    }

    const auto& first_choice = (*choices_it)[0];
    if (!first_choice.is_object()) {
        return {};
    }

    const auto message_it = first_choice.find("message");
    if (message_it == first_choice.end() || !message_it->is_object()) {
        return {};
    }

    const auto content_it = message_it->find("content");
    if (content_it == message_it->end()) {
        return {};
    }

    if (content_it->is_string()) {
        return trim_whitespace(content_it->get<std::string>());
    }

    if (content_it->is_array()) {
        std::string combined;
        for (const auto& part : *content_it) {
            std::string part_text;
            if (part.is_string()) {
                part_text = part.get<std::string>();
            } else if (part.is_object()) {
                const auto text_it = part.find("text");
                if (text_it != part.end() && text_it->is_string()) {
                    part_text = text_it->get<std::string>();
                } else {
                    const auto content_it2 = part.find("content");
                    if (content_it2 != part.end() && content_it2->is_string()) {
                        part_text = content_it2->get<std::string>();
                    }
                }
            }
            if (part_text.empty()) {
                continue;
            }
            if (!combined.empty()) {
                combined.push_back('\n');
            }
            combined += part_text;
        }
        return trim_whitespace(combined);
    }

    return {};
}

// AI analysis with fresh context for each request
void analyze_text(const std::string& text) {
    AnalysisSession session(analysis_mutex, active_analyses);
    const int analysis_id = ++counter_value;
    temp_counter_value = 0; // Reset temp counter for each main analysis
    say_info("Analysis of Recording[" + std::to_string(analysis_id) + "] Started ------------------->>>\n");

    const std::string filename = "results_analysis" + std::to_string(analysis_id) + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        say_error("[ERROR] Unable to open results file: " + filename + "\n");
        say_info("Analysis of Recording[" + std::to_string(analysis_id) + "] Finished ------------------->>>\n");
        return;
    }

    file << "Using model: " << MODEL_NAME << "\n";
    file << "Endpoint: " << OPENWEBUI_URL << "\n";
    file << "Prompt: " << PROMPT << "\n" << text << "\n";

    if (!file) {
        say_error("[ERROR] Failed to write analysis header to " + filename + "\n");
    }

    std::string response_string;

    try {
        openai::start({
            API_KEY
        });

        json body = {
            {"model", MODEL_NAME},
            {"messages", {
                {{"role", "system"}, {"content", "You are a helpful assistant."}},
                {{"role", "user"}, {"content", PROMPT + "\n" + text}}
            }},
            {"stream", false},
            {"enable_websearch", true}
        };

        if (!KNOWLEDGE_BASE_IDS.empty()) {
            body["knowledge_base_ids"] = json::array({KNOWLEDGE_BASE_IDS});
        }

        auto chat = openai::chat().create(body);
        response_string = strip_internal_reasoning_tags(extract_message_content(chat));
        response_string = revise_fhir_bundle_in_response(response_string, std::to_string(analysis_id), file);
        if (response_string.empty()) {
            file << "\n[WARN] No textual content found in primary response. Full payload:\n"
                 << chat.dump(2) << "\n";
            say_error(std::string{"[WARN] Analysis["} + std::to_string(analysis_id) +
                      "] returned no text content; see results file.\n");
        }

        file << "\n\nFull response received:\n" << response_string << "\n";
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id << "] failed: " << e.what() << "\n";
        say_error(std::string{"[ERROR] Analysis["} + std::to_string(analysis_id) + "] failed: " + e.what() + "\n");
    }

    if (!response_string.empty()) {
        try {
            json summary_body = {
                {"model", MODEL_NAME},
                {"messages", {
                    {{"role", "system"}, {"content", "You are a helpful assistant."}},
                    {{"role", "user"}, {"content", "Provide only a concise summary (max 3 short sentences) of the following text. Do not include internal reasoning, tags, or analysis steps.\n" + response_string + "\n\n"}}
                }},
                {"stream", false},
                {"enable_websearch", false}
            };

            auto summary_chat = openai::chat().create(summary_body);
            const std::string summary_string = strip_internal_reasoning_tags(extract_message_content(summary_chat));
            if (summary_string.empty()) {
                file << "\n[WARN] No textual summary returned. Full payload:\n"
                     << summary_chat.dump(2) << "\n";
                say_error(std::string{"[WARN] Summary generation returned no text for Analysis["} +
                          std::to_string(analysis_id) + "]; see results file.\n");
            }

            file << "\nShort summary of response:\n" << summary_string << "\n";
            speak_text("Analysis[" + std::to_string(analysis_id) + "] completed. Summary: " + summary_string);
        } catch (const std::exception& e) {
            file << "\n[ERROR] Summary generation failed: " << e.what() << "\n";
            say_error(std::string{"[ERROR] Summary generation failed for Analysis["} + std::to_string(analysis_id) + "]: " + e.what() + "\n");
        }
    }

    if (!file) {
        say_error("[ERROR] Writing to results file failed for Analysis[" + std::to_string(analysis_id) + "]\n");
    }

    say_info("Analysis of Recording[" + std::to_string(analysis_id) + "] Finished ------------------->>>\n");
}

void temp_analyze_text(const std::string& text) {
    AnalysisSession session(analysis_mutex, active_analyses);
    const int analysis_id = ++temp_counter_value;
    const std::string analysis_id_str = std::to_string(counter_value + 1) + "." + std::to_string(analysis_id); 
    say_info("Temporary_Analysis of Recording[" + analysis_id_str + "] Started ------------------->>>\n");

    const std::string filename = "tmp_results_analysis" + analysis_id_str + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        say_error("[ERROR] Unable to open results file: " + filename + "\n");
        say_info("Temporary Analysis of Recording[" + analysis_id_str + "] Finished ------------------->>>\n");
        return;
    }

    file << "Using model: " << MODEL_NAME << "\n";
    file << "Endpoint: " << OPENWEBUI_URL << "\n";
    file << "Prompt: " << TEMP_PROMPT << "\n" << text << "\n";

    if (!file) {
        say_error("[ERROR] Failed to write analysis header to " + filename + "\n");
    }

    std::string response_string;

    try {
        openai::start({
            API_KEY
        });

        json body = {
            {"model", MODEL_NAME},
            {"messages", {
                {{"role", "system"}, {"content", "You are a helpful assistant."}},
                {{"role", "user"}, {"content", TEMP_PROMPT + "\n" + text}}
            }},
            {"stream", false},
            {"enable_websearch", true}
        };

        if (!KNOWLEDGE_BASE_IDS.empty()) {
            body["knowledge_base_ids"] = json::array({KNOWLEDGE_BASE_IDS});
        }

        auto chat = openai::chat().create(body);
        response_string = strip_internal_reasoning_tags(extract_message_content(chat));
        response_string = revise_fhir_bundle_in_response(response_string, "tmp_" + analysis_id_str, file);
        if (response_string.empty()) {
            file << "\n[WARN] No textual content found in temporary response. Full payload:\n"
                 << chat.dump(2) << "\n";
            say_error(std::string{"[WARN] Analysis["} + analysis_id_str +
                      "] returned no text content; see results file.\n");
        }

        file << "\n\nTemporary response received:\n" << response_string << "\n";
        speak_text("Temporary Analysis[" + analysis_id_str + "] completed. Response: " + response_string);
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id_str << "] failed: " << e.what() << "\n";
        say_error(std::string{"[ERROR] Analysis["} + analysis_id_str + "] failed: " + e.what() + "\n");
    }

    if (!file) {
        say_error("[ERROR] Writing to results file failed for Analysis[" + analysis_id_str + "]\n");
    }

    say_info("Temporary Analysis of Recording[" + analysis_id_str + "] Finished ------------------->>>\n");
}

// Main loop
int main() {
    if (!load_config("./config.ini")) {
        say_error("Failed to load config.ini\n");
        return 1;
    }

    say_info("Listening for input...\n");

    std::string line;
    std::string collected_text;
    bool collect_text = false;

    while (std::getline(std::cin, line)) {
        std::cout << line << std::endl;

        std::string lower_line = line;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);

        const bool line_contains_start = contains_substring(lower_line, TRIGGER_START);
        const bool line_contains_stop = contains_substring(lower_line, TRIGGER_STOP);
        const bool line_contains_temp_check = contains_substring(lower_line, TRIGGER_TEMP_CHECK);

        if (line_contains_start) {
            if (collect_text) {
                say_info("Recording has already been started ------------------->>>\n");
            } else {
                say_info("Recording started ------------------->>>\n");
                collected_text.clear();
                collect_text = true;
            }
        }

        if (line_contains_stop) {
            if (!collect_text) {
                say_info("No recording is currently running ------------------->>>\n");
            } else {
                say_info("Recording stopped ------------------->>>\n");
                std::string text_to_analyze = collected_text;
                collected_text.clear();
                collect_text = false;
                if (active_analyses.load() > 0) {
                    say_info("Another analysis is running; this one will start once it finishes ------------------->>>\n");
                }
                std::thread(analyze_text, std::move(text_to_analyze)).detach();
            }
        }

        if (line_contains_temp_check) {
            if (!collect_text) {
                say_info("No recording is currently running ------------------->>>\n");
            } else {
                say_info("Temporary check requested ------------------->>>\n");
                if (active_analyses.load() > 0) {
                    say_info("Another analysis is running; this one will start once it finishes ------------------->>>\n");
                }
                std::string snapshot = collected_text;
                std::thread(temp_analyze_text, std::move(snapshot)).detach();
            }
        }

        if (collect_text && !line_contains_start && !line_contains_stop && !line_contains_temp_check) {
            collected_text += line + "\n";
        }
    }

    return 0;
}
