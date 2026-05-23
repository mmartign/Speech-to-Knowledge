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

// Global runtime configuration loaded from config.ini.
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
bool MAPPER_NETWORK_ENABLED = true;
std::string MAPPER_CACHE_DIR = "./terminology_cache";
std::string MAPPER_CACHE_TTL_DAYS = "7";
std::string MAPPER_LOINC_USER;
std::string MAPPER_LOINC_PASS;
std::string MAPPER_TIMEOUT_SECONDS = "10";

std::mutex analysis_mutex;
std::atomic<int> counter_value{0};
std::atomic<int> temp_counter_value{0};
std::atomic<int> active_analyses{0};
std::mutex tts_mutex;
std::once_flag openai_init_flag;
bool check_fhir = false;

// ── Language / i18n ──────────────────────────────────────────────────────────

enum class Lang { EN, IT, FR };
static Lang g_lang = Lang::EN;

enum MsgKey {
    MSG_UNABLE_TO_OPEN_CONFIG = 0,
    MSG_MISSING_REQUIRED_CONFIG,
    MSG_KB_NOT_SET,
    MSG_ERR_OPEN_RESULTS,
    MSG_WARN_NO_TEXT_PREFIX,
    MSG_WARN_NO_TEXT_SUFFIX,
    MSG_ERR_ANALYSIS_FAILED_PREFIX,
    MSG_ERR_ANALYSIS_FAILED_MIDDLE,
    MSG_WARN_SUMMARY_NO_TEXT_PREFIX,
    MSG_WARN_SUMMARY_NO_TEXT_SUFFIX,
    MSG_ERR_SUMMARY_FAILED_PREFIX,
    MSG_ERR_SUMMARY_FAILED_MIDDLE,
    MSG_ERR_WRITE_RESULTS_PREFIX,
    MSG_ERR_WRITE_RESULTS_SUFFIX,
    MSG_ERR_WRITE_HEADER,
    MSG_FAILED_LOAD_CONFIG,
    MSG_LISTENING,
    MSG_ANALYSIS_STARTED_PREFIX,
    MSG_ANALYSIS_STARTED_SUFFIX,
    MSG_ANALYSIS_FINISHED_SUFFIX,
    MSG_TEMP_ANALYSIS_STARTED_PREFIX,
    MSG_TEMP_ANALYSIS_FINISHED_PREFIX,
    MSG_RECORDING_ALREADY_STARTED,
    MSG_RECORDING_STARTED,
    MSG_NO_RECORDING_RUNNING,
    MSG_RECORDING_STOPPED,
    MSG_ANOTHER_ANALYSIS_RUNNING,
    MSG_TEMP_CHECK_REQUESTED,
    MSG_COUNT
};

// Columns: EN=0, IT=1, FR=2
static const char* MESSAGES[MSG_COUNT][3] = {
    /* MSG_UNABLE_TO_OPEN_CONFIG */
    {"Unable to open config file: ",
     "Impossibile aprire il file di configurazione: ",
     "Impossible d'ouvrir le fichier de configuration : "},
    /* MSG_MISSING_REQUIRED_CONFIG */
    {"Missing required config values:",
     "Valori di configurazione richiesti mancanti:",
     "Valeurs de configuration requises manquantes :"},
    /* MSG_KB_NOT_SET */
    {"Warning: analysis.knowledge_base_ids is not set; knowledge base lookups will be skipped.\n",
     "Attenzione: analysis.knowledge_base_ids non è impostato; le ricerche nella knowledge base verranno saltate.\n",
     "Avertissement : analysis.knowledge_base_ids n'est pas défini ; les recherches dans la base de connaissances seront ignorées.\n"},
    /* MSG_ERR_OPEN_RESULTS */
    {"[ERROR] Unable to open results file: ",
     "[ERRORE] Impossibile aprire il file dei risultati: ",
     "[ERREUR] Impossible d'ouvrir le fichier de résultats : "},
    /* MSG_WARN_NO_TEXT_PREFIX */
    {"[WARN] Analysis[",
     "[AVVISO] Analisi[",
     "[AVERT] Analyse["},
    /* MSG_WARN_NO_TEXT_SUFFIX */
    {"] returned no text content; see results file.\n",
     "] non ha restituito contenuto testuale; vedere il file dei risultati.\n",
     "] n'a retourné aucun contenu textuel ; voir le fichier de résultats.\n"},
    /* MSG_ERR_ANALYSIS_FAILED_PREFIX */
    {"[ERROR] Analysis[",
     "[ERRORE] Analisi[",
     "[ERREUR] Analyse["},
    /* MSG_ERR_ANALYSIS_FAILED_MIDDLE */
    {"] failed: ",
     "] fallita: ",
     "] échouée : "},
    /* MSG_WARN_SUMMARY_NO_TEXT_PREFIX */
    {"[WARN] Summary generation returned no text for Analysis[",
     "[AVVISO] La generazione del riepilogo non ha restituito testo per Analisi[",
     "[AVERT] La génération du résumé n'a retourné aucun texte pour Analyse["},
    /* MSG_WARN_SUMMARY_NO_TEXT_SUFFIX */
    {"]; see results file.\n",
     "]; vedere il file dei risultati.\n",
     "] ; voir le fichier de résultats.\n"},
    /* MSG_ERR_SUMMARY_FAILED_PREFIX */
    {"[ERROR] Summary generation failed for Analysis[",
     "[ERRORE] Generazione del riepilogo fallita per Analisi[",
     "[ERREUR] Génération du résumé échouée pour Analyse["},
    /* MSG_ERR_SUMMARY_FAILED_MIDDLE */
    {"]: ",
     "]: ",
     "] : "},
    /* MSG_ERR_WRITE_RESULTS_PREFIX */
    {"[ERROR] Writing to results file failed for Analysis[",
     "[ERRORE] Scrittura nel file dei risultati fallita per Analisi[",
     "[ERREUR] Échec d'écriture dans le fichier de résultats pour Analyse["},
    /* MSG_ERR_WRITE_RESULTS_SUFFIX */
    {"]\n", "]\n", "]\n"},
    /* MSG_ERR_WRITE_HEADER */
    {"[ERROR] Failed to write analysis header to ",
     "[ERRORE] Impossibile scrivere l'intestazione dell'analisi in ",
     "[ERREUR] Impossible d'écrire l'en-tête d'analyse dans "},
    /* MSG_FAILED_LOAD_CONFIG */
    {"Failed to load config.ini\n",
     "Impossibile caricare config.ini\n",
     "Impossible de charger config.ini\n"},
    /* MSG_LISTENING */
    {"Listening for input...\n",
     "In ascolto per l'input...\n",
     "En attente d'entrée...\n"},
    /* MSG_ANALYSIS_STARTED_PREFIX */
    {"Analysis of Recording[",
     "Analisi della Registrazione[",
     "Analyse de l'Enregistrement["},
    /* MSG_ANALYSIS_STARTED_SUFFIX */
    {"] Started ------------------->>>\n",
     "] Avviata ------------------->>>\n",
     "] Démarrée ------------------->>>\n"},
    /* MSG_ANALYSIS_FINISHED_SUFFIX */
    {"] Finished ------------------->>>\n",
     "] Completata ------------------->>>\n",
     "] Terminée ------------------->>>\n"},
    /* MSG_TEMP_ANALYSIS_STARTED_PREFIX */
    {"Temporary_Analysis of Recording[",
     "Analisi_Temporanea della Registrazione[",
     "Analyse_Temporaire de l'Enregistrement["},
    /* MSG_TEMP_ANALYSIS_FINISHED_PREFIX */
    {"Temporary Analysis of Recording[",
     "Analisi Temporanea della Registrazione[",
     "Analyse Temporaire de l'Enregistrement["},
    /* MSG_RECORDING_ALREADY_STARTED */
    {"Recording has already been started ------------------->>>\n",
     "La registrazione è già stata avviata ------------------->>>\n",
     "L'enregistrement a déjà été démarré ------------------->>>\n"},
    /* MSG_RECORDING_STARTED */
    {"Recording started ------------------->>>\n",
     "Registrazione avviata ------------------->>>\n",
     "Enregistrement démarré ------------------->>>\n"},
    /* MSG_NO_RECORDING_RUNNING */
    {"No recording is currently running ------------------->>>\n",
     "Nessuna registrazione è in corso ------------------->>>\n",
     "Aucun enregistrement n'est en cours ------------------->>>\n"},
    /* MSG_RECORDING_STOPPED */
    {"Recording stopped ------------------->>>\n",
     "Registrazione fermata ------------------->>>\n",
     "Enregistrement arrêté ------------------->>>\n"},
    /* MSG_ANOTHER_ANALYSIS_RUNNING */
    {"Another analysis is running; this one will start once it finishes ------------------->>>\n",
     "Un'altra analisi è in corso; questa inizierà al termine ------------------->>>\n",
     "Une autre analyse est en cours ; celle-ci démarrera une fois terminée ------------------->>>\n"},
    /* MSG_TEMP_CHECK_REQUESTED */
    {"Temporary check requested ------------------->>>\n",
     "Controllo temporaneo richiesto ------------------->>>\n",
     "Vérification temporaire demandée ------------------->>>\n"},
};

static const char* tr(MsgKey key) {
    const int idx = (g_lang == Lang::IT) ? 1 : (g_lang == Lang::FR) ? 2 : 0;
    return MESSAGES[key][idx];
}

// ── End i18n ─────────────────────────────────────────────────────────────────

std::string escape_for_single_quotes(const std::string& text);

class AnalysisSession {
public:
    AnalysisSession(std::mutex& mutex, std::atomic<int>& active_count)
        : lock_(mutex), active_count_(active_count) {
        // Serialize analysis sections that share output/logging resources.
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

std::string ensure_trailing_slash(std::string url) {
    if (!url.empty() && url.back() != '/') {
        url.push_back('/');
    }
    return url;
}

std::vector<std::string> split_config_list(const std::string& value) {
    std::vector<std::string> items;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_whitespace(item);
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

std::string strip_json_like_comments(const std::string& input) {
    // Supports model responses that include JSON with JS-style comments.
    std::string out;
    out.reserve(input.size());

    bool in_string = false;
    bool escaped = false;
    bool in_line_comment = false;
    bool in_block_comment = false;

    for (size_t i = 0; i < input.size(); ++i) {
        const char c = input[i];
        const char next = (i + 1 < input.size()) ? input[i + 1] : '\0';

        if (in_line_comment) {
            if (c == '\n') {
                in_line_comment = false;
                out.push_back(c);
            }
            continue;
        }

        if (in_block_comment) {
            if (c == '*' && next == '/') {
                in_block_comment = false;
                ++i;
                continue;
            }
            if (c == '\n') {
                out.push_back(c);
            }
            continue;
        }

        if (in_string) {
            out.push_back(c);
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
            out.push_back(c);
            continue;
        }

        if (c == '/' && next == '/') {
            in_line_comment = true;
            ++i;
            continue;
        }

        if (c == '/' && next == '*') {
            in_block_comment = true;
            ++i;
            continue;
        }

        out.push_back(c);
    }

    return out;
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
    // Remove leaked internal <unused...> segments before user-facing output.
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
                        json parsed;
                        try {
                            parsed = json::parse(candidate);
                        } catch (...) {
                            parsed = json::parse(strip_json_like_comments(candidate));
                        }
                        if (is_fhir_bundle_object(parsed)) {
                            // Return the first valid Bundle object found in the response text.
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
    // Accept either raw Bundle output or wrapped mapper payload.
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
        // Fast path: no Bundle detected, return original model output.
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

    std::ostringstream cmd_builder;
    // Shell out to deterministic mapper to post-process model-generated FHIR.
    cmd_builder << "./deterministic_fhir_mapper.exe '" << escape_for_single_quotes(input_path)
                << "' '" << escape_for_single_quotes(output_path)
                << "' --model-name '" << escape_for_single_quotes(MODEL_NAME) << "'";
    if (!MAPPER_NETWORK_ENABLED) {
        cmd_builder << " --no-network";
    }
    if (!MAPPER_CACHE_DIR.empty()) {
        cmd_builder << " --cache-dir '" << escape_for_single_quotes(MAPPER_CACHE_DIR) << "'";
    }
    if (!MAPPER_CACHE_TTL_DAYS.empty()) {
        cmd_builder << " --cache-ttl-days '" << escape_for_single_quotes(MAPPER_CACHE_TTL_DAYS) << "'";
    }
    if (!MAPPER_LOINC_USER.empty()) {
        cmd_builder << " --loinc-user '" << escape_for_single_quotes(MAPPER_LOINC_USER) << "'";
    }
    if (!MAPPER_LOINC_PASS.empty()) {
        cmd_builder << " --loinc-pass '" << escape_for_single_quotes(MAPPER_LOINC_PASS) << "'";
    }
    if (!MAPPER_TIMEOUT_SECONDS.empty()) {
        cmd_builder << " --timeout '" << escape_for_single_quotes(MAPPER_TIMEOUT_SECONDS) << "'";
    }
    cmd_builder << " >/dev/null 2>&1";
    const std::string cmd = cmd_builder.str();
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
        // Replace only the detected Bundle span, leaving surrounding narrative intact.
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
    // POSIX-shell escaping for single-quoted command arguments.
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

    trimmed = "SI-Listener Assistant: " + trimmed;

    const std::string escaped = escape_for_single_quotes(trimmed);
    const std::string cmd = TTS_COMMAND + " '" + escaped + "' >/dev/null 2>&1 &";

    // TTS backend is shared; avoid overlapping command writes.
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
        say_error(tr(MSG_UNABLE_TO_OPEN_CONFIG) + path + "\n");
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
    // Optional: empty means no KB augmentation, not a hard failure.
    KNOWLEDGE_BASE_IDS = (kb_it != config.end()) ? kb_it->second : std::string{};

    auto mapper_network_it = config.find("deterministic_mapper.network_enabled");
    if (mapper_network_it != config.end()) {
        std::string value = mapper_network_it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        MAPPER_NETWORK_ENABLED = !(value == "false" || value == "0" || value == "no" || value == "off");
    }

    auto mapper_cache_dir_it = config.find("deterministic_mapper.cache_dir");
    if (mapper_cache_dir_it != config.end() && !mapper_cache_dir_it->second.empty()) {
        MAPPER_CACHE_DIR = mapper_cache_dir_it->second;
    }

    auto mapper_cache_ttl_it = config.find("deterministic_mapper.cache_ttl_days");
    if (mapper_cache_ttl_it != config.end() && !mapper_cache_ttl_it->second.empty()) {
        MAPPER_CACHE_TTL_DAYS = mapper_cache_ttl_it->second;
    }

    auto mapper_loinc_user_it = config.find("deterministic_mapper.loinc_user");
    if (mapper_loinc_user_it != config.end()) {
        MAPPER_LOINC_USER = mapper_loinc_user_it->second;
    }

    auto mapper_loinc_pass_it = config.find("deterministic_mapper.loinc_pass");
    if (mapper_loinc_pass_it != config.end()) {
        MAPPER_LOINC_PASS = mapper_loinc_pass_it->second;
    }

    auto mapper_timeout_it = config.find("deterministic_mapper.timeout_seconds");
    if (mapper_timeout_it != config.end() && !mapper_timeout_it->second.empty()) {
        MAPPER_TIMEOUT_SECONDS = mapper_timeout_it->second;
    }

    if (!missing_keys.empty()) {
        std::ostringstream oss;
        oss << tr(MSG_MISSING_REQUIRED_CONFIG);
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
    OPENWEBUI_URL = ensure_trailing_slash(OPENWEBUI_URL);

    if (KNOWLEDGE_BASE_IDS.empty()) {
        say_error(tr(MSG_KB_NOT_SET));
    }

    return true;
}

// Substring check
bool contains_substring(const std::string& str, const std::string& sub) {
    return sub.empty() || str.find(sub) != std::string::npos;
}

// Safely extract a textual message content from an OpenAI-style response
std::string extract_message_content(const json& response) {
    // Normalize multiple OpenAI response shapes into plain text.
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

std::string extract_api_error(const json& response) {
    if (!response.is_object()) {
        return {};
    }

    const auto error_it = response.find("error");
    if (error_it != response.end()) {
        if (error_it->is_string()) {
            return error_it->get<std::string>();
        }
        return error_it->dump();
    }

    const auto detail_it = response.find("detail");
    if (detail_it != response.end()) {
        if (detail_it->is_string()) {
            return detail_it->get<std::string>();
        }
        return detail_it->dump();
    }

    const auto message_it = response.find("message");
    if (message_it != response.end() && message_it->is_string()) {
        return message_it->get<std::string>();
    }

    return {};
}

void attach_configured_knowledge_collections(json& body) {
    const auto collection_ids = split_config_list(KNOWLEDGE_BASE_IDS);
    if (collection_ids.empty()) {
        return;
    }

    json files = json::array();
    for (const auto& id : collection_ids) {
        files.push_back({
            {"type", "collection"},
            {"id", id}
        });
    }
    body["files"] = std::move(files);
}

json build_chat_body(const std::string& user_content, bool enable_websearch, bool use_knowledge = true) {
    json body = {
        {"model", MODEL_NAME},
        {"messages", {
            {{"role", "system"}, {"content", "You are a helpful assistant."}},
            {{"role", "user"}, {"content", user_content}}
        }},
        {"stream", false},
        {"chat_id", ""},
        {"enable_websearch", enable_websearch}
    };

    if (use_knowledge) {
        attach_configured_knowledge_collections(body);
    }
    return body;
}

void initialize_openai_client() {
    std::call_once(openai_init_flag, [] {
        openai::start(API_KEY, "", true, OPENWEBUI_URL);
    });
}

// AI analysis with fresh context for each request
void analyze_text(const std::string& text) {
    AnalysisSession session(analysis_mutex, active_analyses);
    const int analysis_id = ++counter_value;
    temp_counter_value = 0; // Reset temp counter for each main analysis
    say_info(tr(MSG_ANALYSIS_STARTED_PREFIX) + std::to_string(analysis_id) + tr(MSG_ANALYSIS_STARTED_SUFFIX));

    const std::string filename = "results_analysis" + std::to_string(analysis_id) + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        say_error(tr(MSG_ERR_OPEN_RESULTS) + filename + "\n");
        say_info(tr(MSG_ANALYSIS_STARTED_PREFIX) + std::to_string(analysis_id) + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
        return;
    }

    file << "Using model: " << MODEL_NAME << "\n";
    file << "Endpoint: " << OPENWEBUI_URL << "\n";
    file << "Prompt: " << PROMPT << "\n" << text << "\n";

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_HEADER) + filename + "\n");
    }

    std::string response_string;

    try {
        initialize_openai_client();

        json body = build_chat_body(PROMPT + "\n" + text, true);
        auto chat = openai::chat().create(body);
        const std::string api_error = extract_api_error(chat);
        if (!api_error.empty()) {
            throw std::runtime_error("OpenWebUI API error: " + api_error);
        }
        // Strip model-internal tags, then optionally run deterministic FHIR post-processing.
        response_string = strip_internal_reasoning_tags(extract_message_content(chat));
        if (check_fhir) {
            response_string = revise_fhir_bundle_in_response(response_string, std::to_string(analysis_id), file);
        }
        if (response_string.empty()) {
            file << "\n[WARN] No textual content found in primary response. Full payload:\n"
                 << chat.dump(2) << "\n";
            say_error(std::string{tr(MSG_WARN_NO_TEXT_PREFIX)} + std::to_string(analysis_id) +
                      tr(MSG_WARN_NO_TEXT_SUFFIX));
        }

        file << "\n\nFull response received:\n" << response_string << "\n";
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id << "] failed: " << e.what() << "\n";
        say_error(std::string{tr(MSG_ERR_ANALYSIS_FAILED_PREFIX)} + std::to_string(analysis_id) +
                  tr(MSG_ERR_ANALYSIS_FAILED_MIDDLE) + e.what() + "\n");
    }

    if (!response_string.empty()) {
        try {
            // Follow-up summary call keeps spoken output concise.
            json summary_body = build_chat_body(
                "Provide only a concise summary (max 3 short sentences) of the following text. Do not include internal reasoning, tags, or analysis steps.\n" +
                    response_string + "\n\n",
                false,
                false);

            auto summary_chat = openai::chat().create(summary_body);
            const std::string summary_api_error = extract_api_error(summary_chat);
            if (!summary_api_error.empty()) {
                throw std::runtime_error("OpenWebUI API error: " + summary_api_error);
            }
            const std::string summary_string = strip_internal_reasoning_tags(extract_message_content(summary_chat));
            if (summary_string.empty()) {
                file << "\n[WARN] No textual summary returned. Full payload:\n"
                     << summary_chat.dump(2) << "\n";
                say_error(std::string{tr(MSG_WARN_SUMMARY_NO_TEXT_PREFIX)} + std::to_string(analysis_id) +
                          tr(MSG_WARN_SUMMARY_NO_TEXT_SUFFIX));
            }

            file << "\nShort summary of response:\n" << summary_string << "\n";
            speak_text("Analysis[" + std::to_string(analysis_id) + "] completed. Summary: " + summary_string);
        } catch (const std::exception& e) {
            file << "\n[ERROR] Summary generation failed: " << e.what() << "\n";
            say_error(std::string{tr(MSG_ERR_SUMMARY_FAILED_PREFIX)} + std::to_string(analysis_id) +
                      tr(MSG_ERR_SUMMARY_FAILED_MIDDLE) + e.what() + "\n");
        }
    }

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_RESULTS_PREFIX) + std::to_string(analysis_id) + tr(MSG_ERR_WRITE_RESULTS_SUFFIX));
    }

    say_info(tr(MSG_ANALYSIS_STARTED_PREFIX) + std::to_string(analysis_id) + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
}

void temp_analyze_text(const std::string& text) {
    AnalysisSession session(analysis_mutex, active_analyses);
    const int analysis_id = ++temp_counter_value;
    // Use compound id (<main>.<temp>) so temp files sort with their parent analysis.
    const std::string analysis_id_str = std::to_string(counter_value + 1) + "." + std::to_string(analysis_id);
    say_info(tr(MSG_TEMP_ANALYSIS_STARTED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_STARTED_SUFFIX));

    const std::string filename = "tmp_results_analysis" + analysis_id_str + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        say_error(tr(MSG_ERR_OPEN_RESULTS) + filename + "\n");
        say_info(tr(MSG_TEMP_ANALYSIS_FINISHED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
        return;
    }

    file << "Using model: " << MODEL_NAME << "\n";
    file << "Endpoint: " << OPENWEBUI_URL << "\n";
    file << "Prompt: " << TEMP_PROMPT << "\n" << text << "\n";

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_HEADER) + filename + "\n");
    }

    std::string response_string;

    try {
        initialize_openai_client();

        json body = build_chat_body(TEMP_PROMPT + "\n" + text, true);
        auto chat = openai::chat().create(body);
        const std::string api_error = extract_api_error(chat);
        if (!api_error.empty()) {
            throw std::runtime_error("OpenWebUI API error: " + api_error);
        }
        response_string = strip_internal_reasoning_tags(extract_message_content(chat));
        if (check_fhir) {
            response_string = revise_fhir_bundle_in_response(response_string, "tmp_" + analysis_id_str, file);
        }
        if (response_string.empty()) {
            file << "\n[WARN] No textual content found in temporary response. Full payload:\n"
                 << chat.dump(2) << "\n";
            say_error(std::string{tr(MSG_WARN_NO_TEXT_PREFIX)} + analysis_id_str +
                      tr(MSG_WARN_NO_TEXT_SUFFIX));
        }

        file << "\n\nTemporary response received:\n" << response_string << "\n";
        speak_text("Temporary Analysis[" + analysis_id_str + "] completed. Response: " + response_string);
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id_str << "] failed: " << e.what() << "\n";
        say_error(std::string{tr(MSG_ERR_ANALYSIS_FAILED_PREFIX)} + analysis_id_str +
                  tr(MSG_ERR_ANALYSIS_FAILED_MIDDLE) + e.what() + "\n");
    }

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_RESULTS_PREFIX) + analysis_id_str + tr(MSG_ERR_WRITE_RESULTS_SUFFIX));
    }

    say_info(tr(MSG_TEMP_ANALYSIS_FINISHED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
}

static void print_help(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [OPTIONS]\n"
        "\n"
        "Listens on standard input for trigger words, collects spoken text between\n"
        "start/stop triggers, and sends it to a configured AI model for analysis.\n"
        "Results are written to numbered files (results_analysis<N>.txt).\n"
        "\n"
        "Options:\n"
        "  --help                  Show this help message and exit.\n"
        "  --check_fhir            After each analysis, detect and post-process any\n"
        "                          FHIR Bundle found in the model response using the\n"
        "                          deterministic_fhir_mapper tool.\n"
        "  --language <lang>       Set the UI language for console messages.\n"
        "                          Supported values: en (default), it, fr.\n"
        "\n"
        "Configuration:\n"
        "  The program reads ./config.ini on startup. The following sections and\n"
        "  keys are required:\n"
        "    [openai]   base_url, api_key, model_name\n"
        "    [prompts]  prompt, temp_prompt\n"
        "    [triggers] start, stop, temp_check\n"
        "    [tts]      command\n"
        "  Optional keys:\n"
        "    [analysis]             knowledge_base_ids\n"
        "    [deterministic_mapper] network_enabled, cache_dir, cache_ttl_days,\n"
        "                           loinc_user, loinc_pass, timeout_seconds\n"
        "\n"
        "Trigger words (configured in config.ini):\n"
        "  start      Begin collecting transcribed speech.\n"
        "  stop       Stop collecting and send text to AI for full analysis.\n"
        "  temp_check Perform a temporary analysis on a snapshot of collected text\n"
        "             without stopping the recording.\n"
        "\n"
        "Exit status:\n"
        "  0  Normal exit (EOF on stdin).\n"
        "  1  Configuration error or invalid argument.\n";
}

// Main loop
int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--check_fhir") {
            check_fhir = true;
        } else if (arg == "--language") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --language requires an argument. Supported: en, it, fr\n";
                return 1;
            }
            const std::string lang(argv[++i]);
            if (lang == "it") {
                g_lang = Lang::IT;
            } else if (lang == "fr") {
                g_lang = Lang::FR;
            } else if (lang != "en") {
                std::cerr << "Error: unknown language '" << lang << "'. Supported: en, it, fr\n";
                return 1;
            }
        } else {
            std::cerr << "Error: unknown option '" << arg << "'. Use --help for usage information.\n";
            return 1;
        }
    }

    if (!load_config("./config.ini")) {
        say_error(tr(MSG_FAILED_LOAD_CONFIG));
        return 1;
    }

    say_info(tr(MSG_LISTENING));

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
                say_info(tr(MSG_RECORDING_ALREADY_STARTED));
            } else {
                say_info(tr(MSG_RECORDING_STARTED));
                collected_text.clear();
                collect_text = true;
            }
        }

        if (line_contains_stop) {
            if (!collect_text) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else {
                say_info(tr(MSG_RECORDING_STOPPED));
                std::string text_to_analyze = collected_text;
                collected_text.clear();
                collect_text = false;
                if (active_analyses.load() > 0) {
                    say_info(tr(MSG_ANOTHER_ANALYSIS_RUNNING));
                }
                std::thread(analyze_text, std::move(text_to_analyze)).detach();
            }
        }

        if (line_contains_temp_check) {
            if (!collect_text) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else {
                say_info(tr(MSG_TEMP_CHECK_REQUESTED));
                if (active_analyses.load() > 0) {
                    say_info(tr(MSG_ANOTHER_ANALYSIS_RUNNING));
                }
                std::string snapshot = collected_text;
                // Temp analysis runs on a snapshot while recording continues.
                std::thread(temp_analyze_text, std::move(snapshot)).detach();
            }
        }

        if (collect_text && !line_contains_start && !line_contains_stop && !line_contains_temp_check) {
            collected_text += line + "\n";
        }
    }

    return 0;
}
