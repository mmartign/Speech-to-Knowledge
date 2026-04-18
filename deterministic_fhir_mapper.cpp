// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Project: Spazio IT Speech-to-Knowledge
// File: deterministic_fhir_mapper.cpp
//
// Copyright (C) 2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
// Summary:
// Enhanced deterministic FHIR mapper with live terminology lookup.
//
// Features:
// - TerminologyClient for LOINC FHIR R4 and NLM RxNorm lookups
// - Disk-based JSON cache with configurable TTL (default: 7 days)
// - Graceful offline fallback when network calls fail
// - `--no-network` flag for offline-only operation
// - `--cache-dir <path>` to configure cache location (default: ./terminology_cache)
// - `--cache-ttl-days <n>` to configure cache expiration
//
// Build dependencies:
// - libcurl (curl/curl.h)
// - nlohmann/json
//
// Example build:
// g++ -std=c++17 -O2 -o deterministic_fhir_mapper.exe \
//     deterministic_fhir_mapper.cpp \
//     -lcurl -I/path/to/nlohmann
//
// License:
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

// === HTTP Helper (libcurl) ===============================================

namespace Http {

std::string url_encode(const std::string& s);

struct Response {
    long status_code = 0;
    std::string body;
    bool ok() const { return status_code >= 200 && status_code < 300; }
};

static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(ptr, size * nmemb);
    return size * nmemb;
}

// Perform a GET with optional Basic-Auth header.
// Returns an empty Response on curl initialisation failure.
Response get(const std::string& url,
             const std::string& accept = "application/fhir+json",
             const std::string& auth_header = "",
             long timeout_seconds = 10) {
    Response result;
    CURL* curl = curl_easy_init();
    if (!curl) {
        return result;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result.body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    struct curl_slist* headers = nullptr;
    const std::string accept_hdr = "Accept: " + accept;
    headers = curl_slist_append(headers, accept_hdr.c_str());
    if (!auth_header.empty()) {
        headers = curl_slist_append(headers, auth_header.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);
    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &result.status_code);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return result;
}

} // namespace Http

// === Disk Cache ===========================================================

class DiskCache {
public:
    explicit DiskCache(fs::path dir, int ttl_days = 7)
        : dir_(std::move(dir)), ttl_seconds_(static_cast<long long>(ttl_days) * 86400) {
        std::error_code ec;
        fs::create_directories(dir_, ec);
    }

    // Returns cached JSON if present and not expired; nullopt otherwise.
    std::optional<json> get(const std::string& key) const {
        const fs::path p = entry_path(key);
        std::error_code ec;
        if (!fs::exists(p, ec)) {
            return std::nullopt;
        }
        const auto mtime = fs::last_write_time(p, ec);
        if (ec) {
            return std::nullopt;
        }
        const auto now = fs::file_time_type::clock::now();
        const long long age = std::chrono::duration_cast<std::chrono::seconds>(now - mtime).count();
        if (age > ttl_seconds_) {
            return std::nullopt;  // expired
        }
        try {
            std::ifstream f(p);
            json data;
            f >> data;
            return data;
        } catch (...) {
            return std::nullopt;
        }
    }

    void put(const std::string& key, const json& value) const {
        const fs::path p = entry_path(key);
        try {
            std::ofstream f(p);
            f << value.dump(2) << "\n";
        } catch (...) {
            // Cache write failure is non-fatal
        }
    }

private:
    fs::path entry_path(const std::string& key) const {
        // Sanitise key to a safe filename
        std::string safe;
        safe.reserve(key.size());
        for (char c : key) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.') {
                safe.push_back(c);
            } else {
                safe.push_back('_');
            }
        }
        return dir_ / (safe + ".json");
    }

    fs::path dir_;
    long long ttl_seconds_;
};

// === Terminology Client ===================================================

struct ResolvedCoding {
    std::string system;
    std::string code;
    std::string display;
    bool from_network = false;  // false = came from cache or hardcoded fallback
};

class TerminologyClient {
public:
    struct Config {
        bool network_enabled = true;
        fs::path cache_dir = "./terminology_cache";
        int cache_ttl_days = 7;
        long timeout_seconds = 10;
        // Optional LOINC credentials (https://loinc.org/fhir/ requires free registration)
        std::string loinc_user;
        std::string loinc_pass;
    };

    explicit TerminologyClient(Config cfg)
        : cfg_(std::move(cfg)), cache_(cfg_.cache_dir, cfg_.cache_ttl_days) {}

    // Look up a LOINC code. Returns display name if found.
    std::optional<ResolvedCoding> lookupLoinc(const std::string& code) {
        if (code.empty()) return std::nullopt;

        const std::string cache_key = "loinc_" + code;
        if (auto cached = readCachedCoding(cache_key, kLoincSystem, code, ""); cached.has_value()) {
            return cached;
        }
        if (!cfg_.network_enabled) return std::nullopt;

        const std::string url =
            "https://fhir.loinc.org/CodeSystem/$lookup"
            "?system=http%3A%2F%2Floinc.org"
            "&code=" + Http::url_encode(code);

        const auto body = fetchJson(url, "application/fhir+json", loincAuthHeader());
        if (!body.has_value()) return std::nullopt;

        const std::string display = parseFhirLookupDisplay(*body);
        if (display.empty()) return std::nullopt;

        writeCachedCoding(cache_key, code, display);
        return ResolvedCoding{kLoincSystem, code, display, true};
    }

    // Search LOINC by display text — useful when the LLM produced a label
    // but no code (e.g. "Heart rate" → 8867-4).
    // Uses LOINC FHIR CodeSystem search: /CodeSystem?system=http://loinc.org&_text=<TERM>
    std::optional<ResolvedCoding> searchLoincByDisplay(const std::string& display_text,
                                                        const std::string& unit_hint = "") {
        if (display_text.empty()) return std::nullopt;

        // Build a stable cache key from the search term
        const std::string cache_key = "loinc_search_" + lower(display_text) + "_" + lower(unit_hint);
        if (auto cached = readCachedCoding(cache_key, kLoincSystem, "", display_text); cached.has_value()) {
            return cached;
        }
        if (!cfg_.network_enabled) return std::nullopt;

        // LOINC FHIR search: GET /CodeSystem?system=http://loinc.org&_text=<TERM>&_count=5
        const std::string search_term = display_text + (unit_hint.empty() ? "" : " " + unit_hint);
        const std::string url =
            "https://fhir.loinc.org/CodeSystem"
            "?system=http%3A%2F%2Floinc.org"
            "&_text=" + Http::url_encode(search_term) +
            "&_count=5";

        const auto bundle = fetchJson(url, "application/fhir+json", loincAuthHeader());
        if (!bundle.has_value()) return std::nullopt;

        const auto resolved = parseLoincSearchResult(*bundle, display_text);
        if (!resolved.has_value()) return std::nullopt;

        writeCachedCoding(cache_key, resolved->code, resolved->display);
        return ResolvedCoding{kLoincSystem, resolved->code, resolved->display, true};
    }

    // RxNorm lookup: map a free-text drug name to an RxNorm CUI.
    // Uses the NLM RxNorm REST API (no auth required).
    // https://rxnav.nlm.nih.gov/REST/rxcui.json?name=<NAME>&search=2
    std::optional<ResolvedCoding> lookupRxNorm(const std::string& drug_name) {
        if (drug_name.empty()) return std::nullopt;

        const std::string cache_key = "rxnorm_" + lower(drug_name);
        if (auto cached = readCachedCoding(cache_key, kRxNormSystem, "", drug_name); cached.has_value()) {
            return cached;
        }
        if (!cfg_.network_enabled) return std::nullopt;

        const std::string url =
            "https://rxnav.nlm.nih.gov/REST/rxcui.json"
            "?name=" + Http::url_encode(drug_name) +
            "&search=2";

        const auto body = fetchJson(url, "application/json");
        if (!body.has_value() || !body->contains("idGroup") || !(*body)["idGroup"].contains("rxnormId")) {
            return std::nullopt;
        }

        const auto& ids = (*body)["idGroup"]["rxnormId"];
        if (!ids.is_array() || ids.empty()) return std::nullopt;
        const std::string rxcui = ids[0].get<std::string>();

        // Now fetch the display name for this CUI.
        std::string display = drug_name;
        const std::string name_url =
            "https://rxnav.nlm.nih.gov/REST/rxcui/" + rxcui + "/property.json?propName=RxNorm%20Name";
        if (const auto name_body = fetchJson(name_url, "application/json"); name_body.has_value()) {
            if (name_body->contains("propConceptGroup") &&
                (*name_body)["propConceptGroup"].contains("propConcept") &&
                (*name_body)["propConceptGroup"]["propConcept"].is_array() &&
                !(*name_body)["propConceptGroup"]["propConcept"].empty()) {
                display = (*name_body)["propConceptGroup"]["propConcept"][0].value("propValue", drug_name);
            }
        }

        writeCachedCoding(cache_key, rxcui, display);
        return ResolvedCoding{kRxNormSystem, rxcui, display, true};
    }

    // SNOMED CT lookup via NLM FHIR terminology server (no auth required for read).
    // https://cts.nlm.nih.gov/fhir/CodeSystem/$lookup?system=http://snomed.info/sct&code=<CODE>
    std::optional<ResolvedCoding> lookupSnomed(const std::string& code) {
        if (code.empty()) return std::nullopt;

        const std::string cache_key = "snomed_" + code;
        if (auto cached = readCachedCoding(cache_key, kSnomedSystem, code, ""); cached.has_value()) {
            return cached;
        }
        if (!cfg_.network_enabled) return std::nullopt;

        const std::string url =
            "https://cts.nlm.nih.gov/fhir/CodeSystem/$lookup"
            "?system=http%3A%2F%2Fsnomed.info%2Fsct"
            "&code=" + Http::url_encode(code);

        const auto body = fetchJson(url, "application/fhir+json");
        if (!body.has_value()) return std::nullopt;

        const std::string display = parseFhirLookupDisplay(*body);
        if (display.empty()) return std::nullopt;

        writeCachedCoding(cache_key, code, display);
        return ResolvedCoding{kSnomedSystem, code, display, true};
    }

    bool networkEnabled() const { return cfg_.network_enabled; }

private:
    struct ParsedLoincConcept {
        std::string code;
        std::string display;
    };

    static constexpr const char* kLoincSystem = "http://loinc.org";
    static constexpr const char* kRxNormSystem = "http://www.nlm.nih.gov/research/umls/rxnorm";
    static constexpr const char* kSnomedSystem = "http://snomed.info/sct";

    static std::string lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    std::optional<ResolvedCoding> readCachedCoding(const std::string& cache_key,
                                                   const std::string& system,
                                                   const std::string& default_code,
                                                   const std::string& default_display) const {
        if (auto cached = cache_.get(cache_key); cached.has_value()) {
            return ResolvedCoding{
                system,
                cached->value("code", default_code),
                cached->value("display", default_display),
                false
            };
        }
        return std::nullopt;
    }

    void writeCachedCoding(const std::string& cache_key,
                           const std::string& code,
                           const std::string& display) const {
        cache_.put(cache_key, json{{"code", code}, {"display", display}});
    }

    std::string loincAuthHeader() const {
        if (cfg_.loinc_user.empty()) return "";
        return "Authorization: Basic " + base64(cfg_.loinc_user + ":" + cfg_.loinc_pass);
    }

    std::optional<json> fetchJson(const std::string& url,
                                  const std::string& accept,
                                  const std::string& auth_header = "") const {
        const auto resp = Http::get(url, accept, auth_header, cfg_.timeout_seconds);
        if (!resp.ok() || resp.body.empty()) return std::nullopt;
        try {
            return json::parse(resp.body);
        } catch (...) {
            return std::nullopt;
        }
    }

    static std::string parseFhirLookupDisplay(const json& body) {
        if (!body.contains("parameter") || !body["parameter"].is_array()) return "";
        for (const auto& p : body["parameter"]) {
            if (p.value("name", "") == "display" && p.contains("valueString")) {
                return p["valueString"].get<std::string>();
            }
        }
        return "";
    }

    static std::optional<ParsedLoincConcept> parseLoincSearchResult(
        const json& bundle,
        const std::string& default_display) {
        if (!bundle.contains("entry") || !bundle["entry"].is_array() || bundle["entry"].empty()) {
            return std::nullopt;
        }

        const auto& first_entry = bundle["entry"][0];
        if (!first_entry.contains("resource")) return std::nullopt;
        const auto& code_system = first_entry["resource"];
        if (!code_system.contains("concept") || !code_system["concept"].is_array() ||
            code_system["concept"].empty()) {
            return std::nullopt;
        }

        const std::string code = code_system["concept"][0].value("code", "");
        if (code.empty()) return std::nullopt;
        return ParsedLoincConcept{
            code,
            code_system["concept"][0].value("display", default_display)
        };
    }

    // Minimal base64 encoder for Basic Auth header
    static std::string base64(const std::string& in) {
        static const char* table =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve(((in.size() + 2) / 3) * 4);
        for (size_t i = 0; i < in.size(); i += 3) {
            const unsigned char b0 = static_cast<unsigned char>(in[i]);
            const unsigned char b1 = (i + 1 < in.size()) ? static_cast<unsigned char>(in[i + 1]) : 0;
            const unsigned char b2 = (i + 2 < in.size()) ? static_cast<unsigned char>(in[i + 2]) : 0;
            out.push_back(table[b0 >> 2]);
            out.push_back(table[((b0 & 3) << 4) | (b1 >> 4)]);
            out.push_back((i + 1 < in.size()) ? table[((b1 & 15) << 2) | (b2 >> 6)] : '=');
            out.push_back((i + 2 < in.size()) ? table[b2 & 63] : '=');
        }
        return out;
    }

    Config cfg_;
    DiskCache cache_;
};

// Extend Http namespace with URL encoding helper
namespace Http {
std::string url_encode(const std::string& s) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return s;
    }
    char* encoded = curl_easy_escape(curl, s.c_str(), static_cast<int>(s.size()));
    std::string result;
    if (encoded) {
        result = encoded;
        curl_free(encoded);
    } else {
        result = s;
    }
    curl_easy_cleanup(curl);
    return result;
}
} // namespace Http

// === Mapper Logic (v1 structure + terminology integration) ===============

namespace {

struct Coding {
    std::string system;
    std::string code;
    std::string display;
};

struct CandidateBp {
    std::string id;
    std::string subjectRef;
    std::string effective;
    double value = 0.0;
    std::string unit;
    json original;
};

struct MapperIssue {
    std::string severity;
    std::string code;
    std::string details;
    std::string resourceId;
    std::string resourceType;
};

struct MapperResult {
    json resource;
    bool accepted = true;
    std::vector<MapperIssue> issues;
};

struct MapperContext {
    std::vector<MapperIssue> globalIssues;
    std::unordered_map<std::string, Coding> terminologyOverrides;
    std::string profileUrl = "http://example.org/fhir/StructureDefinition/si-listener-bundle";
    std::string deviceName = "SI-Listener";
    std::string modelName = "MedGemma";
    TerminologyClient* terminology = nullptr;  // nullable — null = offline mode
};

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string ltrim_copy(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return s.substr(first);
}

std::string strip_json_like_comments(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    bool in_string = false, escaped = false, in_line = false, in_block = false;
    for (size_t i = 0; i < input.size(); ++i) {
        const char c = input[i];
        const char next = (i + 1 < input.size()) ? input[i + 1] : '\0';
        if (in_line) { if (c == '\n') { in_line = false; out.push_back(c); } continue; }
        if (in_block) {
            if (c == '*' && next == '/') { in_block = false; ++i; continue; }
            if (c == '\n') out.push_back(c);
            continue;
        }
        if (in_string) {
            out.push_back(c);
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') { in_string = true; out.push_back(c); continue; }
        if (c == '/' && next == '/') { in_line = true; ++i; continue; }
        if (c == '/' && next == '*') { in_block = true; ++i; continue; }
        out.push_back(c);
    }
    return out;
}

bool isNonEmptyString(const json& j, const char* key) {
    return j.contains(key) && j[key].is_string() && !j[key].get<std::string>().empty();
}

std::string getString(const json& j, const char* key, const std::string& fallback = "") {
    if (isNonEmptyString(j, key)) return j[key].get<std::string>();
    return fallback;
}

bool looksUnknown(const std::string& s) {
    std::string x = lower(s);
    return x.empty() || x == "unknown" || x == "unk" || x == "n/a" || x == "na";
}

json ensureCodeableConcept(json value) {
    if (value.is_array() && !value.empty() && value[0].is_object()) return value[0];
    if (value.is_object()) return value;
    return json::object();
}

json makeCoding(const std::string& sys, const std::string& code, const std::string& display) {
    return json{{"system", sys}, {"code", code}, {"display", display}};
}

json makeCodeableConcept(const std::string& sys, const std::string& code, const std::string& display) {
    return json{{"coding", json::array({makeCoding(sys, code, display)})}, {"text", display}};
}

json makeIssueJson(const MapperIssue& issue) {
    json out{{"severity", issue.severity}, {"code", issue.code}, {"details", issue.details}};
    if (!issue.resourceId.empty()) out["resourceId"] = issue.resourceId;
    if (!issue.resourceType.empty()) out["resourceType"] = issue.resourceType;
    return out;
}

void addIssue(std::vector<MapperIssue>& issues,
              const std::string& sev, const std::string& code,
              const std::string& details, const json& resource = json::object()) {
    MapperIssue i;
    i.severity = sev; i.code = code; i.details = details;
    if (resource.is_object()) {
        i.resourceId = getString(resource, "id");
        i.resourceType = getString(resource, "resourceType");
    }
    issues.push_back(std::move(i));
}

std::string terminologySourceSuffix(const ResolvedCoding& coding) {
    return coding.from_network ? " (network)" : " (cache)";
}

void addTransactionRequest(json& entry, const std::string& rt, const std::optional<std::string>& id) {
    if (!entry.contains("request")) {
        if (id && !id->empty())
            entry["request"] = json{{"method", "PUT"}, {"url", rt + "/" + *id}};
        else
            entry["request"] = json{{"method", "POST"}, {"url", rt}};
    }
}

bool hasCodingCode(const json& cc, const std::string& code) {
    if (!cc.is_object() || !cc.contains("coding") || !cc["coding"].is_array()) return false;
    for (const auto& c : cc["coding"])
        if (getString(c, "code") == code) return true;
    return false;
}

std::optional<Coding> firstCoding(const json& cc) {
    if (!cc.is_object() || !cc.contains("coding") || !cc["coding"].is_array() ||
        cc["coding"].empty() || !cc["coding"][0].is_object())
        return std::nullopt;
    return Coding{getString(cc["coding"][0], "system"),
                  getString(cc["coding"][0], "code"),
                  getString(cc["coding"][0], "display")};
}

std::optional<double> getQuantityValue(const json& resource) {
    if (!resource.contains("valueQuantity") || !resource["valueQuantity"].is_object()) return std::nullopt;
    const auto& q = resource["valueQuantity"];
    if (q.contains("value") && q["value"].is_number()) return q["value"].get<double>();
    return std::nullopt;
}

std::string getQuantityUnit(const json& resource) {
    if (!resource.contains("valueQuantity") || !resource["valueQuantity"].is_object()) return "";
    const auto& q = resource["valueQuantity"];
    return getString(q, "unit", getString(q, "code"));
}

void normalizeQuantity(json& q) {
    if (!q.is_object()) return;
    std::string unit = getString(q, "unit");
    std::string code = getString(q, "code");
    if (unit == "/min" || unit == "per minute" || unit == "beats/min" || unit == "bpm") {
        q["unit"] = "beats/minute"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "/min"; return;
    }
    if (unit == "mm Hg" || unit == "mmHg" || code == "mm Hg" || code == "mmHg" || code == "mm[Hg]") {
        q["unit"] = "mmHg"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "mm[Hg]"; return;
    }
    if (unit == "%" || code == "%") {
        q["unit"] = "%"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "%"; return;
    }
    if (unit == "mmol/L" || code == "mmol/L") {
        q["unit"] = "mmol/L"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "mmol/L"; return;
    }
    if (unit == "mg/dL" || code == "mg/dL") {
        q["unit"] = "mg/dL"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "mg/dL"; return;
    }
    if (unit == "°C" || unit == "degC" || code == "Cel") {
        q["unit"] = "°C"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "Cel"; return;
    }
    if (unit == "°F" || unit == "degF") {
        q["unit"] = "°F"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "[degF]"; return;
    }
    if (unit == "kg" || code == "kg") {
        q["unit"] = "kg"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "kg"; return;
    }
    if (unit == "cm" || code == "cm") {
        q["unit"] = "cm"; q["system"] = "http://unitsofmeasure.org"; q["code"] = "cm"; return;
    }
}

void normalizeQuantityIfPresent(json& resource) {
    if (resource.contains("valueQuantity")) normalizeQuantity(resource["valueQuantity"]);
}

void normalizePatient(json& resource, std::vector<MapperIssue>& issues) {
    if (isNonEmptyString(resource, "birthDate") && looksUnknown(resource["birthDate"].get<std::string>())) {
        resource.erase("birthDate");
        addIssue(issues, "warning", "patient.birthDate.removed",
                 "Removed invalid Patient.birthDate placeholder value.", resource);
    }
}

void applyTerminologyOverride(json& cc, const Coding& coding,
                               std::vector<MapperIssue>& issues, const json& resource) {
    cc = makeCodeableConcept(coding.system, coding.code, coding.display);
    addIssue(issues, "info", "terminology.override", "Applied deterministic terminology override.", resource);
}

// === Observation Normalization (with live LOINC lookup) ==================
void normalizeObservationCode(json& resource, MapperContext& ctx, std::vector<MapperIssue>& issues) {
    if (!resource.contains("code")) {
        addIssue(issues, "error", "observation.code.missing", "Observation is missing code.", resource);
        return;
    }
    resource["code"] = ensureCodeableConcept(resource["code"]);

    std::string display;
    std::string code;
    std::string system;
    if (auto coding = firstCoding(resource["code"]); coding.has_value()) {
        display = lower(coding->display);
        code = coding->code;
        system = coding->system;
    }

    const std::string unit = lower(getQuantityUnit(resource));

    // 1. Check hardcoded overrides first (fastest, offline-safe)
    if (!code.empty()) {
        auto it = ctx.terminologyOverrides.find(code);
        if (it != ctx.terminologyOverrides.end()) {
            applyTerminologyOverride(resource["code"], it->second, issues, resource);
            normalizeQuantityIfPresent(resource);
            return;
        }
    }

    // 2. If code exists and system is LOINC, validate display via live lookup
    if (!code.empty() && (system == "http://loinc.org" || system.empty()) && ctx.terminology) {
        if (auto resolved = ctx.terminology->lookupLoinc(code); resolved.has_value()) {
            if (!resolved->display.empty()) {
                // Update display to canonical LOINC display
                resource["code"]["coding"][0]["display"] = resolved->display;
                resource["code"]["coding"][0]["system"] = "http://loinc.org";
                if (!resource["code"].contains("text") ||
                    resource["code"]["text"].get<std::string>().empty()) {
                    resource["code"]["text"] = resolved->display;
                }
                addIssue(issues, "info", "loinc.display.resolved",
                         "LOINC display name verified/updated via live terminology service"
                         + terminologySourceSuffix(*resolved) + ".",
                         resource);
                normalizeQuantityIfPresent(resource);
                return;
            }
        }
        // Code not found in LOINC — flag it
        addIssue(issues, "warning", "loinc.code.unverified",
                 "LOINC code '" + code + "' could not be verified against terminology service.", resource);
    }

    // 3. No code but we have a display — try LOINC text search
    if (code.empty() && !display.empty() && ctx.terminology) {
        if (auto resolved = ctx.terminology->searchLoincByDisplay(display, unit); resolved.has_value()) {
            resource["code"] = makeCodeableConcept(resolved->system, resolved->code, resolved->display);
            addIssue(issues, "warning", "loinc.code.inferred",
                     "LOINC code inferred from display text via terminology search"
                     + terminologySourceSuffix(*resolved) + ".",
                     resource);
            normalizeQuantityIfPresent(resource);
            return;
        }
    }

    // 4. Heuristic fallbacks (same as v1, offline-safe)
    if (unit.find("min") != std::string::npos || unit == "/min" || unit == "beats/minute" || unit == "bpm") {
        resource["code"] = makeCodeableConcept("http://loinc.org", "8867-4", "Heart rate");
        addIssue(issues, "warning", "observation.code.repaired",
                 "Repaired observation to Heart rate (LOINC 8867-4) based on unit heuristic.", resource);
        normalizeQuantityIfPresent(resource);
        return;
    }
    if (unit.find("mm") != std::string::npos && display.find("blood pressure") != std::string::npos) {
        normalizeQuantityIfPresent(resource);
        addIssue(issues, "info", "observation.bp.candidate",
                 "Observation flagged as blood-pressure candidate for panel grouping.", resource);
        return;
    }
    if (hasCodingCode(resource["code"], "8480-6") && display.find("heart") != std::string::npos) {
        resource["code"] = makeCodeableConcept("http://loinc.org", "8867-4", "Heart rate");
        addIssue(issues, "warning", "observation.code.repaired",
                 "Corrected mis-coded heart rate observation.", resource);
    }
    // Temperature heuristics
    if (unit == "°c" || unit == "degc" || unit == "cel") {
        if (code.empty()) {
            resource["code"] = makeCodeableConcept("http://loinc.org", "8310-5", "Body temperature");
            addIssue(issues, "warning", "observation.code.repaired",
                     "Inferred body temperature code from unit.", resource);
        }
    }
    // Oxygen saturation
    if (unit == "%" && (display.find("spo2") != std::string::npos ||
                         display.find("oxygen") != std::string::npos ||
                         display.find("saturation") != std::string::npos)) {
        resource["code"] = makeCodeableConcept("http://loinc.org", "59408-5",
                                                "Oxygen saturation in Arterial blood by Pulse oximetry");
        addIssue(issues, "warning", "observation.code.repaired",
                 "Inferred SpO2 code from display and unit.", resource);
    }
    normalizeQuantityIfPresent(resource);
}

// === Medication Normalization (with RxNorm lookup) =======================
void enrichMedicationCoding(json& medCC, MapperContext& ctx, std::vector<MapperIssue>& issues,
                              const json& resource) {
    if (!ctx.terminology) return;

    // Check if we already have an RxNorm coding
    if (medCC.contains("coding") && medCC["coding"].is_array()) {
        for (const auto& c : medCC["coding"]) {
            const std::string sys = getString(c, "system");
            if (sys == "http://www.nlm.nih.gov/research/umls/rxnorm") {
                // Already has RxNorm — validate it
                const std::string rxcui = getString(c, "code");
                if (!rxcui.empty()) {
                    addIssue(issues, "info", "rxnorm.code.present",
                             "RxNorm CUI already present: " + rxcui + ".", resource);
                }
                return;
            }
        }
    }

    // Try to find a drug name in text or existing display to look up
    std::string drug_name = getString(medCC, "text");
    if (drug_name.empty()) {
        if (auto coding = firstCoding(medCC); coding.has_value()) {
            drug_name = coding->display;
        }
    }
    if (drug_name.empty()) return;

    if (auto resolved = ctx.terminology->lookupRxNorm(drug_name); resolved.has_value()) {
        // Add RxNorm as an additional coding (preserve original)
        if (!medCC.contains("coding") || !medCC["coding"].is_array()) {
            medCC["coding"] = json::array();
        }
        medCC["coding"].push_back(makeCoding(resolved->system, resolved->code, resolved->display));
        // Update canonical display to RxNorm preferred name
        medCC["text"] = resolved->display;
        addIssue(issues, "info", "rxnorm.code.added",
                 "Added RxNorm CUI " + resolved->code + " for '" + drug_name + "'"
                 + terminologySourceSuffix(*resolved) + ".",
                 resource);
    } else {
        addIssue(issues, "warning", "rxnorm.lookup.failed",
                 "Could not resolve RxNorm code for drug '" + drug_name + "'.", resource);
    }
}

bool isLikelyGenericBloodPressure(const json& resource) {
    if (getString(resource, "resourceType") != "Observation") return false;
    if (!resource.contains("valueQuantity")) return false;
    const std::string unit = lower(getQuantityUnit(resource));
    if (unit.find("mm") == std::string::npos) return false;
    if (!resource.contains("code")) return false;
    const json cc = ensureCodeableConcept(resource["code"]);
    if (hasCodingCode(cc, "8480-6") || hasCodingCode(cc, "8462-4")) return true;
    if (cc.contains("coding") && cc["coding"].is_array() && !cc["coding"].empty()) {
        const std::string display = lower(getString(cc["coding"][0], "display"));
        return display.find("blood pressure") != std::string::npos;
    }
    return false;
}

std::string bpGroupKey(const json& resource) {
    return getString(resource["subject"], "reference") + "|" + getString(resource, "effectiveDateTime");
}

json makeBpPanel(const CandidateBp& a, const CandidateBp& b, int seq) {
    const CandidateBp* systolic = (a.value >= b.value) ? &a : &b;
    const CandidateBp* diastolic = (a.value >= b.value) ? &b : &a;
    return json{
        {"resourceType", "Observation"},
        {"id", "bp-panel-" + std::to_string(seq)},
        {"status", "final"},
        {"category", json::array({makeCodeableConcept(
            "http://terminology.hl7.org/CodeSystem/observation-category", "vital-signs", "Vital Signs")})},
        {"code", makeCodeableConcept("http://loinc.org", "85354-9",
                                      "Blood pressure panel with all children optional")},
        {"subject", json{{"reference", a.subjectRef}}},
        {"effectiveDateTime", a.effective},
        {"component", json::array({
            json{{"code", makeCodeableConcept("http://loinc.org", "8480-6", "Systolic blood pressure")},
                 {"valueQuantity", json{{"value", systolic->value}, {"unit", "mmHg"},
                                        {"system", "http://unitsofmeasure.org"}, {"code", "mm[Hg]"}}}},
            json{{"code", makeCodeableConcept("http://loinc.org", "8462-4", "Diastolic blood pressure")},
                 {"valueQuantity", json{{"value", diastolic->value}, {"unit", "mmHg"},
                                        {"system", "http://unitsofmeasure.org"}, {"code", "mm[Hg]"}}}}
        })}
    };
}

json convertMedicationRequestToAdministration(json resource, MapperContext& ctx,
                                               std::vector<MapperIssue>& issues) {
    json out;
    out["resourceType"] = "MedicationAdministration";
    if (isNonEmptyString(resource, "id"))
        out["id"] = resource["id"].get<std::string>() + "-admin";
    out["status"] = "completed";

    if (resource.contains("subject")) out["subject"] = resource["subject"];

    json medCC;
    if (resource.contains("code"))
        medCC = ensureCodeableConcept(resource["code"]);
    else if (resource.contains("medicationCodeableConcept"))
        medCC = ensureCodeableConcept(resource["medicationCodeableConcept"]);
    else
        addIssue(issues, "error", "medication.code.missing",
                 "MedicationRequest lacked medication coding.", resource);

    // Enrich with RxNorm if terminology client available
    if (!medCC.is_null() && !medCC.empty()) {
        enrichMedicationCoding(medCC, ctx, issues, resource);
        out["medicationCodeableConcept"] = medCC;
    }

    if (isNonEmptyString(resource, "occurrenceDateTime")) {
        out["effectiveDateTime"] = resource["occurrenceDateTime"];
        addIssue(issues, "info", "medication.request.converted",
                 "Converted MedicationRequest with occurrenceDateTime to MedicationAdministration.", resource);
    } else if (isNonEmptyString(resource, "authoredOn")) {
        out["effectiveDateTime"] = resource["authoredOn"];
    } else {
        addIssue(issues, "warning", "medication.time.missing",
                 "MedicationAdministration missing effectiveDateTime after conversion.", resource);
    }

    if (resource.contains("dosageInstruction") && resource["dosageInstruction"].is_array() &&
        !resource["dosageInstruction"].empty()) {
        const auto& first = resource["dosageInstruction"][0];
        if (isNonEmptyString(first, "text"))
            out["dosage"] = json{{"text", first["text"]}};
    }
    if (resource.contains("reasonCode")) out["reasonCode"] = resource["reasonCode"];
    return out;
}

void normalizeProcedure(json& resource, MapperContext& ctx, std::vector<MapperIssue>& issues) {
    if (!resource.contains("code")) return;
    resource["code"] = ensureCodeableConcept(resource["code"]);
    if (!resource["code"].contains("text") && resource["code"].contains("coding") &&
        resource["code"]["coding"].is_array() && !resource["code"]["coding"].empty()) {
        resource["code"]["text"] = getString(resource["code"]["coding"][0], "display");
    }

    const auto coding = firstCoding(resource["code"]);
    if (!coding.has_value()) return;

    // If procedure uses LOINC, flag it.
    if (coding->system == "http://loinc.org") {
        addIssue(issues, "warning", "procedure.code.suspicious-system",
                 "Procedure uses LOINC; consider SNOMED CT or a procedure coding system.", resource);
    }

    // If SNOMED code, validate via NLM.
    if (coding->system != "http://snomed.info/sct" || !ctx.terminology) return;
    if (auto resolved = ctx.terminology->lookupSnomed(coding->code); resolved.has_value()) {
        resource["code"]["coding"][0]["display"] = resolved->display;
        addIssue(issues, "info", "snomed.display.resolved",
                 "SNOMED CT display verified" + terminologySourceSuffix(*resolved) + ".",
                 resource);
    } else {
        addIssue(issues, "warning", "snomed.code.unverified",
                 "SNOMED CT code '" + coding->code + "' could not be verified.", resource);
    }
}

bool hasRequiredProfileFields(const json& resource, std::vector<MapperIssue>& issues) {
    const std::string type = getString(resource, "resourceType");

    const auto require_field = [&](const char* field,
                                   const std::string& issue_code,
                                   const std::string& message) {
        if (!resource.contains(field)) {
            addIssue(issues, "error", issue_code, message, resource);
            return false;
        }
        return true;
    };

    if (type == "Patient") {
        return require_field("name", "profile.patient.name.missing",
                             "Patient must include a name in this profile.");
    }
    if (type == "Observation") {
        bool ok = true;
        ok = require_field("status", "profile.observation.status.missing", "Observation missing status.") && ok;
        ok = require_field("code", "profile.observation.code.missing", "Observation missing code.") && ok;
        ok = require_field("subject", "profile.observation.subject.missing", "Observation missing subject.") && ok;
        return ok;
    }
    if (type == "MedicationAdministration") {
        bool ok = true;
        ok = require_field("medicationCodeableConcept",
                           "profile.medadmin.medication.missing",
                           "MedicationAdministration missing medicationCodeableConcept.") && ok;
        ok = require_field("subject",
                           "profile.medadmin.subject.missing",
                           "MedicationAdministration missing subject.") && ok;
        return ok;
    }
    return true;
}

bool isUncertainExtraction(const json& resource, std::vector<MapperIssue>& issues) {
    if (getString(resource, "resourceType") == "Observation" &&
        resource.contains("valueString") && resource["valueString"].is_string()) {
        const std::string v = lower(resource["valueString"].get<std::string>());
        if (v.find("possible") != std::string::npos || v.find("uncertain") != std::string::npos ||
            v.find("maybe") != std::string::npos) {
            addIssue(issues, "warning", "uncertainty.detected",
                     "Free-text value suggests uncertainty and requires review.", resource);
            return true;
        }
    }
    if (getString(resource, "resourceType") == "Observation" &&
        resource.contains("valueQuantity") && resource["valueQuantity"].is_object()) {
        if (auto coding = firstCoding(ensureCodeableConcept(resource.value("code", json::object())));
            coding.has_value()) {
            auto value = getQuantityValue(resource);
            if (value.has_value() && coding->code == "2711-2" && *value < 0) {
                addIssue(issues, "warning", "uncertainty.clinical-range",
                         "Negative bicarbonate value detected; flagged for review.", resource);
                return true;
            }
        }
    }
    return false;
}

void attachMetaProfile(json& resource, const MapperContext& ctx) {
    if (!resource.contains("meta") || !resource["meta"].is_object())
        resource["meta"] = json::object();
    if (!resource["meta"].contains("profile") || !resource["meta"]["profile"].is_array())
        resource["meta"]["profile"] = json::array();
    resource["meta"]["profile"].push_back(ctx.profileUrl + "/" + getString(resource, "resourceType"));
}

json buildProvenanceResource(const json& target, int seq, const MapperContext& ctx,
                              const std::vector<MapperIssue>& issues) {
    json reasonExt = json::array();
    for (const auto& issue : issues)
        reasonExt.push_back(json{{"url", "issue"}, {"valueString", issue.code + ": " + issue.details}});

    json agents = json::array({
        json{{"type", makeCodeableConcept("http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                                          "assembler", "Assembler")},
             {"who", json{{"display", ctx.deviceName + " deterministic mapper v2"}}}},
        json{{"type", makeCodeableConcept("http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                                          "author", "Author")},
             {"who", json{{"display", ctx.modelName}}}}
    });

    // Record whether live terminology was used
    if (ctx.terminology && ctx.terminology->networkEnabled()) {
        agents.push_back(json{
            {"type", makeCodeableConcept("http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                                         "verifier", "Verifier")},
            {"who", json{{"display", "LOINC FHIR R4 / NLM RxNorm / SNOMED CT terminology services"}}}
        });
    }

    return json{
        {"resourceType", "Provenance"},
        {"id", "prov-" + std::to_string(seq)},
        {"target", json::array({json{{"reference", getString(target, "resourceType") +
                                                     "/" + getString(target, "id")}}})},
        {"recorded", getString(target, "effectiveDateTime",
                               getString(target, "occurrenceDateTime", "2026-01-01T00:00:00Z"))},
        {"agent", agents},
        {"entity", json::array({
            json{{"role", "source"},     {"what", json{{"display", "Voice transcript extraction"}}}},
            json{{"role", "derivation"}, {"what", json{{"display", "LLM-generated intermediate JSON"}}}}
        })},
        {"extension", reasonExt}
    };
}

MapperResult normalizeResource(json resource, MapperContext& ctx) {
    MapperResult result;
    result.resource = std::move(resource);
    const std::string type = getString(result.resource, "resourceType");

    if (type == "Patient")
        normalizePatient(result.resource, result.issues);
    else if (type == "Observation")
        normalizeObservationCode(result.resource, ctx, result.issues);
    else if (type == "Procedure")
        normalizeProcedure(result.resource, ctx, result.issues);
    else if (type == "MedicationRequest")
        result.resource = convertMedicationRequestToAdministration(result.resource, ctx, result.issues);

    attachMetaProfile(result.resource, ctx);

    if (!hasRequiredProfileFields(result.resource, result.issues)) result.accepted = false;
    if (isUncertainExtraction(result.resource, result.issues))     result.accepted = false;

    return result;
}

json makeEntry(json resource) {
    json entry;
    entry["resource"] = std::move(resource);
    const auto& r = entry["resource"];
    const std::string type = getString(r, "resourceType");
    const std::optional<std::string> id =
        isNonEmptyString(r, "id") ? std::optional<std::string>(r["id"].get<std::string>()) : std::nullopt;
    addTransactionRequest(entry, type, id);
    return entry;
}

json buildOperationOutcome(const std::vector<MapperIssue>& issues) {
    json outcome{{"resourceType", "OperationOutcome"}, {"issue", json::array()}};
    for (const auto& issue : issues)
        outcome["issue"].push_back(json{
            {"severity", issue.severity == "error" ? "error" : "warning"},
            {"code", "processing"},
            {"details", json{{"text", issue.code + ": " + issue.details}}}
        });
    return outcome;
}

json mapBundle(const json& input, const std::string& modelName, TerminologyClient* terminology) {
    MapperContext ctx;
    // Hardcoded overrides retained as the always-reliable offline fallback
    ctx.terminologyOverrides["8480-6"] = Coding{"http://loinc.org", "8480-6", "Systolic blood pressure"};
    ctx.terminologyOverrides["8462-4"] = Coding{"http://loinc.org", "8462-4", "Diastolic blood pressure"};
    ctx.terminologyOverrides["8867-4"] = Coding{"http://loinc.org", "8867-4", "Heart rate"};
    ctx.terminologyOverrides["59408-5"] = Coding{"http://loinc.org", "59408-5",
        "Oxygen saturation in Arterial blood by Pulse oximetry"};
    ctx.terminologyOverrides["8310-5"] = Coding{"http://loinc.org", "8310-5", "Body temperature"};
    ctx.terminologyOverrides["29463-7"] = Coding{"http://loinc.org", "29463-7", "Body weight"};
    ctx.terminologyOverrides["8302-2"] = Coding{"http://loinc.org", "8302-2", "Body height"};
    ctx.terminologyOverrides["39156-5"] = Coding{"http://loinc.org", "39156-5", "Body mass index"};
    ctx.modelName = modelName;
    ctx.terminology = terminology;

    json accepted  = {{"resourceType","Bundle"}, {"type","transaction"}, {"entry", json::array()}};
    json rejected  = {{"resourceType","Bundle"}, {"type","collection"},  {"entry", json::array()}};

    if (!input.is_object() || !input.contains("entry") || !input["entry"].is_array())
        throw std::runtime_error("Input must be a Bundle-like object with an entry array.");

    std::unordered_map<std::string, std::vector<CandidateBp>> bpGroups;
    std::set<std::string> consumedIds;
    int bpSeq = 1, provSeq = 1;
    std::vector<MapperIssue> allIssues;

    // BP grouping pass (unchanged from v1)
    for (const auto& entry : input["entry"]) {
        if (!entry.contains("resource") || !entry["resource"].is_object()) continue;
        const auto& r = entry["resource"];
        if (!isLikelyGenericBloodPressure(r)) continue;
        auto value = getQuantityValue(r);
        if (!value.has_value()) continue;
        CandidateBp c;
        c.id = getString(r, "id");
        c.subjectRef = r.contains("subject") ? getString(r["subject"], "reference") : "";
        c.effective = getString(r, "effectiveDateTime");
        c.value = *value;
        c.unit = getQuantityUnit(r);
        c.original = r;
        bpGroups[bpGroupKey(r)].push_back(c);
    }

    // Main processing pass
    for (const auto& entry : input["entry"]) {
        if (!entry.contains("resource") || !entry["resource"].is_object()) continue;

        json resource = entry["resource"];
        const std::string id = getString(resource, "id");

        if (getString(resource, "resourceType") == "Observation" && isLikelyGenericBloodPressure(resource)) {
            const std::string key = bpGroupKey(resource);
            auto it = bpGroups.find(key);
            if (it != bpGroups.end() && it->second.size() == 2) {
                if (consumedIds.insert(it->second[0].id).second &&
                    consumedIds.insert(it->second[1].id).second) {
                    MapperResult result;
                    result.resource = makeBpPanel(it->second[0], it->second[1], bpSeq++);
                    attachMetaProfile(result.resource, ctx);
                    addIssue(result.issues, "info", "bp.panel.created",
                             "Created blood pressure panel from paired observations.", result.resource);
                    accepted["entry"].push_back(makeEntry(result.resource));
                    accepted["entry"].push_back(
                        makeEntry(buildProvenanceResource(result.resource, provSeq++, ctx, result.issues)));
                    for (const auto& issue : result.issues) allIssues.push_back(issue);
                }
                continue;
            }
        }

        if (!id.empty() && consumedIds.count(id)) continue;

        MapperResult result = normalizeResource(resource, ctx);
        for (const auto& issue : result.issues) allIssues.push_back(issue);

        if (result.accepted) {
            accepted["entry"].push_back(makeEntry(result.resource));
            accepted["entry"].push_back(
                makeEntry(buildProvenanceResource(result.resource, provSeq++, ctx, result.issues)));
        } else {
            rejected["entry"].push_back(json{{"resource", result.resource}});
        }
    }

    json output = {
        {"acceptedBundle", accepted},
        {"rejectedBundle", rejected},
        {"issues", json::array()},
        {"outcome", buildOperationOutcome(allIssues)},
        {"meta", json{
            {"terminologyNetworkUsed", terminology != nullptr && terminology->networkEnabled()},
            {"mapperVersion", "2.0.0"}
        }}
    };
    for (const auto& issue : allIssues) output["issues"].push_back(makeIssueJson(issue));
    return output;
}

} // namespace

// === Main (argument parsing) ==============================================

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr <<
                "Usage: " << argv[0] << " <input.json> [output.json]\n"
                "       --model-name <name>       (required)\n"
                "       --no-network              (disable live terminology lookup)\n"
                "       --cache-dir <path>        (default: ./terminology_cache)\n"
                "       --cache-ttl-days <n>      (default: 7)\n"
                "       --loinc-user <user>       (LOINC FHIR credentials, optional)\n"
                "       --loinc-pass <pass>\n"
                "       --timeout <seconds>       (HTTP timeout, default: 10)\n";
            return 1;
        }

        std::string inputPath, outputPath, modelName;
        TerminologyClient::Config termCfg;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model-name" && i + 1 < argc)   { modelName = argv[++i]; }
            else if (arg == "--no-network")               { termCfg.network_enabled = false; }
            else if (arg == "--cache-dir" && i + 1 < argc){ termCfg.cache_dir = argv[++i]; }
            else if (arg == "--cache-ttl-days" && i + 1 < argc) {
                try { termCfg.cache_ttl_days = std::stoi(argv[++i]); } catch (...) {}
            }
            else if (arg == "--loinc-user" && i + 1 < argc) { termCfg.loinc_user = argv[++i]; }
            else if (arg == "--loinc-pass" && i + 1 < argc) { termCfg.loinc_pass = argv[++i]; }
            else if (arg == "--timeout" && i + 1 < argc) {
                try { termCfg.timeout_seconds = std::stol(argv[++i]); } catch (...) {}
            }
            else if (inputPath.empty())  { inputPath  = arg; }
            else if (outputPath.empty()) { outputPath = arg; }
            else {
                std::cerr << "Error: unexpected argument: " << arg << "\n";
                return 1;
            }
        }

        if (inputPath.empty()) { std::cerr << "Error: missing input file path.\n"; return 1; }
        if (modelName.empty()) { std::cerr << "Error: --model-name is required.\n"; return 1; }

        // Initialise curl globally
        curl_global_init(CURL_GLOBAL_DEFAULT);

        TerminologyClient terminology(termCfg);

        if (termCfg.network_enabled) {
            std::cerr << "[INFO] Live terminology lookup enabled (LOINC FHIR R4, NLM RxNorm, SNOMED CT)\n";
            std::cerr << "[INFO] Cache directory: " << termCfg.cache_dir.string()
                      << " (TTL: " << termCfg.cache_ttl_days << " days)\n";
        } else {
            std::cerr << "[INFO] Offline mode: deterministic rules only (no network calls)\n";
        }

        std::ifstream in(inputPath);
        if (!in) { std::cerr << "Failed to open input file: " << inputPath << "\n"; return 1; }

        std::ostringstream raw;
        raw << in.rdbuf();
        const std::string no_comments = strip_json_like_comments(raw.str());

        std::ostringstream cleaned;
        std::istringstream stream(no_comments);
        std::string line;
        while (std::getline(stream, line)) {
            const std::string t = ltrim_copy(line);
            if (!t.empty()) cleaned << line << '\n';
        }

        json input;
        std::istringstream input_stream(cleaned.str());
        input_stream >> input;

        json output = mapBundle(input, modelName, &terminology);

        if (!outputPath.empty()) {
            std::ofstream out(outputPath);
            if (!out) { std::cerr << "Failed to open output file: " << outputPath << "\n"; return 1; }
            out << std::setw(2) << output << "\n";
        } else {
            std::cout << std::setw(2) << output << "\n";
        }

        curl_global_cleanup();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        curl_global_cleanup();
        return 2;
    }
}
