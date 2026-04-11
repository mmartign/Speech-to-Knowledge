// -*- coding: utf-8 -*-
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// Copyright (C) 2026 Spazio IT
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
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

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
    std::string severity;   // info | warning | error
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
};

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string ltrim_copy(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    return s.substr(first);
}

std::string strip_json_like_comments(const std::string& input) {
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

bool nearlyEqual(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

bool isNonEmptyString(const json& j, const char* key) {
    return j.contains(key) && j[key].is_string() && !j[key].get<std::string>().empty();
}

std::string getString(const json& j, const char* key, const std::string& fallback = "") {
    if (isNonEmptyString(j, key)) {
        return j[key].get<std::string>();
    }
    return fallback;
}

bool looksUnknown(const std::string& s) {
    std::string x = lower(s);
    return x.empty() || x == "unknown" || x == "unk" || x == "n/a" || x == "na";
}

json ensureCodeableConcept(json value) {
    if (value.is_array() && !value.empty() && value[0].is_object()) {
        return value[0];
    }
    if (value.is_object()) {
        return value;
    }
    return json::object();
}

json makeCoding(const std::string& system, const std::string& code, const std::string& display) {
    return json{{"system", system}, {"code", code}, {"display", display}};
}

json makeCodeableConcept(const std::string& system, const std::string& code, const std::string& display) {
    return json{{"coding", json::array({makeCoding(system, code, display)})}, {"text", display}};
}

json makeIssueJson(const MapperIssue& issue) {
    json out{{"severity", issue.severity}, {"code", issue.code}, {"details", issue.details}};
    if (!issue.resourceId.empty()) {
        out["resourceId"] = issue.resourceId;
    }
    if (!issue.resourceType.empty()) {
        out["resourceType"] = issue.resourceType;
    }
    return out;
}

void addIssue(std::vector<MapperIssue>& issues,
              const std::string& severity,
              const std::string& code,
              const std::string& details,
              const json& resource = json::object()) {
    MapperIssue i;
    i.severity = severity;
    i.code = code;
    i.details = details;
    if (resource.is_object()) {
        i.resourceId = getString(resource, "id");
        i.resourceType = getString(resource, "resourceType");
    }
    issues.push_back(std::move(i));
}

void addTransactionRequest(json& entry, const std::string& resourceType, const std::optional<std::string>& id) {
    if (!entry.contains("request")) {
        if (id && !id->empty()) {
            entry["request"] = json{{"method", "PUT"}, {"url", resourceType + "/" + *id}};
        } else {
            entry["request"] = json{{"method", "POST"}, {"url", resourceType}};
        }
    }
}

bool hasCodingCode(const json& cc, const std::string& code) {
    if (!cc.is_object() || !cc.contains("coding") || !cc["coding"].is_array()) {
        return false;
    }
    for (const auto& c : cc["coding"]) {
        if (getString(c, "code") == code) {
            return true;
        }
    }
    return false;
}

std::optional<Coding> firstCoding(const json& cc) {
    if (!cc.is_object() || !cc.contains("coding") || !cc["coding"].is_array() || cc["coding"].empty() || !cc["coding"][0].is_object()) {
        return std::nullopt;
    }
    return Coding{getString(cc["coding"][0], "system"), getString(cc["coding"][0], "code"), getString(cc["coding"][0], "display")};
}

std::optional<double> getQuantityValue(const json& resource) {
    if (!resource.contains("valueQuantity") || !resource["valueQuantity"].is_object()) {
        return std::nullopt;
    }
    const auto& q = resource["valueQuantity"];
    if (q.contains("value") && q["value"].is_number()) {
        return q["value"].get<double>();
    }
    return std::nullopt;
}

std::string getQuantityUnit(const json& resource) {
    if (!resource.contains("valueQuantity") || !resource["valueQuantity"].is_object()) {
        return "";
    }
    const auto& q = resource["valueQuantity"];
    return getString(q, "unit", getString(q, "code"));
}

void normalizeQuantity(json& q) {
    if (!q.is_object()) {
        return;
    }

    std::string unit = getString(q, "unit");
    std::string code = getString(q, "code");

    if (unit == "/min" || unit == "per minute" || unit == "beats/min" || unit == "bpm") {
        q["unit"] = "beats/minute";
        q["system"] = "http://unitsofmeasure.org";
        q["code"] = "/min";
        return;
    }

    if (unit == "mm Hg" || unit == "mmHg" || code == "mm Hg" || code == "mmHg" || code == "mm[Hg]") {
        q["unit"] = "mmHg";
        q["system"] = "http://unitsofmeasure.org";
        q["code"] = "mm[Hg]";
        return;
    }

    if (unit == "%" || code == "%") {
        q["unit"] = "%";
        q["system"] = "http://unitsofmeasure.org";
        q["code"] = "%";
        return;
    }

    if (unit == "mmol/L" || code == "mmol/L") {
        q["unit"] = "mmol/L";
        q["system"] = "http://unitsofmeasure.org";
        q["code"] = "mmol/L";
        return;
    }
}

void normalizePatient(json& resource, std::vector<MapperIssue>& issues) {
    if (isNonEmptyString(resource, "birthDate") && looksUnknown(resource["birthDate"].get<std::string>())) {
        resource.erase("birthDate");
        addIssue(issues, "warning", "patient.birthDate.removed", "Removed invalid Patient.birthDate placeholder value.", resource);
    }
}

void applyTerminologyOverride(json& cc, const Coding& coding, std::vector<MapperIssue>& issues, const json& resource) {
    cc = makeCodeableConcept(coding.system, coding.code, coding.display);
    addIssue(issues, "info", "terminology.override", "Applied deterministic terminology override.", resource);
}

void normalizeObservationCode(json& resource, MapperContext& ctx, std::vector<MapperIssue>& issues) {
    if (!resource.contains("code")) {
        addIssue(issues, "error", "observation.code.missing", "Observation is missing code.", resource);
        return;
    }
    resource["code"] = ensureCodeableConcept(resource["code"]);

    std::string display;
    std::string code;
    if (auto coding = firstCoding(resource["code"]); coding.has_value()) {
        display = lower(coding->display);
        code = coding->code;
    }

    const std::string unit = lower(getQuantityUnit(resource));

    if (!code.empty()) {
        auto it = ctx.terminologyOverrides.find(code);
        if (it != ctx.terminologyOverrides.end()) {
            applyTerminologyOverride(resource["code"], it->second, issues, resource);
            if (resource.contains("valueQuantity")) {
                normalizeQuantity(resource["valueQuantity"]);
            }
            return;
        }
    }

    if (unit.find("min") != std::string::npos || unit == "/min" || unit == "beats/minute" || unit == "bpm") {
        resource["code"] = makeCodeableConcept("http://loinc.org", "8867-4", "Heart rate");
        addIssue(issues, "warning", "observation.code.repaired", "Repaired observation to Heart rate based on unit heuristic.", resource);
        if (resource.contains("valueQuantity")) {
            normalizeQuantity(resource["valueQuantity"]);
        }
        return;
    }

    if (unit.find("mm") != std::string::npos && display.find("blood pressure") != std::string::npos) {
        if (resource.contains("valueQuantity")) {
            normalizeQuantity(resource["valueQuantity"]);
        }
        addIssue(issues, "info", "observation.bp.candidate", "Observation flagged as blood-pressure candidate for panel grouping.", resource);
        return;
    }

    if (hasCodingCode(resource["code"], "8480-6") && display.find("heart") != std::string::npos) {
        resource["code"] = makeCodeableConcept("http://loinc.org", "8867-4", "Heart rate");
        addIssue(issues, "warning", "observation.code.repaired", "Corrected mis-coded heart rate observation.", resource);
    }
    if (resource.contains("valueQuantity")) {
        normalizeQuantity(resource["valueQuantity"]);
    }
}

bool isLikelyGenericBloodPressure(const json& resource) {
    if (getString(resource, "resourceType") != "Observation") {
        return false;
    }
    if (!resource.contains("valueQuantity")) {
        return false;
    }
    const std::string unit = lower(getQuantityUnit(resource));
    if (unit.find("mm") == std::string::npos) {
        return false;
    }
    if (!resource.contains("code")) {
        return false;
    }
    const json cc = ensureCodeableConcept(resource["code"]);
    if (hasCodingCode(cc, "8480-6") || hasCodingCode(cc, "8462-4")) {
        return true;
    }
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
    const CandidateBp* systolic = &a;
    const CandidateBp* diastolic = &b;
    if (a.value < b.value) {
        systolic = &b;
        diastolic = &a;
    }

    return json{
        {"resourceType", "Observation"},
        {"id", "bp-panel-" + std::to_string(seq)},
        {"status", "final"},
        {"category", json::array({makeCodeableConcept("http://terminology.hl7.org/CodeSystem/observation-category", "vital-signs", "Vital Signs")})},
        {"code", makeCodeableConcept("http://loinc.org", "85354-9", "Blood pressure panel with all children optional")},
        {"subject", json{{"reference", a.subjectRef}}},
        {"effectiveDateTime", a.effective},
        {"component", json::array({
            json{{"code", makeCodeableConcept("http://loinc.org", "8480-6", "Systolic blood pressure")},
                 {"valueQuantity", json{{"value", systolic->value}, {"unit", "mmHg"}, {"system", "http://unitsofmeasure.org"}, {"code", "mm[Hg]"}}}},
            json{{"code", makeCodeableConcept("http://loinc.org", "8462-4", "Diastolic blood pressure")},
                 {"valueQuantity", json{{"value", diastolic->value}, {"unit", "mmHg"}, {"system", "http://unitsofmeasure.org"}, {"code", "mm[Hg]"}}}}
        })}
    };
}

json convertMedicationRequestToAdministration(json resource, std::vector<MapperIssue>& issues) {
    json out;
    out["resourceType"] = "MedicationAdministration";
    if (isNonEmptyString(resource, "id")) {
        out["id"] = resource["id"].get<std::string>() + "-admin";
    }
    out["status"] = "completed";

    if (resource.contains("subject")) {
        out["subject"] = resource["subject"];
    }

    if (resource.contains("code")) {
        out["medicationCodeableConcept"] = ensureCodeableConcept(resource["code"]);
    } else if (resource.contains("medicationCodeableConcept")) {
        out["medicationCodeableConcept"] = ensureCodeableConcept(resource["medicationCodeableConcept"]);
    } else {
        addIssue(issues, "error", "medication.code.missing", "MedicationRequest lacked medication coding.", resource);
    }

    if (isNonEmptyString(resource, "occurrenceDateTime")) {
        out["effectiveDateTime"] = resource["occurrenceDateTime"];
        addIssue(issues, "info", "medication.request.converted", "Converted MedicationRequest with occurrenceDateTime to MedicationAdministration.", resource);
    } else if (isNonEmptyString(resource, "authoredOn")) {
        out["effectiveDateTime"] = resource["authoredOn"];
    } else {
        addIssue(issues, "warning", "medication.time.missing", "MedicationAdministration missing effectiveDateTime after conversion.", resource);
    }

    if (resource.contains("dosageInstruction") && resource["dosageInstruction"].is_array() && !resource["dosageInstruction"].empty()) {
        const auto& first = resource["dosageInstruction"][0];
        if (isNonEmptyString(first, "text")) {
            out["dosage"] = json{{"text", first["text"]}};
        }
    }

    if (resource.contains("reasonCode")) {
        out["reasonCode"] = resource["reasonCode"];
    }

    return out;
}

void normalizeProcedure(json& resource, std::vector<MapperIssue>& issues) {
    if (resource.contains("code")) {
        resource["code"] = ensureCodeableConcept(resource["code"]);
        if (!resource["code"].contains("text") && resource["code"].contains("coding") && resource["code"]["coding"].is_array() && !resource["code"]["coding"].empty()) {
            resource["code"]["text"] = getString(resource["code"]["coding"][0], "display");
        }
        if (auto coding = firstCoding(resource["code"]); coding.has_value()) {
            if (coding->system == "http://loinc.org") {
                addIssue(issues, "warning", "procedure.code.suspicious-system", "Procedure uses LOINC; consider SNOMED CT or a procedure coding system.", resource);
            }
        }
    }
}

bool hasRequiredProfileFields(const json& resource, std::vector<MapperIssue>& issues) {
    const std::string type = getString(resource, "resourceType");

    if (type == "Patient") {
        if (!resource.contains("name")) {
            addIssue(issues, "error", "profile.patient.name.missing", "Patient must include a name in this profile.", resource);
            return false;
        }
        return true;
    }

    if (type == "Observation") {
        bool ok = true;
        if (!resource.contains("status")) {
            addIssue(issues, "error", "profile.observation.status.missing", "Observation missing status.", resource);
            ok = false;
        }
        if (!resource.contains("code")) {
            addIssue(issues, "error", "profile.observation.code.missing", "Observation missing code.", resource);
            ok = false;
        }
        if (!resource.contains("subject")) {
            addIssue(issues, "error", "profile.observation.subject.missing", "Observation missing subject.", resource);
            ok = false;
        }
        return ok;
    }

    if (type == "MedicationAdministration") {
        bool ok = true;
        if (!resource.contains("medicationCodeableConcept")) {
            addIssue(issues, "error", "profile.medadmin.medication.missing", "MedicationAdministration missing medicationCodeableConcept.", resource);
            ok = false;
        }
        if (!resource.contains("subject")) {
            addIssue(issues, "error", "profile.medadmin.subject.missing", "MedicationAdministration missing subject.", resource);
            ok = false;
        }
        return ok;
    }

    return true;
}

bool isUncertainExtraction(const json& resource, std::vector<MapperIssue>& issues) {
    if (getString(resource, "resourceType") == "Observation" && resource.contains("valueString") && resource["valueString"].is_string()) {
        const std::string v = lower(resource["valueString"].get<std::string>());
        if (v.find("possible") != std::string::npos || v.find("uncertain") != std::string::npos || v.find("maybe") != std::string::npos) {
            addIssue(issues, "warning", "uncertainty.detected", "Free-text value suggests uncertainty and requires review.", resource);
            return true;
        }
    }
    if (getString(resource, "resourceType") == "Observation" && resource.contains("valueQuantity") && resource["valueQuantity"].is_object()) {
        if (auto coding = firstCoding(ensureCodeableConcept(resource.value("code", json::object()))); coding.has_value()) {
            auto value = getQuantityValue(resource);
            if (value.has_value() && coding->code == "2711-2" && *value < 0) {
                addIssue(issues, "warning", "uncertainty.clinical-range", "Negative bicarbonate value detected; flagged for review.", resource);
                return true;
            }
        }
    }
    return false;
}

void attachMetaProfile(json& resource, const MapperContext& ctx) {
    if (!resource.contains("meta") || !resource["meta"].is_object()) {
        resource["meta"] = json::object();
    }
    if (!resource["meta"].contains("profile") || !resource["meta"]["profile"].is_array()) {
        resource["meta"]["profile"] = json::array();
    }
    resource["meta"]["profile"].push_back(ctx.profileUrl + "/" + getString(resource, "resourceType"));
}

json buildProvenanceResource(const json& targetResource, int seq, const MapperContext& ctx, const std::vector<MapperIssue>& issues) {
    json reasonExt = json::array();
    for (const auto& issue : issues) {
        reasonExt.push_back(json{{"url", "issue"}, {"valueString", issue.code + ": " + issue.details}});
    }

    return json{
        {"resourceType", "Provenance"},
        {"id", "prov-" + std::to_string(seq)},
        {"target", json::array({json{{"reference", getString(targetResource, "resourceType") + "/" + getString(targetResource, "id")}}})},
        {"recorded", getString(targetResource, "effectiveDateTime", getString(targetResource, "occurrenceDateTime", "2026-01-01T00:00:00Z"))},
        {"agent", json::array({
            json{{"type", makeCodeableConcept("http://terminology.hl7.org/CodeSystem/provenance-participant-type", "assembler", "Assembler")},
                 {"who", json{{"display", ctx.deviceName + " deterministic mapper"}}}},
            json{{"type", makeCodeableConcept("http://terminology.hl7.org/CodeSystem/provenance-participant-type", "author", "Author")},
                 {"who", json{{"display", ctx.modelName}}}}
        })},
        {"entity", json::array({
            json{{"role", "source"}, {"what", json{{"display", "Voice transcript extraction"}}}},
            json{{"role", "derivation"}, {"what", json{{"display", "LLM-generated intermediate JSON"}}}}
        })},
        {"extension", reasonExt}
    };
}

MapperResult normalizeResource(json resource, MapperContext& ctx) {
    MapperResult result;
    result.resource = std::move(resource);

    const std::string type = getString(result.resource, "resourceType");

    if (type == "Patient") {
        normalizePatient(result.resource, result.issues);
    } else if (type == "Observation") {
        normalizeObservationCode(result.resource, ctx, result.issues);
    } else if (type == "Procedure") {
        normalizeProcedure(result.resource, result.issues);
    } else if (type == "MedicationRequest") {
        result.resource = convertMedicationRequestToAdministration(result.resource, result.issues);
    }

    attachMetaProfile(result.resource, ctx);

    const bool profileOk = hasRequiredProfileFields(result.resource, result.issues);
    const bool uncertain = isUncertainExtraction(result.resource, result.issues);

    if (!profileOk) {
        result.accepted = false;
    }
    if (uncertain) {
        result.accepted = false;
    }

    return result;
}

json makeEntry(json resource) {
    json entry;
    entry["resource"] = std::move(resource);
    const auto& r = entry["resource"];
    const std::string type = getString(r, "resourceType");
    const std::optional<std::string> id = isNonEmptyString(r, "id") ? std::optional<std::string>(r["id"].get<std::string>()) : std::nullopt;
    addTransactionRequest(entry, type, id);
    return entry;
}

json buildOperationOutcome(const std::vector<MapperIssue>& issues) {
    json outcome{{"resourceType", "OperationOutcome"}, {"issue", json::array()}};
    for (const auto& issue : issues) {
        outcome["issue"].push_back(json{{"severity", issue.severity == "error" ? "error" : "warning"},
                                         {"code", "processing"},
                                         {"details", json{{"text", issue.code + ": " + issue.details}}}});
    }
    return outcome;
}

json mapBundle(const json& input, const std::string& modelName) {
    MapperContext ctx;
    ctx.terminologyOverrides["8480-6"] = Coding{"http://loinc.org", "8480-6", "Systolic blood pressure"};
    ctx.terminologyOverrides["8462-4"] = Coding{"http://loinc.org", "8462-4", "Diastolic blood pressure"};
    ctx.modelName = modelName;

    json accepted = {
        {"resourceType", "Bundle"},
        {"type", "transaction"},
        {"entry", json::array()}
    };
    json rejected = {
        {"resourceType", "Bundle"},
        {"type", "collection"},
        {"entry", json::array()}
    };

    if (!input.is_object() || !input.contains("entry") || !input["entry"].is_array()) {
        throw std::runtime_error("Input must be a Bundle-like object with an entry array.");
    }

    std::unordered_map<std::string, std::vector<CandidateBp>> bpGroups;
    std::set<std::string> consumedIds;
    int bpSeq = 1;
    int provSeq = 1;
    std::vector<MapperIssue> allIssues;

    for (const auto& entry : input["entry"]) {
        if (!entry.contains("resource") || !entry["resource"].is_object()) {
            continue;
        }
        const auto& r = entry["resource"];
        if (!isLikelyGenericBloodPressure(r)) {
            continue;
        }
        auto value = getQuantityValue(r);
        if (!value.has_value()) {
            continue;
        }
        CandidateBp c;
        c.id = getString(r, "id");
        c.subjectRef = r.contains("subject") ? getString(r["subject"], "reference") : "";
        c.effective = getString(r, "effectiveDateTime");
        c.value = *value;
        c.unit = getQuantityUnit(r);
        c.original = r;
        bpGroups[bpGroupKey(r)].push_back(c);
    }

    for (const auto& entry : input["entry"]) {
        if (!entry.contains("resource") || !entry["resource"].is_object()) {
            continue;
        }

        json resource = entry["resource"];
        const std::string id = getString(resource, "id");

        if (getString(resource, "resourceType") == "Observation" && isLikelyGenericBloodPressure(resource)) {
            const std::string key = bpGroupKey(resource);
            auto it = bpGroups.find(key);
            if (it != bpGroups.end() && it->second.size() == 2) {
                if (consumedIds.insert(it->second[0].id).second && consumedIds.insert(it->second[1].id).second) {
                    MapperResult result;
                    result.resource = makeBpPanel(it->second[0], it->second[1], bpSeq++);
                    attachMetaProfile(result.resource, ctx);
                    addIssue(result.issues, "info", "bp.panel.created", "Created blood pressure panel from paired observations.", result.resource);
                    accepted["entry"].push_back(makeEntry(result.resource));
                    accepted["entry"].push_back(makeEntry(buildProvenanceResource(result.resource, provSeq++, ctx, result.issues)));
                    for (const auto& issue : result.issues) {
                        allIssues.push_back(issue);
                    }
                }
                continue;
            }
        }

        if (!id.empty() && consumedIds.count(id)) {
            continue;
        }

        MapperResult result = normalizeResource(resource, ctx);
        for (const auto& issue : result.issues) {
            allIssues.push_back(issue);
        }

        if (result.accepted) {
            accepted["entry"].push_back(makeEntry(result.resource));
            accepted["entry"].push_back(makeEntry(buildProvenanceResource(result.resource, provSeq++, ctx, result.issues)));
        } else {
            rejected["entry"].push_back(json{{"resource", result.resource}});
        }
    }

    json output = {
        {"acceptedBundle", accepted},
        {"rejectedBundle", rejected},
        {"issues", json::array()},
        {"outcome", buildOperationOutcome(allIssues)}
    };

    for (const auto& issue : allIssues) {
        output["issues"].push_back(makeIssueJson(issue));
    }

    return output;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "Usage: " << argv[0] << " <input.json> [output.json] --model-name <name>\n";
            return 1;
        }

        std::string inputPath;
        std::string outputPath;
        std::string modelName;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model-name") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: --model-name requires a value.\n";
                    return 1;
                }
                modelName = argv[++i];
                continue;
            }

            if (inputPath.empty()) {
                inputPath = arg;
            } else if (outputPath.empty()) {
                outputPath = arg;
            } else {
                std::cerr << "Error: unexpected argument: " << arg << "\n";
                return 1;
            }
        }

        if (inputPath.empty()) {
            std::cerr << "Error: missing input file path.\n";
            return 1;
        }
        if (modelName.empty()) {
            std::cerr << "Error: missing required --model-name argument.\n";
            return 1;
        }

        std::ifstream in(inputPath);
        if (!in) {
            std::cerr << "Failed to open input file: " << inputPath << "\n";
            return 1;
        }

        std::ostringstream raw_json;
        raw_json << in.rdbuf();
        const std::string no_comments = strip_json_like_comments(raw_json.str());

        std::ostringstream cleaned_json;
        std::istringstream no_comments_stream(no_comments);
        std::string line;
        while (std::getline(no_comments_stream, line)) {
            const std::string trimmed = ltrim_copy(line);
            if (trimmed.empty()) {
                continue;
            }
            cleaned_json << line << '\n';
        }

        json input;
        std::istringstream cleaned_input_stream(cleaned_json.str());
        cleaned_input_stream >> input;

        json output = mapBundle(input, modelName);

        if (!outputPath.empty()) {
            std::ofstream out(outputPath);
            if (!out) {
                std::cerr << "Failed to open output file: " << outputPath << "\n";
                return 1;
            }
            out << std::setw(2) << output << "\n";
        } else {
            std::cout << std::setw(2) << output << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }
}
