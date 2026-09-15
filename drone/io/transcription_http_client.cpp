// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Copyright (c) 2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
#include "transcription_http_client.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace drone::io {

namespace {

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

}  // namespace

std::string transcribeWavFile(
    const drone::speech_core::AsrConfig& cfg,
    const std::string& wavPath)
{
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        throw std::runtime_error("curl_easy_init() failed");
    }

    curl_mime* mime = curl_mime_init(curl);
    curl_mimepart* filePart = curl_mime_addpart(mime);
    curl_mime_name(filePart, "file");
    curl_mime_filedata(filePart, wavPath.c_str());

    curl_mimepart* modelPart = curl_mime_addpart(mime);
    curl_mime_name(modelPart, "model");
    curl_mime_data(modelPart, cfg.modelName.c_str(), CURL_ZERO_TERMINATED);

    const std::string url = cfg.baseUrl + "audio/transcriptions";
    std::string responseBody;

    curl_slist* headers = nullptr;
    std::string authHeader;
    if (!cfg.apiKey.empty()) {
        authHeader = "Authorization: Bearer " + cfg.apiKey;
        headers = curl_slist_append(headers, authHeader.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    const CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_mime_free(mime);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(
            std::string("ASR request failed: ") + curl_easy_strerror(res));
    }

    nlohmann::json response;
    try {
        response = nlohmann::json::parse(responseBody);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "ASR server response is not valid JSON: " + responseBody);
    }

    if (response.count("error") != 0) {
        throw std::runtime_error(
            "ASR server returned an error: " + response["error"].dump());
    }

    return drone::speech_core::extractTranscriptText(response);
}

}  // namespace drone::io
