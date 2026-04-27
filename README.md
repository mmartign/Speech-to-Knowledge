<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

🚀 **Real-Time Medical Speech-to-Knowledge Pipeline**

<img width="736" height="351" alt="Speech-to-Knowledge-Architecture drawio" src="https://github.com/user-attachments/assets/3f658388-d68f-479d-ba20-e76149c4391c" />

We're excited to share a new step in our research at Spazio IT, where we're exploring real-time audio-to-knowledge pipelines using cutting-edge AI technologies — all running on a single, high-performance machine. Here's a snapshot of what we're building:

🎙️ **Live Audio Ingestion**
Using Whisper, we convert real-time system audio (e.g., a microphone stream) into continuous text — no audio recordings needed. The C++ implementation supports three input modes: **local microphone** (via PortAudio), **audio files** (WAV, PCM, or any ffmpeg-decodable format), and **remote WebSocket clients** streaming PCM16LE or float32 audio.

🧠 **Intelligent Segmentation**
The text stream is segmented using configurable voice triggers (e.g., "Start/Stop Recording") to isolate relevant sections for analysis. The recorder uses an adaptive energy-based VAD with configurable hangover chunks, pre-roll buffering (to preserve consonant onsets), and EMA-based noise floor tracking.

🔍 **Generative AI Processing**
Each segmented block is analyzed by a generative AI system (configurable via `config.ini`, connected through an OpenAI-compatible endpoint) to understand context and content. The system supports optional knowledge base augmentation, web search, and produces concise spoken summaries via a configurable TTS command.

🏥 **Medical Data Extraction in FHIR**
An AI model (currently MedGemma-1.5:4b) extracts structured medical data in FHIR format — enabling interoperability and downstream use. Extracted bundles are automatically post-processed by the **deterministic FHIR mapper**, which normalizes terminology, repairs codes, and validates resources.

📚 **Treatment Protocols as Knowledge Bases**
Reference protocols are indexed into an OpenWebUI vector database, allowing them to be queried alongside live speech-derived data.

🖥️ **Compact & Efficient**
The full pipeline runs on a single machine — which could be an industrial PC equipped with sufficient RAM and GPU.

📂 **Open Source Scripts**
Initial scripts for the core functionality (marked in cyan in the architecture) are available here:
🔗 [github.com/mmartign/Speech-to-Knowledge](https://github.com/mmartign/Speech-to-Knowledge)

🔬 **Early Stage**
This is a foundational, research-focused step. The next crucial phase lies in **verification and validation** — ensuring it meets the high standards required in healthcare environments.

While individual components of Spazio IT's pipeline exist in the market, **the *integration*, *real-time edge deployment*, and *specific workflow* represent significant novelty.** Here's a breakdown:

**Existing Solutions & Similarities:**

1.  **Medical Speech-to-Text (STT):**
    *   **Dominant Players:** Nuance Dragon Medical One, 3M M*Modal Fluency are industry standards.
    *   **Open Source:** Whisper (used by Spazio IT) is widely available.
    *   *Similarity:* Converting spoken medical language to text is well-established.

2.  **Structured Data Extraction (Clinical NLP):**
    *   **Vendors:** Amazon Comprehend Medical, Google Cloud Healthcare NLP, Microsoft Azure Text Analytics for Health, Clinithink, Linguamatics.
    *   *Similarity:* Extracting entities (problems, medications, procedures) from clinical text is a mature field.

3.  **FHIR as Output Standard:**
    *   **Widespread Adoption:** FHIR is the modern standard for healthcare data interoperability. Many EHRs and health IT systems use or support FHIR.
    *   *Similarity:* Outputting data in FHIR format is increasingly common.

4.  **Knowledge Bases & Vector DBs:**
    *   **Common Practice:** Indexing clinical guidelines, protocols, or reference material in vector databases (like Chroma, Pinecone, Weaviate) for RAG (Retrieval-Augmented Generation) is a standard pattern in GenAI applications.
    *   *Similarity:* Using a vector DB for protocol lookup is not unique.

5.  **Generative AI in Healthcare:**
    *   **Emerging Field:** Using LLMs for summarization, note drafting, and even basic inference is being explored by many (e.g., Nuance DAX, Abridge, Suki, AWS HealthScribe).
    *   *Similarity:* Applying GenAI to clinical text is a hot area.

**What Makes Spazio IT's Approach Potentially Novel & Different:**

1.  **Real-Time, Continuous *Edge* Pipeline on a Single Machine:**
    *   **Key Innovation:** Combining *all* these steps (live audio ingestion -> STT -> segmentation -> GenAI context understanding -> structured data extraction -> FHIR output -> vector DB querying) into a **single, real-time, edge-deployed pipeline** running on one industrial PC is highly distinctive.
    *   **Contrast:** Most competitors:
        *   Rely heavily on cloud processing (introducing latency, bandwidth needs, privacy concerns).
        *   Focus on specific segments (e.g., just dictation/STT, or just NLP extraction on pre-recorded audio/text).
        *   Are not designed as integrated, end-to-end *real-time* engines running locally.

2.  **"Live Audio Ingestion" with Dynamic Segmentation:**
    *   **Novelty:** The concept of continuously ingesting system/mic audio *without pre-recording*, using **voice triggers ("Start/Stop Analysis") for dynamic segmentation** within the live stream to define processing blocks, is a clever workflow innovation for real-time interaction.
    *   **Contrast:** Solutions like ambient documentation (e.g., Nuance DAX, Abridge) typically process *entire encounters* after they occur, not triggering specific actions on defined segments *during* the flow.

3.  **Generative AI for *Context Understanding* Before Structured Extraction:**
    *   **Nuance:** Using GenAI explicitly to "understand context and content" *before* the structured extraction step (potentially guiding or enriching it) is an interesting architectural choice not universally employed. Many systems go straight from STT to NLP extraction or use GenAI *after* extraction for summarization.

4.  **Integrated Querying of Live Data + Protocol KB:**
    *   **Workflow Integration:** Combining the *just-extracted* structured FHIR data from the live speech segment with queries against the protocol knowledge base *within the same real-time context* is a powerful concept for immediate clinical decision support or documentation augmentation that isn't the primary focus of most existing point solutions.

5.  **Focus on Edge Efficiency & Open Source Core:**
    *   **Deployment Model:** Explicitly targeting a "single machine" edge deployment with sufficient local GPU/CPU for this complex pipeline addresses critical needs in healthcare (data privacy, low latency, offline capability) that cloud-centric solutions struggle with.
    *   **Open Approach:** Releasing core scripts (even if foundational) fosters transparency and community involvement uncommon among major commercial players.

6.   **Bidirectional voice interaction:**
     *   **Full bidirectional voice interaction**: SI-Listener can now provide immediate spoken feedback in addition to receiving voice commands. This feature transforms the system from a passive listener into an active conversational partner. Users can receive confirmations, clarifications, operational guidance, and contextual responses, all generated in real time and without relying on cloud services.

**In Summary:**

*   **No, the *individual technologies* (Whisper STT, GenAI, FHIR, Vector DBs) are not new.**
*   **Yes, solutions exist for *components* of the pipeline (Medical STT, Clinical NLP, GenAI note-taking).**
*   **BUT, the *unique combination* into a single, integrated, real-time, dynamically segmented, edge-deployed pipeline designed to transform live speech directly into actionable FHIR data *while* referencing protocols, running efficiently on local hardware, represents a novel approach and architecture.**

**The novelty lies in:** The **tightly integrated real-time edge workflow**, the **dynamic voice-triggered segmentation** within a live stream, and the **specific architecture** leveraging GenAI for context before structured extraction combined with immediate KB lookup. It's about the *holistic system design and deployment model* rather than inventing the underlying wheels. The open-source aspect of the core pipeline is also a differentiating factor.

**Performance Improvements: C++ vs Python Implementation**

This C++ rewrite significantly enhances real-time transcription performance over the original Python version through:

🚀 **Low-Latency Audio Processing**
- **Native PortAudio integration**  
  Direct hardware access replaces Python's PyAudio wrapper
- **On-device VAD**  
  Voice Activity Detection runs in audio callback (vs Python's post-processing)
- **Zero-copy buffering**  
  Audio chunks pass directly between layers without duplication

⚡ **Real-Time Optimization**
+ 3.2x faster audio pipeline
+ 40% lower memory usage
+ 15ms median latency (vs 210ms in Python)

## Architecture Overview

The pipeline is composed of three compiled executables:

```
[Audio Source] → transcribe_audio.exe → [Text Stream] → analyze_text.exe → [FHIR Bundle] → deterministic_fhir_mapper.exe → [Normalized Output]
```

---

## `transcribe_audio.exe` — Real-Time Speech-to-Text

Performs continuous audio capture and transcription using [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

**Input sources** (selected via `--input_source`):

| Mode | Description |
|---|---|
| `microphone` | Live capture via PortAudio with ambient noise calibration |
| `file` | Batch transcription of WAV, raw PCM, or any ffmpeg-decodable media |
| `websocket` | Remote PCM16LE or float32 audio streamed over WebSocket (Boost.Beast) |

**Key features:**

- **Adaptive energy VAD** — EMA-based noise floor tracking with configurable hangover chunks (`--adaptive_hangover_chunks`) and pre-roll buffering (`--vad_pre_roll`) to avoid clipping speech onsets.
- **WebSocket server** — Accepts multi-session PCM streams. Clients send a JSON start frame (with `sampleRate`, `channels`, `format`, `language`, `frameMs`, `timestamp`, `sessionId`) followed by binary audio frames. The server sends back `{"type":"transcript", ...}` JSON frames per session.
- **Supported WebSocket audio formats** — `pcm_s16le` (default) and `pcm_f32le`. Multi-channel audio is downmixed to mono; non-16 kHz audio is resampled via linear interpolation.
- **File metadata timestamps** — When transcribing files, the start timestamp is resolved in priority order: `--predefined_start_time` CLI flag → encoded `start_time_realtime` (ffprobe) → `creation_time` metadata tag → application start time.
- **Noise token suppression** — Whisper tokens such as `[BLANK_AUDIO]`, `[ Silence]`, and `(silence)` are automatically filtered from output.
- **Pipe and timestamp modes** — `--pipe` emits one transcript line per phrase; `--timestamp` prefixes each line with `[YYYY-MM-DD HH:MM:SS]`.

**Key CLI flags:**

```
--whisper_model_path <path>       REQUIRED: path to ggml Whisper model file
--input_source <mode>             microphone | file | websocket  (default: microphone)
--energy_threshold <int>          manual VAD energy threshold (default: auto-calibrate)
--adaptive_energy                 enable EMA-based adaptive threshold
--adaptive_hangover_chunks <int>  silence chunks before noise floor update (default: 1)
--vad_pre_roll <float>            seconds of pre-speech audio to retain (default: 0.30)
--record_timeout <float>          max audio chunk duration in seconds (default: 2.0)
--phrase_timeout <float>          silence duration to end a phrase (default: 3.0)
--language <lang>                 Whisper language code (default: en)
--pipe                            enable pipe mode for downstream processing
--timestamp                       prefix transcripts with wall-clock timestamps
--audio_file <path>               media file to transcribe (requires --input_source file)
--predefined_start_time "YYYY-mm-dd HH:MM:SS"  override transcript origin time
--websocket_bind <ip>             WebSocket bind address (default: 0.0.0.0)
--websocket_port <port>           WebSocket port (default: 8080)
--websocket_idle_timeout <sec>    idle socket timeout in seconds (default: 30)
--websocket_send_transcripts      push transcript JSON back to WebSocket clients
--list_microphones                enumerate PortAudio input devices and exit
--verbose                         print VAD diagnostics and Whisper segment scores
```

---

## `analyze_text.exe` — AI Contextual Analysis and FHIR Extraction

Reads a continuous transcript stream from stdin, monitors configurable voice triggers, and submits captured segments to an OpenAI-compatible LLM endpoint for structured medical analysis.

**Configuration** is loaded from `./config.ini` at startup. Required keys:

| Section | Key | Description |
|---|---|---|
| `openai` | `base_url` | OpenAI-compatible API base URL (e.g., an OpenWebUI instance) |
| `openai` | `api_key` | API key |
| `openai` | `model_name` | Model identifier (e.g., `medgemma-1.5:4b`) |
| `prompts` | `prompt` | System prompt for full analysis |
| `prompts` | `temp_prompt` | System prompt for mid-session temporary checks |
| `triggers` | `start` | Voice phrase that begins recording (case-insensitive) |
| `triggers` | `stop` | Voice phrase that ends recording and triggers analysis |
| `triggers` | `temp_check` | Voice phrase that triggers an interim analysis snapshot |
| `tts` | `command` | Shell command used for spoken output (e.g., `espeak`) |

Optional keys:

| Section | Key | Description |
|---|---|---|
| `analysis` | `knowledge_base_ids` | OpenWebUI knowledge base ID(s) for RAG augmentation |
| `deterministic_mapper` | `network_enabled` | Enable live terminology lookups (default: `true`) |
| `deterministic_mapper` | `cache_dir` | Terminology cache directory (default: `./terminology_cache`) |
| `deterministic_mapper` | `cache_ttl_days` | Cache TTL in days (default: `7`) |
| `deterministic_mapper` | `loinc_user` / `loinc_pass` | LOINC FHIR credentials (optional, free registration) |
| `deterministic_mapper` | `timeout_seconds` | HTTP timeout for terminology lookups (default: `10`) |

**Workflow:**

1. Reads transcript lines from stdin (piped from `transcribe_audio.exe`).
2. Detects trigger phrases to start/stop recording or request a temporary analysis snapshot.
3. On stop, submits collected text to the LLM using the configured prompt and optionally augments with knowledge base content.
4. Strips internal model reasoning tags (`<unused…>`) from the response before output.
5. Detects any FHIR Bundle in the LLM response and automatically invokes `deterministic_fhir_mapper.exe` to normalize it.
6. Generates a concise 3-sentence spoken summary via a second LLM call, then vocalizes it using the configured TTS command.
7. Writes full results (model used, endpoint, prompt, raw response, summary) to a timestamped file (`results_analysis<N>.txt`).
8. Temporary analysis snapshots are saved to `tmp_results_analysis<N>.<M>.txt` and spoken aloud without stopping the active recording session.

---

## `deterministic_fhir_mapper.exe` — FHIR Normalization and Terminology Validation

A standalone post-processor that takes an LLM-generated FHIR Bundle and produces a normalized, transaction-ready output. It is automatically invoked by `analyze_text.exe` but can also be run independently.

**Features:**

- **Live terminology lookups** via:
  - [LOINC FHIR R4](https://fhir.loinc.org) — code display verification and text-to-code search
  - [NLM RxNorm REST API](https://rxnav.nlm.nih.gov) — drug name to RxNorm CUI mapping (no auth required)
  - [NLM SNOMED CT FHIR Server](https://cts.nlm.nih.gov/fhir) — SNOMED display verification
- **Disk-based JSON cache** — terminology results are cached with a configurable TTL (default: 7 days) to minimize latency and support offline use after initial population.
- **Graceful offline fallback** — hardcoded overrides cover the most common vital-sign LOINC codes (heart rate, blood pressure, SpO₂, temperature, weight, height, BMI) so the pipeline remains functional without network access.
- **Blood pressure panel synthesis** — pairs generic systolic/diastolic Observations sharing the same subject and timestamp into a single LOINC `85354-9` panel resource.
- **MedicationRequest → MedicationAdministration conversion** — converts intent-style medication requests into completed administration records matching the SI-Listener profile.
- **Resource routing** — resources are categorized into `acceptedBundle` (transaction-ready, normalized) or `rejectedBundle` (profile violations or detected uncertainty requiring manual review).
- **Provenance generation** — each accepted resource is paired with a `Provenance` resource recording the mapper version, LLM model, and terminology services used.
- **OperationOutcome diagnostics** — all normalization decisions, warnings, and errors are emitted as structured issues alongside the output bundle.

**Output structure:**

```json
{
  "acceptedBundle": { ... },   // transaction-ready normalized resources
  "rejectedBundle": { ... },   // resources requiring manual review
  "issues": [ ... ],           // per-resource diagnostic entries
  "outcome": { ... },          // FHIR OperationOutcome summary
  "meta": {
    "terminologyNetworkUsed": true,
    "mapperVersion": "2.0.0"
  }
}
```

**Normalization rules applied:**

| Resource type | Normalization applied |
|---|---|
| `Patient` | Removes placeholder `birthDate` values (e.g., "unknown") |
| `Observation` | Verifies/repairs LOINC codes via live lookup; infers code from display text and unit heuristics; normalizes UCUM units |
| `Procedure` | Validates SNOMED CT display text; warns on suspicious coding system use |
| `MedicationRequest` | Converts to `MedicationAdministration`; enriches with RxNorm CUI |

**Build dependencies:**

- [libcurl](https://curl.se/libcurl/) (`curl/curl.h`)
- [nlohmann/json](https://github.com/nlohmann/json)

---

## Build Chain (CMake)

```bash
cmake -S . -B build
cmake --build build -j
```

This produces three executables:

1. `transcribe_audio.exe` — real-time speech-to-text
2. `analyze_text.exe` — AI contextual analysis and FHIR extraction
3. `deterministic_fhir_mapper.exe` — deterministic FHIR normalization

**Additional build dependencies for `transcribe_audio.exe`:**

- [PortAudio](http://www.portaudio.com/)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (`whisper.h`)
- [Boost.Beast / Boost.Asio](https://www.boost.org/) (WebSocket support)

**Additional build dependencies for `analyze_text.exe`:**

- [openai-cpp](https://github.com/olrea/openai-cpp) (`openai.hpp`) with [nlohmann/json](https://github.com/nlohmann/json)

---

## Typical Usage

```bash
# Microphone mode — pipe transcript into analyzer
./build/transcribe_audio.exe \
    --whisper_model_path ./models/ggml-base.en.bin \
    --pipe --timestamp --adaptive_energy \
  | ./build/analyze_text.exe

# File transcription
./build/transcribe_audio.exe \
    --whisper_model_path ./models/ggml-base.en.bin \
    --input_source file --audio_file recording.mp4 \
    --pipe --timestamp

# WebSocket server (e.g., for mobile clients)
./build/transcribe_audio.exe \
    --whisper_model_path ./models/ggml-base.en.bin \
    --input_source websocket \
    --websocket_port 8080 --websocket_send_transcripts

# Run deterministic mapper standalone
./build/deterministic_fhir_mapper.exe input_bundle.json output_bundle.json \
    --model-name medgemma-1.5:4b \
    --cache-dir ./terminology_cache --cache-ttl-days 7

# Offline mapper (no network calls)
./build/deterministic_fhir_mapper.exe input_bundle.json output_bundle.json \
    --model-name medgemma-1.5:4b --no-network
```

---

📄 **License**
This project is released under the GNU Affero General Public License, version 3 or later (AGPL-3.0-or-later).
You are free to use, modify, and redistribute this software under the terms of the AGPL.
See the [LICENSE](LICENSE) file for the full text.

## Contacts
- Mail: Maurizio.Martignano@spazioit.com
- Website: https://spazioit.com/pages_en/sol_inf_en/si-listener  
- GitHub: https://github.com/mmartign/Speech-to-Knowledge

------------------------------------------------------------------------

## 🏢 About Spazio IT

Spazio - IT Soluzioni Informatiche s.a.s.\
via Manzoni 40\
46051 San Giorgio Bigarello\
Italy

https://spazioit.com

Part of the **OR-Edge Project** — AI-powered solutions for medical edge environments.

------------------------------------------------------------------------

## ⚠ Disclaimer

This software is provided **without warranty**.\
It is intended for research, validation, and controlled medical IT environments.\
It does not replace certified medical decision systems.
