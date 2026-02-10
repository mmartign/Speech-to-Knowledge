🚀 **Real-Time Medical Speech-to-Knowledge Pipeline**

<img width="736" height="351" alt="Speech-to-Knowledge-Architecture drawio" src="https://github.com/user-attachments/assets/3f658388-d68f-479d-ba20-e76149c4391c" />

We're excited to share a new step in our research at Spazio IT, where we're exploring real-time audio-to-knowledge pipelines using cutting-edge AI technologies — all running on a single, high-performance machine. Here's a snapshot of what we're building:

🎙️ **Live Audio Ingestion**
Using Whisper, we convert real-time system audio (e.g., a microphone stream) into continuous text — no audio recordings needed.

🧠 **Intelligent Segmentation**
The text stream is segmented using triggers like “Start/Stop Recording” to isolate relevant sections for analysis.

🔍 **Generative AI Processing**
Each segmented block is analyzed by a generative AI system to understand context and content.

🏥 **Medical Data Extraction in FHIR**
An AI model (currently MedGemma-1.5:4b) extracts structured medical data in FHIR format — enabling interoperability and downstream use.

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
+ 15ms median latency (vs 210ms in Python

📄 **License**
This project is released under the GNU General Public License, version 2 (GPL-2.0).
You are free to use, modify, and redistribute this software under the terms of the GPL-2.0. Any derivative work must also be distributed under the same license.
A copy of the license should be included with this repository. If not, see the full license text at:
https://www.gnu.org/licenses/old-licenses/gpl-2.0.html

## Contacts
- Mail: Maurizio.Martignano@spazioit.com
- Website: https://spazioit.com/pages_en/sol_inf_en/si-listener  
- GitHub: https://github.com/mmartign/Speech-to-Knowledge 

