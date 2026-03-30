---
trigger: always_on
---

# VideoAnalyzer — Project Specification & Developer Blueprint

> **Version:** 1.0.0  
> **Author:** S. Pratham  
> **Status:** Pre-development (Specification Phase)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Pipeline — Step by Step](#4-pipeline--step-by-step)
   - 4.1 Input Layer
   - 4.2 Audio Extraction & Transcription
   - 4.3 Frame Extraction (Scene-Change Based)
   - 4.4 Frame Analysis with Qwen3-VL
   - 4.5 Master Summary Generation
   - 4.6 Prompt Q&A Layer (Optional)
   - 4.7 Output & Cleanup
5. [Models & Dependencies](#5-models--dependencies)
6. [Hardware Detection & Optimization](#6-hardware-detection--optimization)
7. [Module API (Importable Interface)](#7-module-api-importable-interface)
8. [Logging Standard](#8-logging-standard)
9. [Configuration Reference](#9-configuration-reference)
10. [Frame Extraction Module (Reference Code)](#10-frame-extraction-module-reference-code)
11. [Development Guidelines](#11-development-guidelines)
12. [Error Handling Strategy](#12-error-handling-strategy)

---

## 1. Project Overview

**VideoAnalyzer** is a local, fully offline AI pipeline that takes a video file as input and produces an extraordinarily detailed, multi-page natural language analysis of its content — covering both the spoken audio and the visual frames.

It is designed to be:
- **Importable** as a Python module into any project
- **Device-independent**, running on CPU or GPU with automatic detection and optimization
- **Self-sufficient**, detecting and downloading required models on first run only
- **Heavily logged**, with every stage printing structured status output
- **Ephemeral by default**, with all temp files auto-deleted after a run

### What it does in plain English

You point it at a video file. Optionally, you give it a question or instruction (a "prompt"). It extracts the audio, transcribes every word spoken, extracts key visual frames based on scene changes, describes every frame using a vision-language model, feeds everything into a large language model, and returns either a dense multi-page summary — or a precise, evidence-backed answer to your question.

### Example use cases

- "What is being taught in this lecture video?"
- "List every product shown in this advertisement."
- "Summarize the key arguments made in this interview."
- "What happens at the 2-minute mark?"
- Running with no prompt → returns a comprehensive standalone summary document.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                            │
│   video_path: str  +  prompt: str | None                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴────────────────┐
            │                                │
            ▼                                ▼
┌───────────────────────┐        ┌───────────────────────────────┐
│   AUDIO PIPELINE      │        │      VIDEO PIPELINE           │
│                       │        │                               │
│  1. Extract audio     │        │  1. Scene-change frame        │
│     (ffmpeg)          │        │     extraction (OpenCV +      │
│                       │        │     SSIM + Histogram)         │
│  2. Transcribe with   │        │                               │
│     distil-large-v3   │        │  2. Frame analysis with       │
│     (faster-whisper)  │        │     qwen3-vl:2b (Ollama)      │
│                       │        │     → one description         │
│  → transcription.txt  │        │       per keyframe            │
│     (temp)            │        │                               │
└──────────┬────────────┘        │  → frame_analyses.txt (temp)  │
           │                     └──────────────┬────────────────┘
           └──────────────┬──────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │   SYNTHESIS LAYER           │
            │                             │
            │   LLM: qwen3.5:9b (Ollama)  │
            │                             │
            │   Input:                    │
            │   - Full transcription      │
            │   - All frame descriptions  │
            │                             │
            │   Output:                   │
            │   → Master Summary (2-3+    │
            │     pages, ultra-detailed)  │
            └──────────────┬──────────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
        prompt=None               prompt=str
               │                       │
               ▼                       ▼
      ┌─────────────────┐   ┌──────────────────────────┐
      │  Return summary │   │  Q&A LAYER               │
      │  to caller      │   │                          │
      └─────────────────┘   │  LLM: qwen3.5:9b         │
                            │  Input: summary + prompt  │
                            │                          │
                            │  Output:                 │
                            │  - Direct answer         │
                            │  - Relevant summary      │
                            │    sections cited        │
                            └──────────────────────────┘
```

---

## 3. Directory Structure

```
VideoAnalyzer/
│
├── video_analyzer/                  ← Main package (importable)
│   ├── __init__.py                  ← Public API: analyze()
│   ├── analyzer.py                  ← Top-level pipeline orchestrator
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── extractor.py             ← ffmpeg audio extraction
│   │   └── transcriber.py           ← distil-large-v3 transcription
│   ├── video/
│   │   ├── __init__.py
│   │   ├── frame_extractor.py       ← Scene-change keyframe extraction
│   │   └── frame_analyzer.py        ← qwen3-vl:2b frame description
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_manager.py        ← Ollama process + model management
│   │   └── summarizer.py            ← Summary + Q&A prompting logic
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── hardware.py              ← GPU/CPU detection & config
│   │   ├── logger.py                ← Centralized logging setup
│   │   ├── model_manager.py         ← Model download/verification
│   │   └── cleanup.py               ← Temp file lifecycle management
│   └── config.py                    ← All tunable defaults
│
├── ingestion/                       ← Drop your video files here
│   └── video.mp4                    ← (example)
│
├── artifacts/                       ← Runtime artifacts
│   └── video_frames/                ← Keyframes saved during run
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 4. Pipeline — Step by Step

### 4.1 Input Layer

**File:** `video_analyzer/analyzer.py` → `analyze()`

The entry point accepts:

```python
analyze(
    video_path: str,          # Absolute or relative path to video file
    prompt: str | None = None # Optional user question / instruction
) -> AnalysisResult
```

**On entry, the pipeline must:**

1. Log the start timestamp, video path, and detected file size.
2. Validate that the video file exists and is readable. Raise `FileNotFoundError` with a clear message if not.
3. Detect hardware (see §6) and log the result.
4. Verify all required models and tools are available (see §5). Download anything missing.
5. Ensure Ollama is running; start it if not (see §5.3).
6. Create a unique temporary working directory for this run (e.g., `/tmp/videoanalyzer_{uuid4}/`).

---

### 4.2 Audio Extraction & Transcription

**Files:** `audio/extractor.py`, `audio/transcriber.py`

#### Step 1 — Extract audio with ffmpeg

Use `ffmpeg` (via `subprocess`) to strip the audio track from the video and save it as a `.wav` file into the temp directory.

```
LOG: [AUDIO] Extracting audio from video...
LOG: [AUDIO] Source: /path/to/video.mp4 (duration: Xs, size: XMB)
LOG: [AUDIO] Writing audio → /tmp/videoanalyzer_<uuid>/audio.wav
LOG: [AUDIO] ✓ Audio extracted in X.Xs
```

If the video has no audio track, log a clear warning and set transcription to an empty string — **do not crash**.

#### Step 2 — Transcribe with `distil-whisper/distil-large-v3`

Use the `faster-whisper` library with the `distil-large-v3` model.

- On **GPU**: use `compute_type="float16"` and `device="cuda"`
- On **CPU**: use `compute_type="int8"` and `device="cpu"`
- Model files are cached locally after first download. Check the cache before loading.

```
LOG: [TRANSCRIBE] Loading distil-large-v3 (faster-whisper)...
LOG: [TRANSCRIBE] Device: CPU | Compute: int8
LOG: [TRANSCRIBE] Starting transcription...
LOG: [TRANSCRIBE] Progress: segment 1/N — "Hello and welcome to..."
LOG: [TRANSCRIBE] Progress: segment 2/N — "Today we're going to..."
...
LOG: [TRANSCRIBE] ✓ Transcription complete — X segments, ~X words, X.Xs elapsed
LOG: [TRANSCRIBE] Writing → /tmp/videoanalyzer_<uuid>/transcription.txt
```

The transcription file must include segment timestamps:

```
[00:00:00 → 00:00:05] Hello and welcome to this tutorial.
[00:00:05 → 00:00:12] Today we're going to cover the basics of Python.
```

---

### 4.3 Frame Extraction (Scene-Change Based)

**File:** `video/frame_extractor.py`

This module is directly based on the existing scene-change extraction script (see §10). No new logic needs to be invented here — it is a direct integration of that script into the pipeline with the following additions:

#### GPU Batch Mode vs CPU Sequential Mode

- **CPU mode**: Extract and save frames one at a time (existing behavior).
- **GPU mode**: Batch-decode frames using OpenCV's CUDA-accelerated backend where available, then process the batch for scene-change comparison. This accelerates the *extraction* step only — the actual scene-change comparison math (histogram + SSIM) is the same on both.

```
LOG: [FRAMES] Starting keyframe extraction...
LOG: [FRAMES] Mode: GPU-BATCH | Video: 1280x720 @ 30fps
LOG: [FRAMES] Thresholds → hist=0.28, ssim=0.89, min_interval=8 frames
LOG: [FRAMES] Processing frame 0 → saved keyframe_0000 (first frame)
LOG: [FRAMES] Frame 142 → SCENE CHANGE [hist=0.341] → saved keyframe_0001
LOG: [FRAMES] Frame 287 → SCENE CHANGE [ssim=0.812] → saved keyframe_0002
...
LOG: [FRAMES] ✓ Extraction complete — X keyframes from X total frames (X.Xs)
LOG: [FRAMES] Frames saved → /tmp/videoanalyzer_<uuid>/frames/
```

Frames are saved into the temp directory for this run, not `artifacts/video_frames/`, since they will be cleaned up after analysis.

---

### 4.4 Frame Analysis with Qwen3-VL:2b

**File:** `video/frame_analyzer.py`

Each extracted keyframe is sent to the `qwen3-vl:2b` model running via Ollama's local API (`http://localhost:11434/api/generate`).

#### Analysis prompt per frame

Send each frame as a base64-encoded image with the following system instruction:

```
You are a precise visual analyst. Describe this video frame in exhaustive detail.
Include: all visible objects, text, people (appearance, expressions, actions),
environment, colors, spatial layout, any on-screen graphics or UI elements,
and what is happening in the scene. Be thorough — your description will be
used to reconstruct a full understanding of this video.
```

#### Processing order

- Frames are analyzed **sequentially** (one at a time) regardless of GPU/CPU. Vision model inference is a single-model bottleneck — batching does not apply here.
- Each frame description is immediately appended to `frame_analyses.txt` in the temp directory as it completes, so progress is preserved incrementally.

#### Output format per frame

```
=== KEYFRAME 0001 | Frame #142 | Timestamp ~00:00:04 ===
[Full LLM description here]

=== KEYFRAME 0002 | Frame #287 | Timestamp ~00:00:09 ===
[Full LLM descript