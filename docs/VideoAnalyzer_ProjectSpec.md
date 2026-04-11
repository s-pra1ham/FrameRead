# VideoAnalyzer — Project Specification & Developer Blueprint

> **Version:** 2.0.0  
> **Author:** S. Pratham  
> **Status:** MVP (Implementation Complete)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Pipeline — Step by Step](#4-pipeline--step-by-step)
   - 4.1 Input Layer
   - 4.2 Audio Extraction & Transcription
   - 4.3 Frame Extraction (Scene-Change Based)
   - 4.4 Frame Analysis with Qwen2-VL (Local HuggingFace Inference)
   - 4.5 Master Summary Generation
   - 4.6 Prompt Q&A Layer (Optional)
   - 4.7 Output & Cleanup
5. [Models & Dependencies](#5-models--dependencies)
6. [Hardware Detection & Optimization](#6-hardware-detection--optimization)
7. [Module API (Importable Interface)](#7-module-api-importable-interface)
8. [CLI Entry Point](#8-cli-entry-point)
9. [Logging Standard](#9-logging-standard)
10. [Configuration Reference](#10-configuration-reference)
11. [Frame Extraction Module (Reference Code)](#11-frame-extraction-module-reference-code)
12. [Development Guidelines](#12-development-guidelines)
13. [Error Handling Strategy](#13-error-handling-strategy)

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

You point it at a video file. Optionally, you give it a question or instruction (a "prompt"). It extracts the audio, transcribes every word spoken, extracts key visual frames based on scene changes, describes every frame using a **local vision-language model** (Qwen2-VL via HuggingFace Transformers), feeds everything into a large language model (via Ollama), and returns either a dense multi-page summary — or a precise, evidence-backed answer to your question.

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
│     (faster-whisper)  │        │     Qwen2-VL-2B-Instruct      │
│                       │        │     (Local HuggingFace)       │
│  → transcription.txt  │        │     → Dynamic VRAM-probed     │
│     (temp)            │        │       batching on GPU         │
└──────────┬────────────┘        │                               │
           │                     │  → frame_analyses.txt (temp)  │
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

**Key architectural note:** The vision model (Qwen2-VL) runs as a **local HuggingFace Transformers model**, loaded directly into GPU/CPU memory via `transformers.Qwen2VLForConditionalGeneration`. It does **not** use Ollama. Only the summary/Q&A LLM (`qwen3.5:9b`) uses Ollama. After vision inference completes, the model is explicitly unloaded and CUDA cache is cleared so that Ollama can use the freed VRAM for the synthesis step.

---

## 3. Directory Structure

```
FrameRead/
│
├── src/                             ← Main package (importable)
│   ├── __init__.py                  ← Public API: analyze()
│   ├── analyzer.py                  ← Top-level pipeline orchestrator
│   ├── config.py                    ← All tunable defaults
│   ├── audio/
│   │   ├── extractor.py             ← ffmpeg audio extraction
│   │   └── transcriber.py           ← distil-large-v3 transcription
│   ├── video/
│   │   ├── frame_extractor.py       ← Scene-change keyframe extraction
│   │   └── frame_analyzer.py        ← Qwen2-VL frame description (local HF)
│   ├── llm/
│   │   ├── ollama_manager.py        ← Ollama process + model management
│   │   └── summarizer.py            ← Summary + Q&A prompting logic
│   └── utils/
│       ├── hardware.py              ← GPU/CPU detection & config
│       ├── logger.py                ← Centralized logging setup
│       ├── model_manager.py         ← Whisper model download/verification
│       └── cleanup.py               ← Temp file lifecycle management
│
├── docs/
│   └── VideoAnalyzer_ProjectSpec.md ← This file
│
├── run.py                           ← CLI entry point (argparse)
├── requirements.txt
├── setup.py
└── video.mp4                        ← (example input)
```

---

## 4. Pipeline — Step by Step

### 4.1 Input Layer

**File:** `src/analyzer.py` → `analyze()`

The entry point accepts:

```python
analyze(
    video_path: str,          # Absolute or relative path to video file
    prompt: str | None = None # Optional user question / instruction
) -> AnalysisResult
```

**On entry, the pipeline must:**

1. Reset the pipeline timer and log the start timestamp and video path.
2. Resolve the video path to an absolute path. Validate that the file exists and is readable. Raise `FileNotFoundError` with a clear message if not.
3. Detect hardware (see §6) and log the result.
4. Verify the Whisper model is available in cache (see §5.2). Download if missing.
5. Ensure Ollama is running; start it if not (see §5.3).
6. Ensure all required Ollama models are pulled (see §5.3).
7. Create a unique temporary working directory for this run via the `TempDirManager` context manager.

---

### 4.2 Audio Extraction & Transcription

**Files:** `audio/extractor.py`, `audio/transcriber.py`

#### Step 1 — Extract audio with ffmpeg

Use `ffmpeg` (via `subprocess`) to strip the audio track from the video and save it as a `.wav` file (16kHz, mono — optimal for Whisper) into the temp directory.

```
LOG: [AUDIO] Extracting audio from video...
LOG: [AUDIO] Source: /path/to/video.mp4 (duration: Xs, size: XMB)
LOG: [AUDIO] Writing audio → /tmp/videoanalyzer_<uuid>/audio.wav
LOG: [AUDIO] ✓ Audio extracted in X.Xs
```

If the video has no audio track, log a clear warning and set transcription to `"[Video contains no audio track]"` — **do not crash**.

The extractor also validates that the `ffmpeg` binary is available on startup and raises `RuntimeError` with install instructions if missing.

#### Step 2 — Transcribe with `distil-whisper/distil-large-v3`

Use the `faster-whisper` library with the `distil-large-v3` model.

- On **GPU**: use `compute_type="float16"` and `device="cuda"`
- On **CPU**: use `compute_type="int8"` and `device="cpu"`
- Model files are cached locally after first download. Check the cache before loading.

```
LOG: [TRANSCRIBE] Loading distil-large-v3 (faster-whisper)...
LOG: [TRANSCRIBE] Device: CPU | Compute: int8
LOG: [TRANSCRIBE] Starting transcription...
LOG: [TRANSCRIBE] Progress: segment 1 — "Hello and welcome to..."
LOG: [TRANSCRIBE] Progress: segment 6 — "Today we're going to..."
...
LOG: [TRANSCRIBE] ✓ Transcription complete — X segments, ~X words, X.Xs elapsed
LOG: [TRANSCRIBE] Writing → /tmp/videoanalyzer_<uuid>/transcription.txt
```

Progress is logged every 5 segments (not every segment) to avoid terminal spam.

The transcription file uses the format:

```
[00:00:00 → 00:00:05] Hello and welcome to this tutorial.
[00:00:05 → 00:00:12] Today we're going to cover the basics of Python.
```

---

### 4.3 Frame Extraction (Scene-Change Based)

**File:** `video/frame_extractor.py`

This module is based on the scene-change extraction algorithm using a dual-metric approach:
1. **Color histogram difference** (χ² comparison) — detects broad color shifts.
2. **SSIM structural similarity** — detects structural/layout changes.

A frame is saved as a keyframe when either threshold is breached and the minimum interval requirement is met.

#### Extraction modes

- **CPU mode** (`CPU-SEQ`): Extract and process frames one at a time.
- **GPU mode** (`GPU-BATCH`): Label only — the frame extraction and scene-change detection logic itself is CPU-bound (OpenCV reads + histogram/SSIM math). The mode label reflects the hardware context; the actual decoding path is the same.

#### Output data structure

The extractor returns a list of `ExtractedFrame` dataclass instances:

```python
@dataclass
class ExtractedFrame:
    path: str           # Absolute path to saved keyframe JPEG
    timestamp_str: str  # Formatted as "HH:MM:SS"
    frame_idx: int      # Original frame index in the video
```

#### Logs

```
LOG: [FRAMES] Starting keyframe extraction...
LOG: [FRAMES] Mode: GPU-BATCH | Video: 1280x720 @ 30.0fps
LOG: [FRAMES] Thresholds → hist=0.28, ssim=0.89, min_interval=8 frames
LOG: [FRAMES] Processing frame 0 → saved keyframe_0000 (first frame)
LOG: [FRAMES] Frame 142 → SCENE CHANGE [hist=0.341] → saved keyframe_0001
LOG: [FRAMES] Frame 287 → SCENE CHANGE [ssim=0.812] → saved keyframe_0002
...
LOG: [FRAMES] ✓ Extraction complete — X keyframes from X total frames (X.Xs)
LOG: [FRAMES] Frames saved → /tmp/videoanalyzer_<uuid>/frames/
```

Frames are saved into the temp directory for this run and cleaned up after analysis.

---

### 4.4 Frame Analysis with Qwen2-VL (Local HuggingFace Inference)

**File:** `video/frame_analyzer.py`

Each extracted keyframe is analyzed by the `Qwen/Qwen2-VL-2B-Instruct` vision-language model running **locally via HuggingFace Transformers** — not via Ollama. The model is loaded once into GPU/CPU memory, inference runs in batches, and the model is aggressively unloaded upon completion to free VRAM for the subsequent Ollama synthesis step.

#### Vision model loading

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    VISION_MODEL,
    torch_dtype=hardware.torch_dtype,   # bfloat16 / float16 / None
    device_map=hardware.device          # "cuda" or "cpu"
)
processor = AutoProcessor.from_pretrained(VISION_MODEL)
```

#### Analysis prompt per frame

Each frame is provided as a PIL Image object with the following system instruction:

```
You are a precise visual analyst. Describe this video frame in exhaustive detail.
Include: all visible objects, text, people (appearance, expressions, actions),
environment, colors, spatial layout, any on-screen graphics or UI elements,
and what is happening in the scene. Be thorough — your description will be
used to reconstruct a full understanding of this video.
```

#### Dynamic VRAM-Probed Batching (GPU Mode)

On CUDA devices, the batch size is **not hardcoded** — it is determined at runtime through a 3-phase VRAM probe protocol:

1. **Baseline Measurement**: After loading the model, record the VRAM consumed by model weights alone.
2. **Single-Frame Probe**: Run inference on the first keyframe and measure the peak VRAM delta. This captures the actual per-frame memory cost (activations, KV cache, attention maps, etc.).
3. **Batch Size Calculation**: Compute the optimal batch size using:
   ```
   safety_buffer = total_vram × 0.225   (22.5% of total — midpoint of 20–25% target range)
   usable_vram   = total_vram − baseline − safety_buffer
   batch_size     = max(1, floor(usable_vram / per_frame_cost))
   ```

The first frame's probe result is **not discarded** — its description is written immediately, and processing continues from frame index 1.

On CPU, the batch size is fixed at 1.

#### Batch inference

Frames are processed in batches of `batch_size`:
- Multiple images are loaded as PIL `RGB` objects.
- Chat templates are applied per-image using the processor.
- Inference runs with `torch.no_grad()` and `max_new_tokens=256`.
- Results are decoded via `processor.batch_decode(...)`.

If a frame cannot be read (corrupt file), it is logged as a warning and skipped.  
If a batch inference fails, the error is caught and a placeholder `[Vision inference failed: ...]` is written for each frame in that batch.

#### Output format per frame

```
=== KEYFRAME 0001 | Frame #142 | Timestamp ~00:00:04 ===
[Full LLM description here]

=== KEYFRAME 0002 | Frame #287 | Timestamp ~00:00:09 ===
[Full LLM description here]
```

Descriptions are written to `frame_analyses.txt` **incrementally** as each batch completes (via `f.flush()`), preserving partial progress.

#### Critical VRAM Cleanup

After all frames are analyzed (or if an error occurs), the model and processor are **explicitly deleted** and `torch.cuda.empty_cache()` is called. This is wrapped in a `try/finally` block to guarantee cleanup even on failure. This step is essential — the Ollama synthesis model cannot load if the vision model is still holding VRAM.

```python
finally:
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

#### Logs

```
LOG: [VISION] Analyzing X keyframes with Qwen/Qwen2-VL-2B-Instruct...
LOG: [VISION] Loading local vision model into memory. This may take a moment...
LOG: [VISION] Model loaded successfully.
LOG: [VISION] -- VRAM Probe ------------------------------------------
LOG: [VISION] Total VRAM:       8.00 GB
LOG: [VISION] Baseline (model): 3.81 GB
LOG: [VISION] Probing VRAM cost with 1 frame...
LOG: [VISION] Peak after probe: 5.20 GB
LOG: [VISION] Per-frame cost:   1.39 GB
LOG: [VISION] Safety buffer:    1.80 GB (22.5% of total)
LOG: [VISION] Usable VRAM:      2.39 GB
LOG: [VISION] Optimal batch:    1 frames
LOG: [VISION] ----------------------------------------------------
LOG: [VISION] Vision Mode: Local Inference (Dynamic Batch size: 1)
LOG: [VISION] Frame 0001 described (~X.Xs per frame in batch)
LOG: [VISION] Processing frames 2 to 3 of X...
LOG: [VISION] Frame 0002 described (~X.Xs per frame in batch)
...
LOG: [VISION] Unloading model and freeing CUDA cache safely...
LOG: [VISION] All frames analyzed -- X.Xs total (Dynamic Batch Size: X)
LOG: [VISION] Writing -> /tmp/videoanalyzer_<uuid>/frame_analyses.txt
```

---

### 4.5 Master Summary Generation

**File:** `llm/summarizer.py`

This is the most important step. The LLM (`qwen3.5:9b` via Ollama's `/api/chat` endpoint) receives the full transcription and all frame descriptions and is instructed to produce an extraordinarily detailed summary.

#### System prompt

```
You are a senior multimedia analyst producing a comprehensive written record of a video.
You have been provided:
  1. A full timestamped transcription of all spoken audio in the video.
  2. Detailed visual descriptions of every significant scene/frame in the video.

Your task: Synthesize both sources into a single, exhaustive, multi-page written analysis.

Structure your summary as follows:
  - OVERVIEW: What is this video? What is its purpose, genre, and main subject?
  - DETAILED NARRATIVE: Walk through the video chronologically, integrating what was
    said and what was shown at each point. Do not summarize loosely — describe everything
    in precise, specific language. This section should be very long.
  - KEY POINTS & CONCEPTS: A thorough list of every important idea, instruction, claim,
    or demonstration present in the video.
  - VISUAL HIGHLIGHTS: Notable visual elements, UI shown, graphics, on-screen text,
    demonstrations.
  - SPEAKERS & PARTICIPANTS: Who appears, what they say, their apparent role.
  - TONE & STYLE: Pacing, presentation style, intended audience.

Do not truncate. Do not summarize loosely. A 30-second video should produce at minimum
600–800 words of analysis. A 5-minute video should produce 3,000–5,000 words.
Be precise, be thorough, and leave nothing out.
```

#### User message format

```
TRANSCRIPTION:
{transcription}

FRAME DESCRIPTIONS:
{frame_analyses}
```

#### Ollama request configuration

- Endpoint: `POST {OLLAMA_HOST}/api/chat`
- `stream: false` (wait for complete response)
- `num_ctx: 32768` (explicitly set to ensure large context window is utilized)
- Timeout: 600 seconds (10 minutes)

#### Logs

```
LOG: [SUMMARY] Sending transcription + X frame descriptions to qwen3.5:9b...
LOG: [SUMMARY] Input size: ~X tokens
LOG: [SUMMARY] Generating master summary (this may take a moment)...
LOG: [SUMMARY] ✓ Summary generated — X words, X tokens (X.Xs)
```

---

### 4.6 Prompt Q&A Layer (Optional)

**File:** `llm/summarizer.py` → `answer_with_summary()`

This step only runs if the user provided a `prompt` argument. It takes the master summary produced in §4.5 and the user's prompt, and asks the LLM to answer the prompt using the summary as the sole knowledge source.

#### System prompt

```
You are an expert analyst. You have access to a detailed summary of a video (provided below).
A user has asked a specific question about this video.

Your task:
  1. Answer the user's question precisely and completely.
  2. After your answer, cite the relevant sections of the summary that support your answer,
     quoting or paraphrasing key lines to show your reasoning.

Do not go beyond the summary. If the answer cannot be found in the summary, say so clearly.
```

#### User message format

```
SUMMARY:
{master_summary}

USER QUESTION:
{prompt}
```

#### Ollama request configuration

- `num_ctx: 16384` (Q&A context window is smaller than synthesis)
- Timeout: 300 seconds

#### Logs

```
LOG: [QA] User prompt detected: "{prompt[:60]}..."
LOG: [QA] Sending summary + prompt to qwen3.5:9b...
LOG: [QA] ✓ Prompt answer generated — X words (X.Xs)
```

If Q&A inference fails, a warning is logged and a placeholder `[Failed to answer prompt: ...]` string is returned — it does **not** crash the pipeline.

---

### 4.7 Output & Cleanup

**File:** `utils/cleanup.py`

#### Output

The `analyze()` function returns an `AnalysisResult` dataclass:

```python
@dataclass
class AnalysisResult:
    summary: str                    # Always present — full master summary
    prompt_answer: str | None       # Present only if prompt was provided
    keyframe_count: int             # Number of keyframes analyzed
    transcription: str              # Full timestamped transcription
    duration_seconds: float         # Total pipeline wall-clock time
    video_path: str                 # Echo of input path (absolute)
```

If `PRINT_SUMMARY` is enabled (default: `True`), the summary is printed to terminal in a clean, bordered format. If a Q&A prompt was provided, the question and answer are also printed.

#### Cleanup

The `TempDirManager` context manager handles cleanup automatically. When the pipeline's `with` block exits (success or failure), the temp directory and all its contents — audio, frames, transcription, frame analysis files — are deleted.

```
LOG: [CLEANUP] Removing temp directory: /tmp/videoanalyzer_<uuid>/
LOG: [CLEANUP] ✓ Cleaned up X files
```

If cleanup fails, a warning is logged — **it does not raise**, since the result is already computed and returned.

```
LOG: [DONE] ✨ Total pipeline time: X.Xs
```

---

## 5. Models & Dependencies

### 5.1 Python dependencies (`requirements.txt`)

```
faster-whisper>=1.0.0       # Whisper transcription (distil-large-v3)
opencv-python>=4.8.0        # Frame extraction
scikit-image>=0.21.0        # SSIM computation
numpy>=1.24.0               # Array ops
requests>=2.31.0            # Ollama API calls
Pillow>=10.0.0              # Image handling for vision model
torch>=2.0.0                # GPU detection, tensor ops, model inference
ffmpeg-python>=0.2.0        # Audio extraction wrapper
transformers>=4.45.0        # Qwen2-VL model loading & inference
accelerate>=0.26.0          # HuggingFace model device mapping
qwen-vl-utils>=0.0.1        # Qwen-VL chat template utilities
```

> **Note:** `ffmpeg` binary must also be installed on the system. The pipeline detects its presence on startup and raises a `RuntimeError` with install instructions if missing.

### 5.2 Whisper model management

- Model: `distil-whisper/distil-large-v3`
- Managed by `faster-whisper` — cached to `~/.cache/huggingface/` automatically after first download.
- On startup, `model_manager.py` performs a lightweight dummy load to check/trigger the download:

```python
WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", download_root=None)
```

```
LOG: [MODEL] Checking local cache for distil-large-v3...
LOG: [MODEL] ✓ distil-large-v3 is ready in cache.
```

### 5.3 Vision model management (HuggingFace)

- Model: `Qwen/Qwen2-VL-2B-Instruct`
- Loaded at runtime via `Qwen2VLForConditionalGeneration.from_pretrained()` and `AutoProcessor.from_pretrained()`.
- Model weights are cached to `~/.cache/huggingface/hub/` automatically by the Transformers library after first download.
- **No separate startup check** is performed for the vision model — it is loaded directly at the point of use in `frame_analyzer.py`. If it fails to load, a `RuntimeError` is raised with a clear message.

### 5.4 Ollama model management & process control

- Models needed via Ollama: **`qwen3.5:9b`** (language synthesis only)
- Managed by `llm/ollama_manager.py`

> **Note:** The vision model (`Qwen2-VL`) does **not** use Ollama. Only the summary/Q&A model uses Ollama.

#### Ollama process lifecycle

On every run, the pipeline must:

1. Check if the `ollama` binary exists on PATH. If not → raise `RuntimeError` with installation URL.
2. Check if Ollama is already running by hitting `GET http://localhost:11434/api/tags`.
3. If not running → call `subprocess.Popen(["ollama", "serve"])`, poll for up to 10 seconds for it to respond, then proceed.

```
LOG: [OLLAMA] Checking if Ollama is running...
LOG: [OLLAMA] Not running — starting ollama serve...
LOG: [OLLAMA] ✓ Ollama is live at http://localhost:11434
```

#### Model availability check

After confirming Ollama is running, call `GET /api/tags` and check the returned model list.

For the required model (`qwen3.5:9b`):
- If present → log `✓ Model qwen3.5:9b found.`
- If absent → trigger pull via `POST /api/pull` with streaming progress, retrying once via subprocess if the REST pull fails.

```
LOG: [OLLAMA] Required models: qwen3.5:9b
LOG: [OLLAMA] ✓ qwen3.5:9b — found
```

Or if pulling:

```
LOG: [OLLAMA] ✗ qwen3.5:9b — not found
LOG: [OLLAMA] Pulling qwen3.5:9b (one-time download)...
LOG: [OLLAMA] Pull progress (qwen3.5:9b): 20%
LOG: [OLLAMA] Pull progress (qwen3.5:9b): 40%
...
LOG: [OLLAMA] ✓ qwen3.5:9b ready.
```

---

## 6. Hardware Detection & Optimization

**File:** `utils/hardware.py`

On startup, the pipeline runs a full hardware survey and logs the result. This determines behavior throughout the pipeline.

### HardwareConfig dataclass

```python
@dataclass
class HardwareConfig:
    device: str                    # "cuda" or "cpu"
    gpu_name: str | None           # e.g. "NVIDIA GeForce RTX 4060"
    vram_gb: float                 # Total VRAM in GB (0.0 for CPU)
    whisper_dtype: str             # "float16" (GPU) or "int8" (CPU)
    torch_dtype: torch.dtype | None  # torch.bfloat16, torch.float16, or None (CPU)
```

### Detection logic

```python
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    if vram_gb >= 20.0:
        torch_dtype = torch.bfloat16    # High-end (A100, H100, RTX 4090)
    else:
        torch_dtype = torch.float16     # Mid/low-end
    whisper_dtype = "float16"
else:
    device = "cpu"
    torch_dtype = None
    whisper_dtype = "int8"
```

### Log output

```
LOG: [HARDWARE] ── Hardware Survey ──────────────────────────
LOG: [HARDWARE] Device:       CUDA (GPU)
LOG: [HARDWARE] GPU:          NVIDIA GeForce RTX 4060
LOG: [HARDWARE] VRAM:         8.0 GB
LOG: [HARDWARE] Torch dtype:  float16
LOG: [HARDWARE] ─────────────────────────────────────────────
```

Or for CPU:

```
LOG: [HARDWARE] Device:       CPU
LOG: [HARDWARE] Whisper dtype: int8
```

### GPU tier optimization table

| GPU Class | Examples | `torch_dtype` | Whisper compute | Vision batch strategy |
|---|---|---|---|---|
| High-end (≥20GB VRAM) | A100, H100, RTX 4090 | `bfloat16` | float16 | Dynamic VRAM-probed (likely multi-frame) |
| Mid/Low-end (<20GB VRAM) | RTX 3060–4070 | `float16` | float16 | Dynamic VRAM-probed (likely 1–2 frames) |
| CPU (no GPU) | Any | `None` | int8 | Fixed batch_size=1 |

The `HardwareConfig` object is passed to every pipeline module that needs it. No module does its own hardware detection — they all receive it from the shared config.

---

## 7. Module API (Importable Interface)

**File:** `src/__init__.py`

The entire pipeline is exposed as a single importable function:

```python
from src import analyze

result = analyze(
    video_path="video.mp4",
    prompt="What products are demonstrated in this video?"
)

print(result.summary)          # Full multi-page summary
print(result.prompt_answer)    # Answer to the question
print(result.transcription)    # Full timestamped transcript
print(result.keyframe_count)   # e.g. 14
print(result.duration_seconds) # e.g. 47.3
```

### `AnalysisResult` dataclass

```python
@dataclass
class AnalysisResult:
    summary: str
    prompt_answer: str | None
    keyframe_count: int
    transcription: str
    duration_seconds: float
    video_path: str
```

### Terminal output behavior

Even when used as a module (not run directly), the pipeline always prints structured log output to `stdout`. This is intentional — logs are a first-class feature of this system, not optional debug output. If a caller wants to suppress them, they can redirect `stdout`.

---

## 8. CLI Entry Point

**File:** `run.py`

A standalone CLI wrapper using `argparse`:

```
python run.py <video_path> [-p|--prompt "your question"]
```

### Usage examples

```bash
# Generate a full summary
python run.py video.mp4

# Ask a specific question
python run.py lecture.mp4 --prompt "What tools were mentioned?"
```

### Error handling

- If the video file doesn't exist, exits with a clear error and code 1.
- `KeyboardInterrupt` prints `[USER ABORT]` and exits cleanly.
- Any unhandled exception prints `[FATAL ERROR]` with the message and exits with code 1.

---

## 9. Logging Standard

**File:** `utils/logger.py`

Every single log line follows this format:

```
[TIMESTAMP] [MODULE     ] MESSAGE
```

Example:

```
[00:00:01.234] [AUDIO]      Extracting audio from video...
[00:00:03.891] [AUDIO]      ✓ Audio extracted in 2.7s
[00:00:03.892] [TRANSCRIBE] Loading distil-large-v3...
[00:00:07.441] [TRANSCRIBE] Device: CPU | Compute: int8
[00:00:07.442] [TRANSCRIBE] Starting transcription...
[00:00:08.011] [TRANSCRIBE] Segment 1 — "Hello and welcome..."
[00:00:14.220] [TRANSCRIBE] ✓ Transcription complete — 12 segments, 143 words, 6.8s
[00:00:14.221] [FRAMES]     Starting keyframe extraction...
[00:00:14.880] [FRAMES]     Frame 0 → saved (first frame)
[00:00:16.100] [FRAMES]     Frame 142 → SCENE CHANGE [hist=0.341] → saved
...
```

### Logging rules

1. Every function entry of significance gets an opening log line.
2. Every function completion gets a `✓` success log with timing.
3. All errors log the full exception message before raising.
4. Use a single centralized logger — not `print()` scattered everywhere.
5. Module tags are fixed-width (padded to 12 chars) for visual alignment.
6. Timestamps are relative to pipeline start, not wall clock. The timer is reset via `reset_timer()` at the beginning of each pipeline run.
7. `sys.stdout.flush()` is called after every log line for immediate visibility.

---

## 10. Configuration Reference

**File:** `src/config.py`

All tunable parameters live here. No magic numbers anywhere else in the codebase.

```python
# ── Frame Extraction ──────────────────────────────────
HIST_THRESHOLD       = 0.28    # χ² histogram distance. Lower = more sensitive.
SSIM_THRESHOLD       = 0.89    # Structural similarity. Lower = more keyframes.
MIN_FRAME_INTERVAL   = 8       # Minimum frames between two saved keyframes.

# ── Models ────────────────────────────────────────────
WHISPER_MODEL        = "distil-large-v3"
VISION_MODEL         = "Qwen/Qwen2-VL-2B-Instruct"   # HuggingFace model ID (local inference)
SUMMARY_MODEL        = "qwen3.5:9b"                   # Ollama model (synthesis + Q&A)

# ── Ollama ────────────────────────────────────────────
OLLAMA_HOST          = "http://localhost:11434"
OLLAMA_STARTUP_WAIT  = 10      # Seconds to wait after starting ollama serve

# ── Output ────────────────────────────────────────────
PRINT_SUMMARY        = True    # Print summary to terminal on completion
TEMP_DIR_PREFIX      = "/tmp/videoanalyzer_"

# ── Vision prompt ─────────────────────────────────────
FRAME_ANALYSIS_PROMPT = """
You are a precise visual analyst. Describe this video frame in exhaustive detail.
Include: all visible objects, text, people (appearance, expressions, actions),
environment, colors, spatial layout, any on-screen graphics or UI elements,
and what is happening in the scene. Be thorough — your description will be
used to reconstruct a full understanding of this video.
"""

# ── Summary prompt ────────────────────────────────────
SUMMARY_SYSTEM_PROMPT = """
[...full prompt as specified in §4.5...]
"""

# ── Q&A prompt ────────────────────────────────────────
QA_SYSTEM_PROMPT = """
[...full prompt as specified in §4.6...]
"""
```

### Additional runtime constants (in `frame_analyzer.py`)

```python
# Fraction of total VRAM to keep free as a safety margin to avoid OOM.
# Midpoint of the 20-25% target range.
VRAM_SAFETY_BUFFER_FRACTION = 0.225
```

---

## 11. Frame Extraction Module (Reference Code)

This is the core scene-change extraction algorithm integrated into `video/frame_extractor.py`. The key logic is preserved from the original reference script:

```python
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def extract_frames(
    video_path,
    output_folder,
    hist_threshold=0.30,
    ssim_threshold=0.88,
    min_frame_interval=8,
    use_grayscale_for_ssim=True,
    hist_method=cv2.HISTCMP_CHISQR
):
    """
    Extract keyframes on structural / scene changes.
    Two checks:
      1. Color histogram difference
      2. SSIM structural similarity
    Triggers save when either difference is large enough.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, prev_frame = cap.read()
    if not ret:
        return

    prev_hist = []
    for channel in cv2.split(prev_frame):
        h = cv2.calcHist([channel], [0], None, [256], [0, 256])
        h = cv2.normalize(h, h).flatten()
        prev_hist.append(h)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    saved_count = 0
    cv2.imwrite(
        os.path.join(output_folder, f"keyframe_{saved_count:04d}_frame_{frame_idx:06d}.jpg"),
        prev_frame
    )
    saved_count += 1
    last_save_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx - last_save_idx < min_frame_interval:
            continue

        hist_diff_max = 0
        for i, channel in enumerate(cv2.split(frame)):
            h = cv2.calcHist([channel], [0], None, [256], [0, 256])
            h = cv2.normalize(h, h).flatten()
            diff = cv2.compareHist(prev_hist[i], h, hist_method)
            hist_diff_max = max(hist_diff_max, diff)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(
            prev_gray, curr_gray,
            data_range=curr_gray.max() - curr_gray.min()
        )

        should_save = False
        reason = ""
        if hist_diff_max > hist_threshold:
            should_save = True
            reason += f"hist={hist_diff_max:.3f} "
        if ssim_value < ssim_threshold:
            should_save = True
            reason += f"ssim={ssim_value:.3f}"

        if should_save:
            filename = os.path.join(
                output_folder,
                f"keyframe_{saved_count:04d}_frame_{frame_idx:06d}.jpg"
            )
            cv2.imwrite(filename, frame)
            # Update references
            prev_hist = []
            for channel in cv2.split(frame):
                h = cv2.calcHist([channel], [0], None, [256], [0, 256])
                h = cv2.normalize(h, h).flatten()
                prev_hist.append(h)
            prev_gray = curr_gray
            last_save_idx = frame_idx
            saved_count += 1

    cap.release()
```

**Key integration changes made in the MVP:**
- All `print()` calls replaced with the centralized `log()` function.
- Accepts `HardwareConfig` as a parameter.
- Returns a list of `ExtractedFrame` dataclass instances (not just saving files).
- Frame filenames simplified to `keyframe_XXXX.jpg` (no embedded frame index in filename).
- Timestamps computed from frame index and fps, formatted as `HH:MM:SS`.
- Raises `RuntimeError` (not silent return) if video cannot be opened.

---

## 12. Development Guidelines

### Code documentation standard

Every file must start with a module-level docstring explaining its purpose, inputs, and outputs. Every function must have a docstring. Every non-obvious logic block must have an inline comment. The test for "non-obvious" is simple: if you had to think about it for more than 5 seconds, it needs a comment.

```python
# ── Bad ──────────────────────────────────────
ssim_value = ssim(prev_gray, curr_gray, data_range=255)

# ── Good ─────────────────────────────────────
# SSIM measures structural similarity between two grayscale frames.
# A value close to 1.0 means nearly identical frames.
# We save a keyframe when this drops below ssim_threshold,
# indicating a meaningful scene change has occurred.
ssim_value = ssim(prev_gray, curr_gray, data_range=255)
```

### Module independence

Each module (`audio/`, `video/`, `llm/`, `utils/`) should be independently importable and testable. Cross-module communication is coordinated through `analyzer.py`. Shared types like `ExtractedFrame` and `HardwareConfig` are imported where needed.

### No magic numbers

Every numeric constant — thresholds, wait times, token limits, model names — lives in `config.py` or is defined as a named constant at module level (e.g. `VRAM_SAFETY_BUFFER_FRACTION`). If you write a literal number in a module file without a clear name, it is a bug.

### One-time downloads only

Model downloads (Whisper, HuggingFace vision model, Ollama models) must be gated by a cache check. The check must happen before any download is attempted, and the result must be logged. On re-runs, the log should confirm the cached model is being used, not re-downloaded.

### Fail loudly and clearly

If a required dependency is missing (ffmpeg, ollama binary, Python package), do not raise a cryptic `AttributeError` deep in the stack. Catch it at startup, log a specific human-readable message like `"ffmpeg binary not found. Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)"`, and raise a `RuntimeError` immediately.

---

## 13. Error Handling Strategy

| Scenario | Behavior |
|---|---|
| Video file not found | `FileNotFoundError` with full path in message |
| Video has no audio | Log warning, continue with `"[Video contains no audio track]"` as transcription |
| ffmpeg not installed | `RuntimeError` with install instructions |
| Ollama not installed | `RuntimeError` with install URL |
| Ollama model pull fails | Retry once via subprocess, then raise `RuntimeError` |
| Frame extraction produces 0 frames | Log warning, skip vision step, summarize from transcription only |
| Vision model fails to load | `RuntimeError` with model name and error message |
| Vision batch inference fails | Log warning, write placeholder text per frame, continue |
| Single frame image unreadable | Log warning, skip that frame, continue |
| LLM summary generation fails | Raise `RuntimeError` — this is unrecoverable |
| Q&A generation fails | Log warning, return placeholder text — does **not** crash |
| Temp directory cleanup fails | Log warning only — do not raise, the result is already returned |
| VRAM probe fails | Log warning, fall back to `batch_size=1` |

---

*End of specification.*
