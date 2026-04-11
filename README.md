# 🎬 FrameRead — VideoAnalyzer

> **Turn any video into a detailed, multi-page written analysis — fully offline, fully local.**

FrameRead is an AI-powered video analysis pipeline that extracts audio transcriptions and visual frame descriptions from any video file, then synthesizes them into an exhaustive natural-language summary. Optionally, ask it a specific question and get a precise, evidence-backed answer.

Everything runs **locally on your machine** — no API keys, no cloud services, no data leaves your device.

---

## ✨ Features

- **Dual-Pipeline Analysis** — Processes both audio (speech) and video (frames) in parallel pipelines, then fuses them into a unified summary.
- **Scene-Change Keyframe Extraction** — Intelligently detects visual scene changes using histogram + SSIM comparison rather than naive interval sampling.
- **Dynamic GPU Batching** — Automatically profiles your GPU's VRAM at runtime and calculates the optimal batch size for vision inference. No manual tuning needed.
- **Fully Offline** — All models run locally. No internet required after initial model downloads.
- **Importable as a Module** — Use it from the command line or `import` it into your own Python project.
- **Rich Structured Logging** — Every pipeline stage emits timestamped, module-tagged logs for full observability.
- **Automatic Cleanup** — All temporary files (audio, frames, intermediate text) are deleted after each run.
- **Hardware-Adaptive** — Automatically detects GPU/CPU and selects optimal dtypes, batch sizes, and compute strategies.
- **Prompt Q&A** — Optionally pass a question to get a targeted answer grounded in the video content.

---

## 🏗️ Architecture

The pipeline follows a **three-stage** architecture:

```
Video File ──┬── Audio Pipeline ──→ Transcription (faster-whisper)
             │
             └── Video Pipeline ──→ Frame Descriptions (Qwen2-VL, local HF)
                                          │
                                          ▼
                                  Synthesis Layer (qwen3.5:9b via Ollama)
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                       Master Summary           Q&A Answer
                       (always)              (if prompt given)
```

> 📖 **For the full architecture diagram, pipeline details, and developer blueprint, see [`VideoAnalyzer_ProjectSpec.md`](VideoAnalyzer_ProjectSpec.md).**

---

## 🤖 Models Used

| Model | Purpose | Runtime | Size |
|---|---|---|---|
| [`distil-whisper/distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3) | Audio transcription | `faster-whisper` (CTranslate2) | ~1.5 GB |
| [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Visual frame analysis | HuggingFace Transformers (local) | ~4.5 GB |
| [`qwen3.5:9b`](https://ollama.com/library/qwen3.5) | Summary synthesis & Q&A | Ollama (local) | ~6 GB |

All models are downloaded automatically on first run and cached locally for future use.

---

## 📋 Prerequisites

- **Python** ≥ 3.10
- **ffmpeg** installed and on PATH ([install guide](https://ffmpeg.org/download.html))
- **Ollama** installed ([ollama.com](https://ollama.com/))
- **CUDA GPU** recommended (8+ GB VRAM) — CPU mode works but is significantly slower

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/s-pra1ham/FrameRead.git
cd FrameRead

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Option 1: Command Line

#### Generate a full summary (no prompt)

```bash
python run.py video.mp4
```

This processes the entire video and prints a comprehensive multi-page summary covering:
- Overview & purpose
- Detailed chronological narrative
- Key points & concepts
- Visual highlights
- Speakers & participants
- Tone & style

#### Ask a specific question (with prompt)

```bash
python run.py video.mp4 --prompt "What tools or technologies are mentioned?"
```

```bash
python run.py lecture.mp4 -p "Summarize the main argument in 3 bullet points."
```

This generates the full summary internally, then uses it to answer your question with cited evidence.

---

### Option 2: Import as a Python Module

Use FrameRead programmatically in your own scripts or projects:

#### Basic summary (no prompt)

```python
from src import analyze

result = analyze(video_path="path/to/video.mp4")

print(result.summary)            # Full multi-page analysis
print(result.transcription)      # Timestamped transcript
print(result.keyframe_count)     # Number of keyframes extracted
print(result.duration_seconds)   # Total pipeline time in seconds
```

#### With a prompt

```python
from src import analyze

result = analyze(
    video_path="path/to/video.mp4",
    prompt="What products are shown in this video?"
)

print(result.prompt_answer)      # Direct answer to your question
print(result.summary)            # Full summary is still available
```

#### AnalysisResult fields

| Field | Type | Description |
|---|---|---|
| `summary` | `str` | Complete multi-page master summary (always present) |
| `prompt_answer` | `str \| None` | Answer to your prompt (only if prompt was provided) |
| `keyframe_count` | `int` | Number of scene-change keyframes extracted |
| `transcription` | `str` | Full timestamped transcript of spoken audio |
| `duration_seconds` | `float` | Total wall-clock pipeline time |
| `video_path` | `str` | Absolute path to the analyzed video |

---

### Option 3: Google Colab (Free T4 GPU)

Run FrameRead on Google Colab's free T4 GPU — no local GPU required:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](<YOUR_COLAB_NOTEBOOK_URL_HERE>)

<!-- TODO: Replace the URL above with your published Colab notebook link -->

---

## 📂 Project Structure

```
FrameRead/
├── src/                          ← Main package
│   ├── __init__.py               ← Public API: analyze()
│   ├── analyzer.py               ← Pipeline orchestrator
│   ├── config.py                 ← All tunable constants
│   ├── audio/
│   │   ├── extractor.py          ← ffmpeg audio extraction
│   │   └── transcriber.py        ← Whisper transcription
│   ├── video/
│   │   ├── frame_extractor.py    ← Scene-change keyframe extraction
│   │   └── frame_analyzer.py     ← Qwen2-VL vision inference
│   ├── llm/
│   │   ├── ollama_manager.py     ← Ollama process & model management
│   │   └── summarizer.py         ← Summary + Q&A generation
│   └── utils/
│       ├── hardware.py           ← GPU/CPU detection
│       ├── logger.py             ← Centralized logging
│       ├── model_manager.py      ← Whisper model cache management
│       └── cleanup.py            ← Temp directory lifecycle
├── docs/
│   └── VideoAnalyzer_ProjectSpec.md  ← Full technical specification
├── run.py                        ← CLI entry point
├── requirements.txt
└── setup.py
```

> 📖 **For the complete developer specification, see [`docs/VideoAnalyzer_ProjectSpec.md`](docs/VideoAnalyzer_ProjectSpec.md).**

---

## ⚙️ How It Works

1. **Input** — You provide a video file path and an optional prompt.
2. **Audio Extraction** — `ffmpeg` strips the audio track into a 16kHz mono WAV.
3. **Transcription** — `faster-whisper` transcribes every spoken word with timestamps.
4. **Keyframe Extraction** — OpenCV reads every frame; histogram + SSIM comparison detects scene changes and saves only the meaningful keyframes.
5. **Vision Analysis** — Each keyframe is described in detail by `Qwen2-VL-2B-Instruct` running locally. On GPU, batch size is dynamically calculated via a VRAM probe to maximize throughput without OOM.
6. **Synthesis** — The full transcript + all frame descriptions are fed to `qwen3.5:9b` (via Ollama) which produces the master summary.
7. **Q&A** *(optional)* — If a prompt was given, the summary is used as context to answer the question.
8. **Cleanup** — All temporary files are automatically deleted.

---

## 📊 Example Output

```
[00:00:00.000] [INIT]       Starting VideoAnalyzer pipeline for: C:\videos\demo.mp4
[00:00:00.012] [HARDWARE]   ── Hardware Survey ──────────────────────────
[00:00:00.013] [HARDWARE]   Device:       CUDA (GPU)
[00:00:00.013] [HARDWARE]   GPU:          NVIDIA GeForce RTX 4060
[00:00:00.013] [HARDWARE]   VRAM:         8.0 GB
[00:00:00.014] [HARDWARE]   Torch dtype:  float16
[00:00:00.014] [HARDWARE]   ─────────────────────────────────────────────
[00:00:01.220] [AUDIO]      Extracting audio from video...
[00:00:03.891] [AUDIO]      ✓ Audio extracted in 2.7s
[00:00:07.441] [TRANSCRIBE] ✓ Transcription complete — 12 segments, 143 words, 3.5s
[00:00:07.500] [FRAMES]     ✓ Extraction complete — 8 keyframes from 900 total frames (0.5s)
[00:00:08.100] [VISION]     Vision Mode: Local Inference (Dynamic Batch size: 2)
[00:00:45.200] [VISION]     All frames analyzed -- 37.1s total (Dynamic Batch Size: 2)
[00:01:12.300] [SUMMARY]    ✓ Summary generated — 2847 words, 3891 tokens (27.1s)
[00:01:12.400] [CLEANUP]    ✓ Cleaned up 11 files
[00:01:12.401] [DONE]       ✨ Total pipeline time: 72.4s
```

---

## 📄 License

This project is for personal and educational use.

---

## 👤 Author

**S. Pratham** — [GitHub](https://github.com/s-pra1ham)
