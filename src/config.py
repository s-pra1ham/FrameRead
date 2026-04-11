"""
config.py
All generic constants, thresholds, model names, and system prompts.
"""

# ── Frame Extraction ──────────────────────────────────
HIST_THRESHOLD       = 0.28    # χ² histogram distance. Lower = more sensitive.
SSIM_THRESHOLD       = 0.89    # Structural similarity. Lower = more keyframes.
MIN_FRAME_INTERVAL   = 8       # Minimum frames between two saved keyframes.

# ── Models ────────────────────────────────────────────
WHISPER_MODEL        = "distil-large-v3"
VISION_MODEL         = "Qwen/Qwen2-VL-2B-Instruct"
SUMMARY_MODEL        = "qwen3.5:9b"

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
"""

# ── Q&A prompt ────────────────────────────────────────
QA_SYSTEM_PROMPT = """
You are an expert analyst. You have access to a detailed summary of a video (provided below).
A user has asked a specific question about this video.

Your task:
  1. Answer the user's question precisely and completely.
  2. After your answer, cite the relevant sections of the summary that support your answer,
     quoting or paraphrasing key lines to show your reasoning.

Do not go beyond the summary. If the answer cannot be found in the summary, say so clearly.
"""
