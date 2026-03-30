"""
analyzer.py
The top-level orchestrator. Ties all utilities, audio extraction, video extraction,
and LLM synthesis together into the finalized pipeline.
"""
import os
import time
from dataclasses import dataclass

from video_analyzer.config import PRINT_SUMMARY
from video_analyzer.utils.logger import log, reset_timer
from video_analyzer.utils.hardware import detect_hardware
from video_analyzer.utils.cleanup import TempDirManager
from video_analyzer.utils.model_manager import ensure_whisper_model

from video_analyzer.llm.ollama_manager import check_ollama_process, ensure_models
from video_analyzer.audio.extractor import extract_audio
from video_analyzer.audio.transcriber import transcribe_audio
from video_analyzer.video.frame_extractor import extract_frames
from video_analyzer.video.frame_analyzer import analyze_frames
from video_analyzer.llm.summarizer import generate_master_summary, answer_with_summary


@dataclass
class AnalysisResult:
    summary: str
    prompt_answer: str | None
    keyframe_count: int
    transcription: str
    duration_seconds: float
    video_path: str


def analyze(video_path: str, prompt: str | None = None) -> AnalysisResult:
    """
    Main entry point. Runs the full video analysis pipeline.
    
    Args:
        video_path: Absolute or relative path to a video file.
        prompt: Optional specific query regarding the video. If omitted,
                will simply generate the exhaustive master summary.
                
    Returns:
        AnalysisResult containing the summary, transcription, timing, and Q&A (if requested).
    """
    wall_start = time.time()
    reset_timer()
    
    # ── Input Layer ───────────────────────────────────────────────────────────
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # Start the engine
    log("INIT", f"Starting VideoAnalyzer pipeline for: {video_path}")
    
    # Diagnostics & verification
    hardware_cfg = detect_hardware()
    ensure_whisper_model()
    check_ollama_process()
    ensure_models()

    transcription = ""
    frame_analyses = ""
    keyframe_count = 0
    prompt_answer = None
    master_summary = ""
    
    # ── Pipeline Execution ────────────────────────────────────────────────────
    with TempDirManager() as temp_dir:
        audio_wav = os.path.join(temp_dir, "audio.wav")
        transcript_txt = os.path.join(temp_dir, "transcription.txt")
        frames_dir = os.path.join(temp_dir, "frames")
        frames_txt = os.path.join(temp_dir, "frame_analyses.txt")
        
        # 1. Audio
        has_audio = extract_audio(video_path, audio_wav)
        if has_audio:
            transcription = transcribe_audio(audio_wav, transcript_txt, hardware_cfg)
        else:
            transcription = "[Video contains no audio track]"
            
        # 2. Video frames extraction
        extracted_frames = extract_frames(video_path, frames_dir, hardware_cfg)
        keyframe_count = len(extracted_frames)
        
        # 3. Vision inference per frame
        if keyframe_count > 0:
            frame_analyses = analyze_frames(extracted_frames, frames_txt, hardware_cfg)
        else:
            log("FRAMES", "Warning: 0 keyframes extracted. Skipping vision analysis.")
            frame_analyses = "[No significant visual frames detected]"
            
        # 4. Master Synth
        master_summary = generate_master_summary(transcription, frame_analyses, keyframe_count)
        
        # 5. Optional Q&A Layer
        if prompt and prompt.strip():
            prompt_answer = answer_with_summary(master_summary, prompt)
            
    # Pipeline done, context manager tears down temp
    wall_duration = time.time() - wall_start
    log("DONE", f"✨ Total pipeline time: {wall_duration:.1f}s")
    
    result = AnalysisResult(
        summary=master_summary,
        prompt_answer=prompt_answer,
        keyframe_count=keyframe_count,
        transcription=transcription,
        duration_seconds=wall_duration,
        video_path=video_path
    )
    
    if PRINT_SUMMARY:
        print("\n\n" + "="*80)
        print(" MASTER SUMMARY ".center(80, "="))
        print("="*80 + "\n")
        print(master_summary)
        print("\n" + "="*80)
        
        if prompt_answer:
            print(f"\nQ: {prompt}\n")
            print(f"A: {prompt_answer}\n")
            print("="*80 + "\n")
            
    return result
