"""
extractor.py
Audio handler via ffmpeg to rip audio from the video safely.
"""
import os
import shutil
import subprocess
import time
from video_analyzer.utils.logger import log

def extract_audio(video_path: str, output_wav: str) -> bool:
    """
    Extracts the audio track from the given video file into a WAV format.
    
    Returns:
        True if extraction successful, False if no audio found.
    Raises:
        RuntimeError if ffmpeg is not installed or fails unrecoverably.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg binary not found. Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")

    # File stats
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    # Use ffprobe to get duration easily
    try:
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode("utf-8").strip())
    except Exception:
        duration = 0.0

    log("AUDIO", f"Extracting audio from video...")
    log("AUDIO", f"Source: {video_path} (duration: {duration:.1f}s, size: {size_mb:.1f}MB)")
    log("AUDIO", f"Writing audio → {output_wav}")

    start_time = time.time()
    try:
        # Run ffmpeg to extract audio (16kHz, mono) typical for Whisper
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",          # No video
            "-acodec", "pcm_s16le", # WAV format
            "-ar", "16000", # 16kHz
            "-ac", "1",     # Mono
            "-y",           # Overwrite output
            output_wav
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            err = result.stderr.decode("utf-8").lower()
            if "does not contain any stream" in err or "no such stream" in err:
                log("AUDIO", "Warning: Video file has no audio track. Transcription will be empty.")
                return False
            # Otherwise it's a real failure
            raise RuntimeError(f"ffmpeg extraction failed: {err}")
            
        elapsed = time.time() - start_time
        log("AUDIO", f"✓ Audio extracted in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise e
        raise RuntimeError(f"Unexpected error extracting audio: {str(e)}")
