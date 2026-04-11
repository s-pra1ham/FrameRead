"""
model_manager.py
Responsible for checking Whisper model cache and pulling if necessary.
Faster-whisper automatically saves to `~/.cache/huggingface/`, but we 
force standard logging around the pull.
"""
from faster_whisper import WhisperModel
from video_analyzer.config import WHISPER_MODEL
from video_analyzer.utils.logger import log

def ensure_whisper_model() -> None:
    """
    Idempotent check/download for distil-whisper logic.
    Since `faster-whisper` handles internal HF caching, we intercept it 
    to log what occurs.
    """
    try:
        log("MODEL", f"Checking local cache for {WHISPER_MODEL}...")
        
        # We perform a dummy load. If model is in cache, it's fast. 
        # If not, it triggers download exactly as required by the pipeline structure.
        # device="cpu" is harmless here as we're just loading weights to ensure presence 
        # before the true transcriber script does the memory-heavy setup.
        WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", download_root=None)
        
        log("MODEL", f"✓ {WHISPER_MODEL} is ready in cache.")
        
    except Exception as e:
        log("MODEL", f"Failed verifying {WHISPER_MODEL}: {str(e)}")
        raise RuntimeError(f"Unable to download/verify whisper model {WHISPER_MODEL}: {str(e)}")
