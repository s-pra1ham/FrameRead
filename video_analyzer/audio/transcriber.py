"""
transcriber.py
Uses faster-whisper to generate dense timestamped dialogue out of audio.
"""
import time
from faster_whisper import WhisperModel
from video_analyzer.config import WHISPER_MODEL
from video_analyzer.utils.hardware import HardwareConfig
from video_analyzer.utils.logger import log

def transcribe_audio(audio_path: str, output_txt: str, hardware: HardwareConfig) -> str:
    """
    Transcribes audio into timestamped segments using faster-whisper.
    Returns the full string of transcription and writes to a text file.
    
    Outputs format:
        [HH:MM:SS -> HH:MM:SS] text
    """
    log("TRANSCRIBE", f"Loading {WHISPER_MODEL} (faster-whisper)...")
    log("TRANSCRIBE", f"Device: {hardware.device.upper()} | Compute: {hardware.whisper_dtype}")
    
    try:
        model = WhisperModel(
            model_size_or_path=WHISPER_MODEL,
            device=hardware.device,
            compute_type=hardware.whisper_dtype,
            download_root=None
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load WhisperModel: {str(e)}")
        
    log("TRANSCRIBE", "Starting transcription...")
    
    start_time = time.time()
    try:
        segments_generator, info = model.transcribe(audio_path, beam_size=5)
    except Exception as e:
        raise RuntimeError(f"faster-whisper inference failed: {str(e)}")

    full_transcript = []
    word_count = 0
    segment_count = 0
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        # iterate through segments lazy-loaded
        for segment in segments_generator:
            segment_count += 1
            
            # Format times: seconds to HH:MM:SS
            def fmt_time(t):
                h, r = divmod(t, 3600)
                m, s = divmod(r, 60)
                return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
                
            start_fmt = fmt_time(segment.start)
            end_fmt = fmt_time(segment.end)
            text = segment.text.strip()
            
            word_count += len(text.split())
            
            line = f"[{start_fmt} → {end_fmt}] {text}"
            full_transcript.append(line)
            f.write(line + "\n")
            
            # Optionally log progress every few segments to avoid terminal spam
            if segment_count % 5 == 1 or segment_count == 1:
                short_text = text[:30] + "..." if len(text) > 30 else text
                log("TRANSCRIBE", f"Progress: segment {segment_count} — \"{short_text}\"")
                
    elapsed = time.time() - start_time
    log("TRANSCRIBE", f"✓ Transcription complete — {segment_count} segments, ~{word_count} words, {elapsed:.1f}s elapsed")
    log("TRANSCRIBE", f"Writing → {output_txt}")
    
    return "\n".join(full_transcript)
