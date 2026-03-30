"""
frame_analyzer.py
Iterates through extracted video frames and queries Vision API context for each.
"""
import os
import json
import base64
import time
import requests
from typing import List
from video_analyzer.config import OLLAMA_HOST, VISION_MODEL, FRAME_ANALYSIS_PROMPT
from video_analyzer.video.frame_extractor import ExtractedFrame
from video_analyzer.utils.hardware import HardwareConfig
from video_analyzer.utils.logger import log

def analyze_frames(frames: List[ExtractedFrame], output_txt: str, hardware: HardwareConfig) -> str:
    """
    Sends each extracted frame to the vision model via Ollama API.
    Supports concurrent processing automatically scaled by available VRAM.
    Writes progressive output to `output_txt` and returns the complete text.
    """
    import concurrent.futures
    
    if not frames:
        log("VISION", "No frames provided to analyze. Returning empty.")
        return ""
        
    num_frames = len(frames)
    
    # Determine concurrency based on VRAM
    workers = 1
    mode = "SEQUENTIAL"
    if hardware.device == "cuda":
        if hardware.vram_gb >= 14.0:
            workers = 3
            mode = f"CONCURRENT-3 (VRAM: {hardware.vram_gb:.1f}GB)"
        elif hardware.vram_gb >= 10.0:
            workers = 2
            mode = f"CONCURRENT-2 (VRAM: {hardware.vram_gb:.1f}GB)"

    log("VISION", f"Analyzing {num_frames} keyframes with {VISION_MODEL}...")
    log("VISION", f"Vision Mode: {mode} ({workers} workers)")

    total_start = time.time()
    
    def _process_frame(idx: int, frame: ExtractedFrame):
        frame_log_id = f"{(idx+1):04d}/{num_frames} (frame #{frame.frame_idx}, ~{frame.timestamp_str})"
        log("VISION", f"Processing frame {frame_log_id}...")
        
        try:
            with open(frame.path, "rb") as img_file:
                b64_str = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            log("VISION", f"Warning: Failed to read image {frame.path}: {str(e)}")
            err_text = f"=== KEYFRAME {(idx+1):04d} | Frame #{frame.frame_idx} | Timestamp ~{frame.timestamp_str} ===\n[Error reading image file on disk.]\n\n"
            return err_text, 0, 0.0
            
        frame_start = time.time()
        
        try:
            payload = {
                "model": VISION_MODEL,
                "prompt": FRAME_ANALYSIS_PROMPT,
                "images": [b64_str],
                "stream": False
            }
            resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            description = data.get("response", "").strip()
            tokens = data.get("eval_count", 0)
        except requests.exceptions.RequestException as e:
            log("VISION", f"Warning: Vision model API failure for Frame {idx+1}: {str(e)}")
            description = f"[Vision inference failed: {str(e)}]"
            tokens = 0
            
        frame_elapsed = time.time() - frame_start
        
        header = f"\n=== KEYFRAME {(idx+1):04d} | Frame #{frame.frame_idx} | Timestamp ~{frame.timestamp_str} ===\n"
        formatted_out = header + description + "\n\n"
        
        return formatted_out, tokens, frame_elapsed

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for idx, frame in enumerate(frames):
            futures.append(executor.submit(_process_frame, idx, frame))
            
        with open(output_txt, 'w', encoding='utf-8') as f:
            for idx, future in enumerate(futures):
                formatted_out, tokens, frame_elapsed = future.result()
                f.write(formatted_out)
                f.flush()
                log("VISION", f"✓ Frame {(idx+1):04d} described ({tokens} tokens, {frame_elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    log("VISION", f"✓ All frames analyzed — {total_elapsed:.1f}s total ({mode})")
    log("VISION", f"Writing → {output_txt}")

    # Return full read file
    with open(output_txt, 'r', encoding='utf-8') as f:
        return f.read()
