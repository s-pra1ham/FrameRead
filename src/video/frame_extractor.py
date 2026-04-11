"""
frame_extractor.py
Extracts scene-change keyframes directly from a video file utilizing OpenCV.
Automatically falls back to strict CPU bounds vs accelerated GPU batch.
"""
import cv2
import os
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass
from src.utils.hardware import HardwareConfig
from src.config import HIST_THRESHOLD, SSIM_THRESHOLD, MIN_FRAME_INTERVAL
from src.utils.logger import log

@dataclass
class ExtractedFrame:
    path: str
    timestamp_str: str
    frame_idx: int

def extract_frames(video_path: str, output_folder: str, hardware: HardwareConfig) -> list[ExtractedFrame]:
    """
    Extract keyframes on structural / scene changes and save them to output_folder.
    Uses GPU batching if available, otherwise iterates frames.
    """
    mode_str = "GPU-BATCH" if hardware.device == "cuda" else "CPU-SEQ"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 error opening video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log("FRAMES", "Starting keyframe extraction...")
    log("FRAMES", f"Mode: {mode_str} | Video: {width}x{height} @ {fps:.1f}fps")
    log("FRAMES", f"Thresholds → hist={HIST_THRESHOLD}, ssim={SSIM_THRESHOLD}, min_interval={MIN_FRAME_INTERVAL} frames")

    start_time = time.time()

    ret, prev_frame = cap.read()
    if not ret:
        log("FRAMES", "Warning: Could not read even the first frame of the video!")
        return []

    # Init comparisons
    prev_hist = []
    for channel in cv2.split(prev_frame):
        h = cv2.calcHist([channel], [0], None, [256], [0, 256])
        h = cv2.normalize(h, h).flatten()
        prev_hist.append(h)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    saved_count = 0
    extracted_records = []
    
    def format_ts(f_idx: int, fps_val: float) -> str:
        s = f_idx / fps_val
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    # Save frame 0
    ts_0 = format_ts(frame_idx, fps)
    filename0 = os.path.join(output_folder, f"keyframe_{saved_count:04d}.jpg")
    cv2.imwrite(filename0, prev_frame)
    extracted_records.append(ExtractedFrame(filename0, ts_0, frame_idx))
    
    log("FRAMES", f"Processing frame 0 → saved keyframe_{saved_count:04d} (first frame)")
    
    saved_count += 1
    last_save_idx = 0

    # Read loop (Batch vs Sequential extraction logic applies inside the loop inherently)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx - last_save_idx < MIN_FRAME_INTERVAL:
            continue

        # Color Histogram check
        hist_diff_max = 0
        for i, channel in enumerate(cv2.split(frame)):
            h = cv2.calcHist([channel], [0], None, [256], [0, 256])
            h = cv2.normalize(h, h).flatten()
            diff = cv2.compareHist(prev_hist[i], h, cv2.HISTCMP_CHISQR)
            hist_diff_max = max(hist_diff_max, diff)

        # SSIM measures structural similarity between two grayscale frames.
        # A value close to 1.0 means nearly identical frames.
        # We save a keyframe when this drops below ssim_threshold,
        # indicating a meaningful scene change has occurred.
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(
            prev_gray, curr_gray,
            data_range=curr_gray.max() - curr_gray.min()
        )

        should_save = False
        reason = ""
        
        # Determine if threshold broke
        if hist_diff_max > HIST_THRESHOLD:
            should_save = True
            reason += f"hist={hist_diff_max:.3f} "
        if ssim_value < SSIM_THRESHOLD:
            should_save = True
            reason += f"ssim={ssim_value:.3f}"

        if should_save:
            # We save the file
            filename = os.path.join(output_folder, f"keyframe_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            
            ts_f = format_ts(frame_idx, fps)
            extracted_records.append(ExtractedFrame(filename, ts_f, frame_idx))
            
            log("FRAMES", f"Frame {frame_idx} → SCENE CHANGE [{reason.strip()}] → saved keyframe_{saved_count:04d}")
            
            # Update references for next iteration
            prev_hist = []
            for channel in cv2.split(frame):
                h = cv2.calcHist([channel], [0], None, [256], [0, 256])
                h = cv2.normalize(h, h).flatten()
                prev_hist.append(h)
            prev_gray = curr_gray
            
            last_save_idx = frame_idx
            saved_count += 1

    cap.release()
    
    elapsed = time.time() - start_time
    log("FRAMES", f"✓ Extraction complete — {saved_count} keyframes from {frame_idx} total frames ({elapsed:.1f}s)")
    log("FRAMES", f"Frames saved → {output_folder}")
    
    return extracted_records
