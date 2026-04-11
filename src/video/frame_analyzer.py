"""
frame_analyzer.py
Iterates through extracted video frames and queries Vision context for each.
Now relies on native Hugging Face Transformers pipeline (Qwen2-VL) running locally.

Batch size is determined dynamically at runtime by probing actual VRAM consumption
of a single frame, then scaling up to fill available memory minus a safety buffer.
"""
import os
import time
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import List

from src.config import VISION_MODEL, FRAME_ANALYSIS_PROMPT
from src.video.frame_extractor import ExtractedFrame
from src.utils.hardware import HardwareConfig
from src.utils.logger import log

# How much VRAM (in GB) to keep free as a safety margin to avoid OOM.
# Midpoint of the 1.5-2 GB target range.
VRAM_SAFETY_BUFFER_GB = 1.75


def _build_single_message():
    """Returns the chat message list for one image inference."""
    return [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": FRAME_ANALYSIS_PROMPT}]
    }]


def _run_inference(model, processor, images: list, device: str):
    """
    Run batched vision inference on a list of PIL images.
    Returns a list of decoded description strings (one per image).
    """
    messages_batch = [_build_single_message() for _ in images]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids = [
        output_ids[k][len(inputs.input_ids[k]):]
        for k in range(len(output_ids))
    ]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def _probe_vram_and_compute_batch_size(
    model, processor, first_frame: ExtractedFrame, hardware: HardwareConfig
) -> tuple[int, list[str]]:
    """
    PHASE 1-3 of the dynamic batching protocol.

    1. Record baseline VRAM (model weights already loaded).
    2. Run inference on a single frame and measure the peak VRAM delta.
    3. Calculate the optimal batch size that fills available VRAM minus the
       safety buffer.

    Returns:
        (optimal_batch_size, [description_of_first_frame])
        The first frame's result is returned so it is NOT wasted.
    """
    # -- Phase 1: Baseline -------------------------------------------------
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    log("VISION", f"-- VRAM Probe ------------------------------------------")
    log("VISION", f"Total VRAM:       {total_vram_gb:.2f} GB")
    log("VISION", f"Baseline (model): {baseline_vram_gb:.2f} GB")

    # -- Phase 2: Probe with 1 frame ---------------------------------------
    log("VISION", f"Probing VRAM cost with 1 frame...")
    img = Image.open(first_frame.path).convert("RGB")
    probe_decoded = _run_inference(model, processor, [img], hardware.device)

    torch.cuda.synchronize()
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    per_frame_vram_gb = peak_vram_gb - baseline_vram_gb

    # Guard against nonsensical measurements (e.g. shared memory reporting quirks)
    if per_frame_vram_gb <= 0:
        per_frame_vram_gb = 0.5  # sensible fallback estimate
        log("VISION", f"Warning: measured per-frame VRAM <= 0, using fallback {per_frame_vram_gb:.2f} GB")

    log("VISION", f"Peak after probe: {peak_vram_gb:.2f} GB")
    log("VISION", f"Per-frame cost:   {per_frame_vram_gb:.2f} GB")

    # -- Phase 3: Compute optimal batch size --------------------------------
    usable_vram_gb = total_vram_gb - baseline_vram_gb - VRAM_SAFETY_BUFFER_GB
    optimal_batch = max(1, int(usable_vram_gb / per_frame_vram_gb))

    log("VISION", f"Usable VRAM:      {usable_vram_gb:.2f} GB (after {VRAM_SAFETY_BUFFER_GB} GB buffer)")
    log("VISION", f"Optimal batch:    {optimal_batch} frames")
    log("VISION", f"----------------------------------------------------")

    return optimal_batch, probe_decoded


def _write_frame_result(f, frame: ExtractedFrame, description: str, actual_idx: int, avg_time: float):
    """Write a single frame's analysis to the output file and log it."""
    header = (
        f"\n=== KEYFRAME {actual_idx:04d} | Frame #{frame.frame_idx} "
        f"| Timestamp ~{frame.timestamp_str} ===\n"
    )
    f.write(header)
    f.write(description.strip() + "\n\n")
    f.flush()
    log("VISION", f"Frame {actual_idx:04d} described (~{avg_time:.1f}s per frame in batch)")


def analyze_frames(frames: List[ExtractedFrame], output_txt: str, hardware: HardwareConfig) -> str:
    """
    Sends each extracted frame to the vision model via local HF Transformers.

    Batch size is determined dynamically at runtime:
      1. Load model and record baseline VRAM.
      2. Process 1 frame and measure actual per-frame VRAM cost.
      3. Calculate max batch size that leaves ~1.75 GB free.
      4. Process all remaining frames using that batch size.

    Writes progressive output to `output_txt` and returns the complete text.
    Assures aggressive cleanup of VRAM upon completion.
    """
    if not frames:
        log("VISION", "No frames provided to analyze. Returning empty.")
        return ""

    num_frames = len(frames)
    log("VISION", f"Analyzing {num_frames} keyframes with {VISION_MODEL}...")

    total_start = time.time()

    # -- Load model & processor --------------------------------------------
    log("VISION", "Loading local vision model into memory. This may take a moment...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            VISION_MODEL,
            torch_dtype=hardware.torch_dtype,
            device_map=hardware.device
        )
        processor = AutoProcessor.from_pretrained(VISION_MODEL)
    except Exception as e:
        log("VISION", f"Fatal error loading vision model: {str(e)}")
        raise RuntimeError(f"Could not load HuggingFace vision model {VISION_MODEL}: {str(e)}")

    log("VISION", "Model loaded successfully.")

    # -- Determine batch size ----------------------------------------------
    batch_size = 1  # default / CPU fallback

    if hardware.device == "cuda" and num_frames > 0:
        try:
            probe_start = time.time()
            batch_size, probe_result = _probe_vram_and_compute_batch_size(
                model, processor, frames[0], hardware
            )
            probe_elapsed = time.time() - probe_start
        except Exception as e:
            log("VISION", f"VRAM probe failed ({str(e)}), falling back to batch_size=1")
            probe_result = None
            probe_elapsed = 0
    else:
        probe_result = None
        probe_elapsed = 0
        if hardware.device != "cuda":
            log("VISION", "CPU mode -- batch size fixed at 1.")

    log("VISION", f"Vision Mode: Local Inference (Dynamic Batch size: {batch_size})")

    # -- Process all frames ------------------------------------------------
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:

            # Write probe frame result first (frame[0]) if it succeeded
            start_idx = 0
            if probe_result is not None:
                _write_frame_result(f, frames[0], probe_result[0], 1, probe_elapsed)
                start_idx = 1  # skip frame[0] in the main loop

            # Main batch loop over remaining frames
            remaining_frames = frames[start_idx:]
            for i in range(0, len(remaining_frames), batch_size):
                batch_frames = remaining_frames[i:i + batch_size]
                batch_start = time.time()

                # Open images safely
                images = []
                valid_frames = []
                for frame in batch_frames:
                    try:
                        img = Image.open(frame.path).convert("RGB")
                        images.append(img)
                        valid_frames.append(frame)
                    except Exception as e:
                        log("VISION", f"Warning: Failed to read image {frame.path}: {str(e)}")

                if not images:
                    continue

                global_start = start_idx + i + 1
                global_end = global_start + len(valid_frames) - 1
                log("VISION", f"Processing frames {global_start} to {global_end} of {num_frames}...")

                try:
                    decoded = _run_inference(model, processor, images, hardware.device)
                except Exception as e:
                    log("VISION", f"Inference error on batch starting at frame {global_start}: {str(e)}")
                    decoded = [f"[Vision inference failed: {str(e)}]"] * len(valid_frames)

                batch_elapsed = time.time() - batch_start
                avg_time = batch_elapsed / len(valid_frames)

                for frame, description in zip(valid_frames, decoded):
                    actual_idx = frames.index(frame) + 1
                    _write_frame_result(f, frame, description, actual_idx, avg_time)

    finally:
        # ABSOLUTE CRITICAL CLEANUP: Force free GPU overhead so Ollama can live securely.
        log("VISION", "Unloading model and freeing CUDA cache safely...")
        try:
            del model
            del processor
        except NameError:
            pass  # if initialization failed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    log("VISION", f"All frames analyzed -- {total_elapsed:.1f}s total (Dynamic Batch Size: {batch_size})")
    log("VISION", f"Writing -> {output_txt}")

    # Return full read file
    with open(output_txt, 'r', encoding='utf-8') as f:
        return f.read()
