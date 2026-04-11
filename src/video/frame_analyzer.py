"""
frame_analyzer.py
Iterates through extracted video frames and queries Vision context for each.
Now relies on native Hugging Face Transformers pipeline (Qwen2-VL) running locally.
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

def analyze_frames(frames: List[ExtractedFrame], output_txt: str, hardware: HardwareConfig) -> str:
    """
    Sends each extracted frame to the vision model via local HF Transformers.
    Supports concurrent batching automatically scaled by available VRAM.
    Writes progressive output to `output_txt` and returns the complete text.
    Assures aggressive cleanup of VRAM upon completion.
    """
    if not frames:
        log("VISION", "No frames provided to analyze. Returning empty.")
        return ""
        
    num_frames = len(frames)
    
    # Determine batch_size based on VRAM
    batch_size = 1
    if hardware.device == "cuda":
        if hardware.vram_gb >= 14.0:
            batch_size = 3
        elif hardware.vram_gb >= 10.0:
            batch_size = 2

    log("VISION", f"Analyzing {num_frames} keyframes with {VISION_MODEL}...")
    log("VISION", f"Vision Mode: Local Inference (Batch size: {batch_size})")

    total_start = time.time()
    
    # Load model and processor into memory
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

    log("VISION", "Model loaded. Starting batch processing...")
    
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            for i in range(0, num_frames, batch_size):
                batch_frames = frames[i:i+batch_size]
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
                    
                log("VISION", f"Processing frames {i+1} to {i+len(valid_frames)} of {num_frames}...")
                
                # Build messages for each image based on prompt
                messages_batch = [[{
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": FRAME_ANALYSIS_PROMPT}]
                }] for _ in images]
                
                try:
                    # Apply chat template and processor
                    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
                    
                    # Ensure properly mapped to device 
                    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(hardware.device)
                    
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, max_new_tokens=256)
                        
                    # Calculate new tokens purely by stripping prompt length from generated output
                    generated_ids = [output_ids[k][len(inputs.input_ids[k]):] for k in range(len(output_ids))]
                    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                except Exception as e:
                    log("VISION", f"Inference error on batch starting at {i}: {str(e)}")
                    decoded = [f"[Vision inference failed: {str(e)}]"] * len(valid_frames)
                
                batch_elapsed = time.time() - batch_start
                avg_time = batch_elapsed / len(valid_frames)
                
                # Zip strictly sequentially and stream out exactly as expected
                for frame, description in zip(valid_frames, decoded):
                    actual_idx = frames.index(frame) + 1
                    header = f"\n=== KEYFRAME {actual_idx:04d} | Frame #{frame.frame_idx} | Timestamp ~{frame.timestamp_str} ===\n"
                    f.write(header)
                    f.write(description.strip() + "\n\n")
                    f.flush()
                    log("VISION", f"✓ Frame {actual_idx:04d} described (~{avg_time:.1f}s per frame in batch)")

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
    log("VISION", f"✓ All frames analyzed — {total_elapsed:.1f}s total (Batch Size: {batch_size})")
    log("VISION", f"Writing → {output_txt}")

    # Return full read file
    with open(output_txt, 'r', encoding='utf-8') as f:
        return f.read()
