"""
hardware.py
Responsible for GPU/CPU hardware detection using PyTorch, 
and computing the optimal configurations for vision processing.
"""
import torch
from dataclasses import dataclass
from video_analyzer.utils.logger import log

@dataclass
class HardwareConfig:
    device: str
    gpu_name: str | None
    vram_gb: float
    whisper_dtype: str
    torch_dtype: torch.dtype | None

def detect_hardware() -> HardwareConfig:
    """
    Detects hardware specifications and builds the configuration object.
    Applies the GPU tier optimization model to select proper dtypes.
    """
    device = "cpu"
    gpu_name = None
    vram_gb = 0.0
    whisper_dtype = "int8"
    torch_dtype = None

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # High end: >= 20GB (A100, H100, RTX 4090) -> use bfloat16
        if vram_gb >= 20.0:
            torch_dtype = torch.bfloat16
            whisper_dtype = "float16"
        # Mid/low end: < 20GB -> use float16
        else:
            torch_dtype = torch.float16
            whisper_dtype = "float16"
            
    config = HardwareConfig(
        device=device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        whisper_dtype=whisper_dtype,
        torch_dtype=torch_dtype
    )

    log("HARDWARE", "── Hardware Survey ──────────────────────────")
    if config.device == "cuda":
        log("HARDWARE", f"Device:       CUDA (GPU)")
        log("HARDWARE", f"GPU:          {config.gpu_name}")
        log("HARDWARE", f"VRAM:         {config.vram_gb:.1f} GB")
        log("HARDWARE", f"Torch dtype:  {str(config.torch_dtype).split('.')[-1]}")
    else:
        log("HARDWARE", f"Device:       CPU")
        log("HARDWARE", f"Whisper dtype: {config.whisper_dtype}")
    log("HARDWARE", "─────────────────────────────────────────────")

    return config
