"""
logger.py
Centralized logger handling standardized pipeline log formatting.
Format: [TIMESTAMP] [MODULE     ] Message
"""
import time
from datetime import datetime
import sys

_START_TIME = time.time()

def log(module: str, message: str):
    """
    Logs a message to stdout following the system-wide logging standard.
    
    Args:
        module: The name of the module/component (e.g. 'AUDIO', 'TRANSCRIBE'). 
                Will be padded to 12 chars.
        message: The actual log content.
    """
    # Calculate elapsed pipeline time
    elapsed = time.time() - _START_TIME
    
    # Format elapsed time as HH:MM:SS.mmm
    ms = int((elapsed % 1) * 1000)
    minutes, seconds = divmod(int(elapsed), 60)
    hours, minutes = divmod(minutes, 60)
    
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
    
    # Format the module name to be fixed length (12 chars)
    padded_module = f"[{module.upper().strip()}]".ljust(12)
    
    # Print exactly as required
    print(f"[{timestamp}] {padded_module} {message}")
    sys.stdout.flush()

def reset_timer():
    """Reset the pipeline start time (useful for multiple runs in one session)."""
    global _START_TIME
    _START_TIME = time.time()
