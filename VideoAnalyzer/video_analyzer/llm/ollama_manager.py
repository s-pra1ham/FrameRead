"""
ollama_manager.py
Ensures Ollama is running and required models are pulled.
Does startup checks and subprocess spin-ups.
"""
import time
import requests
import subprocess
import shutil
import json
from video_analyzer.config import OLLAMA_HOST, OLLAMA_STARTUP_WAIT, VISION_MODEL, SUMMARY_MODEL
from video_analyzer.utils.logger import log

def check_ollama_process() -> None:
    """Checks if Ollama is running; starts it via subprocess if not."""
    
    # First check if the Ollama binary exists
    if not shutil.which("ollama"):
        raise RuntimeError("Ollama binary not found. Please install from https://ollama.com/")
    
    log("OLLAMA", "Checking if Ollama is running...")
    
    # Attempt to ping
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if r.status_code == 200:
            log("OLLAMA", f"✓ Ollama is live at {OLLAMA_HOST}")
            return
    except requests.exceptions.RequestException:
        pass
        
    # Not running, need to start it
    log("OLLAMA", "Not running — starting ollama serve...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Poll for readiness
        for i in range(OLLAMA_STARTUP_WAIT):
            try:
                r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
                if r.status_code == 200:
                    log("OLLAMA", f"✓ Ollama is live at {OLLAMA_HOST}")
                    return
            except requests.exceptions.RequestException:
                time.sleep(1)
                
        raise RuntimeError(f"Ollama failed to start after {OLLAMA_STARTUP_WAIT} seconds.")
    except Exception as e:
        raise RuntimeError(f"Could not start Ollama process: {str(e)}")


def ensure_models() -> None:
    """Checks if required Ollama models are present. Pulls them if not."""
    required = [VISION_MODEL, SUMMARY_MODEL]
    log("OLLAMA", f"Required models: {', '.join(required)}")
    
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags")
        r.raise_for_status()
        existing = [m["name"] for m in r.json().get("models", [])]
        
        for model in required:
            # Handle possible lack of explicit tag in Ollama output
            # (e.g. they ask for "qwen3.5:9b", ollama returns "qwen3.5:9b")
            has_model = False
            for e in existing:
                if model in e or e in model:  
                    has_model = True
                    break
                    
            if has_model:
                log("OLLAMA", f"✓ {model} — found")
            else:
                log("OLLAMA", f"✗ {model} — not found")
                _pull_model(model)
                
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to query Ollama models: {str(e)}")


def _pull_model(model: str) -> None:
    """Pulls a single Ollama model via the REST API with progress tracking."""
    log("OLLAMA", f"Pulling {model} (one-time download)...")
    
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model, "stream": True}, stream=True)
        r.raise_for_status()
        
        last_percent = -1
        # Streaming response comes back as newline-delimited JSON
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status")
                
                # If there's download progress
                if "total" in data and "completed" in data and data["total"] > 0:
                    percent = int((data["completed"] / data["total"]) * 100)
                    
                    # Log every ~20%
                    if percent >= last_percent + 20 or percent == 100:
                        log("OLLAMA", f"Pull progress ({model}): {percent}%")
                        last_percent = percent
        
        log("OLLAMA", f"✓ {model} ready.")
        
    except requests.exceptions.RequestException as e:
        log("OLLAMA", f"Failed to pull {model}: {str(e)}")
        log("OLLAMA", f"Retrying once via subprocess...")
        
        try:
            subprocess.run(["ollama", "pull", model], check=True)
            log("OLLAMA", f"✓ {model} ready.")
        except subprocess.CalledProcessError as e2:
            raise RuntimeError(f"Ollama model pull failed permanently: {str(e2)}")
