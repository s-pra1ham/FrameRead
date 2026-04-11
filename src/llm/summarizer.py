"""
summarizer.py
Synthesizes the parsed text and analyzed frames into a cohesive document
using qwen3.5:9b via Ollama API. 
Optionally runs the secondary QA pass across the generated master summary.
"""
import time
import requests
from src.config import OLLAMA_HOST, SUMMARY_MODEL, SUMMARY_SYSTEM_PROMPT, QA_SYSTEM_PROMPT
from src.utils.logger import log

def generate_master_summary(transcription: str, frame_analyses: str, num_frames: int) -> str:
    """
    Feeds complete transcription and all frame analyses to the LLM to generate 
    a highly detailed master summary.
    """
    log("SUMMARY", f"Sending transcription + {num_frames} frame descriptions to {SUMMARY_MODEL}...")
    
    # Estimate token size simply (avg 4 chars per token roughly)
    user_content = f"TRANSCRIPTION:\n{transcription}\n\nFRAME DESCRIPTIONS:\n{frame_analyses}\n"
    est_tokens = len(user_content) // 4
    log("SUMMARY", f"Input size: ~{est_tokens} tokens")
    log("SUMMARY", "Generating master summary (this may take a moment)...")
    
    start_time = time.time()
    try:
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        payload = {
            "model": SUMMARY_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": 32768 # Required explicitly to ensure large context window is utilized
            }
        }
        
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        
        summary = data.get("message", {}).get("content", "").strip()
        tokens = data.get("eval_count", 0)
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to generate master summary: {str(e)}")
        
    elapsed = time.time() - start_time
    word_count = len(summary.split())
    log("SUMMARY", f"✓ Summary generated — {word_count} words, {tokens} tokens ({elapsed:.1f}s)")
    
    return summary


def answer_with_summary(summary: str, prompt: str) -> str:
    """
    Executes the Q&A feature, passing the user's explicit prompt alongside the 
    generated master summary.
    """
    short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
    log("QA", f"User prompt detected: \"{short_prompt}\"")
    log("QA", f"Sending summary + prompt to {SUMMARY_MODEL}...")
    
    start_time = time.time()
    try:
        user_content = f"SUMMARY:\n{summary}\n\nUSER QUESTION:\n{prompt}\n"
        
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        payload = {
            "model": SUMMARY_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": 16384
            }
        }
        
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        
        answer = data.get("message", {}).get("content", "").strip()
        
    except requests.exceptions.RequestException as e:
        log("QA", f"Warning: Q&A failed to generate response: {str(e)}")
        return f"[Failed to answer prompt: {str(e)}]"
        
    elapsed = time.time() - start_time
    word_count = len(answer.split())
    log("QA", f"✓ Prompt answer generated — {word_count} words ({elapsed:.1f}s)")
    
    return answer
