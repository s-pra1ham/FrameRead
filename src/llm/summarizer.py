"""
summarizer.py
Synthesizes the parsed text and analyzed frames into a cohesive document
using qwen3.5:9b via Ollama API. 
Optionally runs the secondary QA pass across the generated master summary.
"""
import time
import requests
from src.config import OLLAMA_HOST, SUMMARY_MODEL, SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE, QA_SYSTEM_PROMPT
from src.utils.logger import log

def generate_master_summary(transcription: str, frame_analyses: str, num_frames: int) -> str:
    """
    Feeds complete transcription and all frame analyses to the LLM to generate 
    a structured video summary using the Content Strategist prompt.
    """
    log("SUMMARY", f"Sending transcription + {num_frames} frame descriptions to {SUMMARY_MODEL}...")
    
    # ── Data Preview (diagnostic) ─────────────────────────────────────────────
    log("SUMMARY", "─── TRANSCRIPT PREVIEW (first 3 lines) ───")
    transcript_lines = transcription.strip().splitlines()
    for line in transcript_lines[:3]:
        log("SUMMARY", f"  │ {line}")
    log("SUMMARY", f"  └ ... ({len(transcript_lines)} total lines)")
    
    log("SUMMARY", "─── FRAME ANALYSIS PREVIEW (first 5 lines) ───")
    frames_lines = frame_analyses.strip().splitlines()
    for line in frames_lines[:5]:
        log("SUMMARY", f"  │ {line}")
    log("SUMMARY", f"  └ ... ({len(frames_lines)} total lines)")
    
    # Confirm which system prompt is loaded
    prompt_first_line = SUMMARY_SYSTEM_PROMPT.strip().splitlines()[0]
    log("SUMMARY", f"─── SYSTEM PROMPT CHECK ───")
    log("SUMMARY", f"  Prompt starts with: \"{prompt_first_line}\"")
    # ──────────────────────────────────────────────────────────────────────────
    
    # Build user message from template with actual pipeline data
    user_content = SUMMARY_USER_TEMPLATE.format(
        transcript=transcription,
        frames_data=frame_analyses
    )
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
