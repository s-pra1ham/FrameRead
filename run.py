import argparse
import sys
import os
from src.analyzer import analyze

def main():
    parser = argparse.ArgumentParser(
        description="VideoAnalyzer: Create detailed text summaries of video files using local AI."
    )
    parser.add_argument(
        "video_path", 
        type=str, 
        help="Path to the video file to analyze."
    )
    parser.add_argument(
        "-p", "--prompt", 
        type=str, 
        default=None,
        help="Optional prompt/question to ask about the video."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Target video '{args.video_path}' does not exist.")
        sys.exit(1)
        
    try:
        # The analyze function internally handles all logging, orchestration, and output
        # to the terminal, and returns the structured AnalysisResult dataclass.
        result = analyze(
            video_path=args.video_path,
            prompt=args.prompt
        )
    except KeyboardInterrupt:
        print("\n\n[USER ABORT] Pipeline execution interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] Pipeline naturally failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
