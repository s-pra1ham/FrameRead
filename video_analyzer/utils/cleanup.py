"""
cleanup.py
Manages run-specific ephemeral storage.
Creates UUID-named temporary directories and aggressively cleans them up.
"""
import os
import shutil
import uuid
from video_analyzer.config import TEMP_DIR_PREFIX
from video_analyzer.utils.logger import log

class TempDirManager:
    """Context manager for temporary job directories to ensure cleanup."""
    
    def __init__(self):
        self.temp_dir = f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}"

    def __enter__(self):
        """Create the temporary directory and frames subfolder."""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "frames"), exist_ok=True)
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup logic when leaving scope."""
        log("CLEANUP", f"Removing temp directory: {self.temp_dir}")
        try:
            # Count files
            num_files = sum(len(files) for _, _, files in os.walk(self.temp_dir))
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            log("CLEANUP", f"✓ Cleaned up {num_files} files")
        except Exception as e:
            log("CLEANUP", f"Warning: Failed to clean up temp dir {self.temp_dir} fully: {str(e)}")
