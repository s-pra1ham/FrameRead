from setuptools import setup, find_packages

setup(
    name="VideoAnalyzer",
    version="1.0.0",
    description="A local, fully offline AI pipeline for comprehensive video analysis using audio transcription and visual frame inspection.",
    author="S. Pratham",
    packages=find_packages(),
    install_requires=[
        "faster-whisper>=1.0.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "ffmpeg-python>=0.2.0",
    ],
    python_requires=">=3.10",
)
