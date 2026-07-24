"""
Download the pretrained MediaPipe model assets required by this project.

MediaPipe's Tasks API (unlike the deprecated `mediapipe.solutions` API)
ships its models as separate `.task` files instead of bundling them with
the pip package, so they need to be fetched once after installation.

Usage:
    python scripts/download_models.py
"""

import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_FILENAME = "gesture_recognizer.task"


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        sys.stdout.write(f"\rDownloading {MODEL_FILENAME}... {percent}%")
        sys.stdout.flush()


def download_gesture_model() -> str:
    """Download the Gesture Recognizer model if it isn't present yet.

    Returns:
        The absolute path to the downloaded (or already-existing) model file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    destination = os.path.join(MODELS_DIR, MODEL_FILENAME)

    if os.path.isfile(destination):
        print(f"✅ Model already present at '{destination}'. Skipping download.")
        return destination

    print("⬇️  Downloading MediaPipe Gesture Recognizer model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, destination, reporthook=_report_progress)
    except Exception as exc:
        print(f"\n❌ Download failed: {exc}")
        print(f"You can also download it manually from:\n  {MODEL_URL}")
        print(f"and place it at:\n  {destination}")
        raise
    print(f"\n✅ Model saved to '{destination}'.")
    return destination


if __name__ == "__main__":
    download_gesture_model()
