"""
Centralized configuration for the Industrial & Biometric Vision Engine.

Keeping every tunable threshold in one dataclass makes the pipeline easy
to calibrate for a specific camera, lighting rig, or QA tolerance without
touching the processing logic itself.
"""

from dataclasses import dataclass


@dataclass
class InspectionConfig:
    """Tunable thresholds and paths used across the pipeline.

    Attributes:
        min_brightness: Lower bound (V channel, 0-255) for an
            acceptable lighting condition.
        max_brightness: Upper bound (V channel, 0-255) for an
            acceptable lighting condition.
        canny_lower: Lower hysteresis threshold for Canny edge detection.
        canny_upper: Upper hysteresis threshold for Canny edge detection.
        blur_threshold: Minimum Laplacian variance for a frame to be
            considered "in focus". Values below this are flagged BLURRY.
        pinch_distance_threshold: Normalized 3D distance between thumb
            and index fingertip below which a pinch is registered.
        gesture_min_confidence: Minimum confidence score for a
            recognized gesture to be reported instead of "NONE".
        num_hands: Maximum number of hands to track per frame.
        model_path: Path to the MediaPipe Gesture Recognizer `.task`
            model asset (download via `scripts/download_models.py`).
        camera_index: Default OpenCV camera device index for the live demo.
    """

    min_brightness: float = 40.0
    max_brightness: float = 220.0
    canny_lower: int = 50
    canny_upper: int = 150
    blur_threshold: float = 100.0
    pinch_distance_threshold: float = 0.06
    gesture_min_confidence: float = 0.5
    num_hands: int = 1
    model_path: str = "models/gesture_recognizer.task"
    camera_index: int = 0
