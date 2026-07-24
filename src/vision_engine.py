"""
Industrial & Biometric Vision Engine (InspecVision AI) — v2.0
----------------------------------------------------------------
Unified Computer Vision module combining classical matrix analysis with
OpenCV/NumPy and pretrained gesture inference with MediaPipe's Tasks API
for industrial quality auditing and gesture-based, touchless interfaces.

v2.0 replaces the deprecated `mediapipe.solutions` API — which recent
MediaPipe releases (0.10.3x) ship broken on several platforms, raising
`AttributeError: module 'mediapipe' has no attribute 'solutions'`
(see google-ai-edge/mediapipe issues #6200, #6204, #6261) — with the
actively maintained `mediapipe.tasks.vision.GestureRecognizer`, which
also unlocks a pretrained 7-class gesture vocabulary for free.
"""

import csv
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from .config import InspectionConfig  # imported as part of the `src` package
except ImportError:
    from config import InspectionConfig  # run directly as a script (python src/vision_engine.py)


# --------------------------------------------------------------------------- #
# Structured report types
# --------------------------------------------------------------------------- #

@dataclass
class LightingAudit:
    average_brightness: float
    status: str  # "APPROVED" | "REJECTED"


@dataclass
class StructuralAnalysis:
    edge_density_percentage: float
    sharpness_score: float
    focus_status: str  # "SHARP" | "BLURRY"


@dataclass
class GestureReading:
    hand_detected: bool
    gesture: str
    confidence: float
    is_pinching: bool
    pinch_distance: Optional[float]


@dataclass
class InspectionReport:
    operator: str
    timestamp: str
    processing_latency_ms: float
    lighting: LightingAudit
    structural: StructuralAnalysis
    gesture: GestureReading

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report using the project's stable JSON schema."""
        return {
            "metadata": {
                "operator": self.operator,
                "timestamp": self.timestamp,
                "processing_latency_ms": self.processing_latency_ms,
            },
            "image_audit": asdict(self.lighting),
            "biometric_telemetry": asdict(self.gesture),
            "structural_analysis": asdict(self.structural),
        }


# --------------------------------------------------------------------------- #
# Classical computer vision
# --------------------------------------------------------------------------- #

class VisualFeatureExtractor:
    """Classical matrix-based analysis and feature-map extraction module."""

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()

    def audit_lighting(self, bgr_image: np.ndarray) -> LightingAudit:
        """Compute the average brightness on the V channel of HSV.

        Args:
            bgr_image: Input image in BGR color space.

        Returns:
            A `LightingAudit` with the measured brightness and an
            APPROVED/REJECTED status based on the configured range.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        average_brightness = float(np.mean(hsv[:, :, 2]))
        is_valid = self.config.min_brightness <= average_brightness <= self.config.max_brightness
        return LightingAudit(
            average_brightness=round(average_brightness, 2),
            status="APPROVED" if is_valid else "REJECTED",
        )

    def generate_canny_edge_map(self, bgr_image: np.ndarray) -> np.ndarray:
        """Run Gaussian smoothing followed by Canny edge extraction.

        Args:
            bgr_image: Input image in BGR color space.

        Returns:
            A single-channel binary edge map.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(smoothed, self.config.canny_lower, self.config.canny_upper)
        return edges

    def measure_sharpness(self, bgr_image: np.ndarray) -> float:
        """Estimate focus sharpness via the variance of the Laplacian.

        A low-variance Laplacian indicates few high-frequency edges,
        which typically corresponds to a blurry or out-of-focus frame —
        a common real-world QA check in industrial inspection lines.

        Args:
            bgr_image: Input image in BGR color space.

        Returns:
            The Laplacian variance (higher = sharper).
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def analyze_structure(self, bgr_image: np.ndarray) -> StructuralAnalysis:
        """Compute the full structural-quality readout for a frame."""
        edges = self.generate_canny_edge_map(bgr_image)
        edge_density = float(np.count_nonzero(edges) / edges.size) * 100.0
        sharpness = self.measure_sharpness(bgr_image)
        focus_status = "SHARP" if sharpness >= self.config.blur_threshold else "BLURRY"
        return StructuralAnalysis(
            edge_density_percentage=round(edge_density, 2),
            sharpness_score=round(sharpness, 2),
            focus_status=focus_status,
        )


# --------------------------------------------------------------------------- #
# Gesture recognition (MediaPipe Tasks API)
# --------------------------------------------------------------------------- #

class GestureRecognitionEngine:
    """Pretrained gesture recognition + custom biometric distance metric.

    Wraps `mediapipe.tasks.vision.GestureRecognizer`, which classifies
    one of 7 built-in gestures (Thumbs_Up, Thumbs_Down, Victory,
    Pointing_Up, Closed_Fist, Open_Palm, ILoveYou) per detected hand,
    while also computing a custom thumb-to-index pinch distance from
    the returned hand landmarks for finer-grained "click" detection.
    """

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()

        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(
                f"Gesture model not found at '{self.config.model_path}'.\n"
                "Run 'python scripts/download_models.py' first to download it."
            )

        base_options = mp_python.BaseOptions(model_asset_path=self.config.model_path)
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=self.config.num_hands,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    def recognize(self, bgr_image: np.ndarray) -> GestureReading:
        """Detect a hand, classify its gesture, and measure pinch distance.

        Args:
            bgr_image: Input image in BGR color space.

        Returns:
            A `GestureReading` describing detection status, the top
            classified gesture with its confidence, and pinch telemetry.
        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._recognizer.recognize(mp_image)

        if not result.hand_landmarks:
            return GestureReading(
                hand_detected=False,
                gesture="NONE",
                confidence=0.0,
                is_pinching=False,
                pinch_distance=None,
            )

        # Use the first detected hand
        landmarks = result.hand_landmarks[0]
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
        pinch_distance = float(np.linalg.norm(thumb_tip - index_tip))
        is_pinching = pinch_distance < self.config.pinch_distance_threshold

        gesture_name, confidence = "NONE", 0.0
        if result.gestures and result.gestures[0]:
            top_category = result.gestures[0][0]
            if top_category.score >= self.config.gesture_min_confidence:
                gesture_name = top_category.category_name or "NONE"
                confidence = float(top_category.score)

        return GestureReading(
            hand_detected=True,
            gesture=gesture_name,
            confidence=round(confidence, 4),
            is_pinching=is_pinching,
            pinch_distance=round(pinch_distance, 4),
        )

    def close(self) -> None:
        """Release the underlying MediaPipe model resources."""
        self._recognizer.close()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class VisionInspectorPipeline:
    """Central orchestrator for inspection, history tracking, and export."""

    def __init__(self, operator_id: str, config: Optional[InspectionConfig] = None):
        """Create a new inspection pipeline.

        Args:
            operator_id: Identifier of the operator/station running the
                inspection, embedded into every generated report.
            config: Optional `InspectionConfig` override. Defaults are used
                if omitted.
        """
        self.operator_id = operator_id
        self.config = config or InspectionConfig()
        self.extractor = VisualFeatureExtractor(self.config)
        self.gesture_engine = GestureRecognitionEngine(self.config)
        self.history: List[InspectionReport] = []

    def process_frame(self, bgr_frame: np.ndarray) -> InspectionReport:
        """Run the full audit and compile an executive metrics report.

        Args:
            bgr_frame: Input frame in BGR color space.

        Returns:
            A structured `InspectionReport`. Also appended to `self.history`.
        """
        start_time = time.time()

        lighting = self.extractor.audit_lighting(bgr_frame)
        structural = self.extractor.analyze_structure(bgr_frame)
        gesture = self.gesture_engine.recognize(bgr_frame)

        latency_ms = (time.time() - start_time) * 1000.0

        report = InspectionReport(
            operator=self.operator_id,
            timestamp=time.ctime(),
            processing_latency_ms=round(latency_ms, 2),
            lighting=lighting,
            structural=structural,
            gesture=gesture,
        )
        self.history.append(report)
        return report

    def export_history_csv(self, path: str) -> None:
        """Write every report processed so far to a flat CSV file.

        Useful for trend analysis (e.g. brightness drift, blur rate,
        or gesture frequency over a session) in spreadsheets or BI tools.

        Args:
            path: Destination path for the CSV file.
        """
        fieldnames = [
            "timestamp",
            "operator",
            "processing_latency_ms",
            "average_brightness",
            "lighting_status",
            "edge_density_percentage",
            "sharpness_score",
            "focus_status",
            "hand_detected",
            "gesture",
            "confidence",
            "is_pinching",
            "pinch_distance",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for report in self.history:
                row = {
                    "timestamp": report.timestamp,
                    "operator": report.operator,
                    "processing_latency_ms": report.processing_latency_ms,
                    **asdict(report.lighting),
                    **asdict(report.structural),
                    **asdict(report.gesture),
                }
                writer.writerow(row)

    def close(self) -> None:
        """Release all resources held by the pipeline."""
        self.gesture_engine.close()


def main() -> None:
    """Run a self-contained demo using a synthetic frame."""
    import json

    print("⚡ Initializing Industrial & Biometric Vision Engine...")

    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(synthetic_frame, (100, 100), (300, 300), (200, 200, 200), -1)

    try:
        inspector = VisionInspectorPipeline(operator_id="Eng_Sanchez_Univalle")
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return

    try:
        report = inspector.process_frame(synthetic_frame)
        report_dict = report.to_dict()

        output_filename = "vision_report.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4)

        print(f"✅ Inspection completed successfully. Report saved to '{output_filename}'.")
        print(json.dumps(report_dict, indent=4))
    finally:
        inspector.close()


if __name__ == "__main__":
    main()
