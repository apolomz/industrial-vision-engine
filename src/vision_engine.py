"""
Industrial & Biometric Vision Engine (InspecVision AI)
--------------------------------------------------------
Unified Computer Vision module combining classical matrix analysis with
OpenCV/NumPy and intelligent inference with MediaPipe for industrial
quality auditing and gesture-based, touchless interfaces.

Built as an integration project showcasing core Computer Vision skills:
image quality auditing, structural feature extraction, and real-time
biometric gesture recognition.
"""

import time
import json
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import mediapipe as mp


class VisualFeatureExtractor:
    """Classical matrix-based analysis and feature-map extraction module."""

    @staticmethod
    def audit_lighting(bgr_image: np.ndarray) -> Tuple[bool, float]:
        """Compute the average brightness on the V channel of HSV.

        Args:
            bgr_image: Input image in BGR color space.

        Returns:
            A tuple ``(is_valid, average_brightness)`` where ``is_valid``
            indicates whether the frame's lighting falls within an
            acceptable range for downstream inspection.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        average_brightness = float(np.mean(hsv[:, :, 2]))
        is_valid = 40.0 <= average_brightness <= 220.0
        return is_valid, average_brightness

    @staticmethod
    def generate_canny_edge_map(bgr_image: np.ndarray) -> np.ndarray:
        """Run Gaussian smoothing followed by Canny edge extraction.

        Args:
            bgr_image: Input image in BGR color space.

        Returns:
            A single-channel binary edge map.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(smoothed, 50, 150)
        return edges


class MediaPipeGestureEngine:
    """Gesture inference engine for hand biometric analysis."""

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7):
        """Initialize the MediaPipe Hands solution.

        Args:
            max_hands: Maximum number of hands to detect per frame.
            detection_confidence: Minimum confidence threshold for a
                detection to be considered valid.
        """
        self._mp_hands = mp.solutions.hands
        self.hands = self._mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
        )

    def evaluate_pinch_gesture(self, bgr_image: np.ndarray) -> Dict[str, Any]:
        """Detect a hand and measure the thumb-to-index-finger distance.

        Args:
            bgr_image: Input image in BGR color space.

        Returns:
            A dictionary describing whether a hand was found, the
            classified gesture, and the normalized 3D distance between
            the thumb tip and the index fingertip.
        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return {
                "hand_detected": False,
                "gesture": "NONE",
                "normalized_distance": None,
            }

        # Use the first detected hand
        landmarks = results.multi_hand_landmarks[0].landmark
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])

        distance = float(np.linalg.norm(thumb_tip - index_tip))
        active_gesture = "PINCH_CLICK" if distance < 0.06 else "OPEN_HAND"

        return {
            "hand_detected": True,
            "gesture": active_gesture,
            "normalized_distance": round(distance, 4),
        }

    def close(self) -> None:
        """Release the underlying MediaPipe model resources."""
        self.hands.close()


class VisionInspectorPipeline:
    """Central orchestrator for inspection and structured data export."""

    def __init__(self, operator_id: str):
        """Create a new inspection pipeline.

        Args:
            operator_id: Identifier of the operator/station running the
                inspection, embedded into every generated report.
        """
        self.operator_id = operator_id
        self.extractor = VisualFeatureExtractor()
        self.gesture_engine = MediaPipeGestureEngine()

    def process_frame(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """Run the full audit and compile an executive metrics report.

        Args:
            bgr_frame: Input frame in BGR color space.

        Returns:
            A structured dictionary with metadata, lighting audit,
            gesture telemetry, and structural analysis results.
        """
        start_time = time.time()

        # 1. Quality audit
        lighting_ok, brightness = self.extractor.audit_lighting(bgr_frame)

        # 2. Gesture analysis
        gesture_result = self.gesture_engine.evaluate_pinch_gesture(bgr_frame)

        # 3. Feature-map metrics
        edges = self.extractor.generate_canny_edge_map(bgr_frame)
        edge_density = float(np.count_nonzero(edges) / edges.size) * 100.0

        latency_ms = (time.time() - start_time) * 1000.0

        report = {
            "metadata": {
                "operator": self.operator_id,
                "timestamp": time.ctime(),
                "processing_latency_ms": round(latency_ms, 2),
            },
            "image_audit": {
                "average_brightness": round(brightness, 2),
                "lighting_status": "APPROVED" if lighting_ok else "REJECTED",
            },
            "biometric_telemetry": gesture_result,
            "structural_analysis": {
                "edge_density_percentage": round(edge_density, 2),
            },
        }

        return report

    def close(self) -> None:
        """Release all resources held by the pipeline."""
        self.gesture_engine.close()


def main() -> None:
    """Run a self-contained demo using a synthetic frame."""
    print("⚡ Initializing Industrial & Biometric Vision Engine...")

    # 1. Create a synthetic test frame (camera simulation)
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(synthetic_frame, (100, 100), (300, 300), (200, 200, 200), -1)

    # 2. Instantiate the pipeline
    inspector = VisionInspectorPipeline(operator_id="Eng_Sanchez_Univalle")

    try:
        # 3. Process the frame
        final_report = inspector.process_frame(synthetic_frame)

        # 4. Persist the structured JSON report for the portfolio
        output_filename = "vision_report.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=4)

        print(f"✅ Inspection completed successfully. Report saved to '{output_filename}'.")
        print(json.dumps(final_report, indent=4))
    finally:
        inspector.close()


if __name__ == "__main__":
    main()
