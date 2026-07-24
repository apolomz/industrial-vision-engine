"""
Real-time webcam demo for the Industrial & Biometric Vision Engine.

Opens the default camera, runs every captured frame through the
VisionInspectorPipeline, and overlays the resulting metrics directly on
the live video feed (lighting status, active gesture, and processing
latency). Press 'q' to exit.

Usage:
    python src/webcam_demo.py
"""

import cv2

from vision_engine import VisionInspectorPipeline


def run_demo(camera_index: int = 0) -> None:
    """Launch the live inspection demo on the given camera device.

    Args:
        camera_index: Index of the camera device to open (0 = default).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    inspector = VisionInspectorPipeline(operator_id="Live_Demo_Station")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural user-facing view
            report = inspector.process_frame(frame)

            lighting = report["image_audit"]["lighting_status"]
            brightness = report["image_audit"]["average_brightness"]
            gesture = report["biometric_telemetry"]["gesture"]
            latency = report["metadata"]["processing_latency_ms"]

            overlay_color = (0, 220, 0) if lighting == "APPROVED" else (0, 0, 255)
            overlay_lines = [
                f"Lighting: {lighting} ({brightness:.1f})",
                f"Gesture: {gesture}",
                f"Latency: {latency:.1f} ms",
            ]
            for i, line in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    line,
                    (15, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    overlay_color,
                    2,
                )

            cv2.imshow("Industrial & Biometric Vision Engine - Live Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        inspector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
