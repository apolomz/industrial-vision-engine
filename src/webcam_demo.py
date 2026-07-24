"""
Real-time webcam demo for the Industrial & Biometric Vision Engine.

Opens the default camera, runs every captured frame through the
VisionInspectorPipeline, and overlays lighting status, focus status,
the recognized gesture (with confidence), and processing latency
directly on the live video feed.

Controls:
    q - quit and export the session history to session_report.csv
    c - export the session history to CSV at any time without quitting

Usage:
    python src/webcam_demo.py
"""

import cv2

try:
    from .vision_engine import VisionInspectorPipeline
except ImportError:
    from vision_engine import VisionInspectorPipeline


def run_demo(camera_index: int = 0, csv_output: str = "session_report.csv") -> None:
    """Launch the live inspection demo on the given camera device.

    Args:
        camera_index: Index of the camera device to open (0 = default).
        csv_output: Path where the session history CSV will be saved.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    try:
        inspector = VisionInspectorPipeline(operator_id="Live_Demo_Station")
    except FileNotFoundError as exc:
        cap.release()
        print(f"❌ {exc}")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural user-facing view
            report = inspector.process_frame(frame)

            lighting_color = (0, 220, 0) if report.lighting.status == "APPROVED" else (0, 0, 255)
            focus_color = (0, 220, 0) if report.structural.focus_status == "SHARP" else (0, 165, 255)

            lines = [
                (f"Lighting: {report.lighting.status} ({report.lighting.average_brightness:.1f})", lighting_color),
                (f"Focus: {report.structural.focus_status} ({report.structural.sharpness_score:.0f})", focus_color),
                (f"Gesture: {report.gesture.gesture} ({report.gesture.confidence:.0%})", (255, 255, 255)),
                (f"Pinch: {'YES' if report.gesture.is_pinching else 'no'}", (255, 255, 255)),
                (f"Latency: {report.processing_latency_ms:.1f} ms", (200, 200, 200)),
            ]
            for i, (line, color) in enumerate(lines):
                cv2.putText(
                    frame, line, (15, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                )
            cv2.putText(
                frame, "q: quit & export  |  c: export now",
                (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
            )

            cv2.imshow("Industrial & Biometric Vision Engine - Live Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                inspector.export_history_csv(csv_output)
                print(f"📄 Session history exported to '{csv_output}' ({len(inspector.history)} frames).")
    finally:
        if inspector.history:
            inspector.export_history_csv(csv_output)
            print(f"📄 Session history exported to '{csv_output}' ({len(inspector.history)} frames).")
        inspector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
