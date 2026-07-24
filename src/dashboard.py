"""
Streamlit dashboard for the Industrial & Biometric Vision Engine.

A visual front-end over `VisionInspectorPipeline` with two modes:

- 🖼️  Upload Image  — audit a single photo (works anywhere, no camera
    needed; ideal for a deployed demo or a portfolio screenshot).
- 📷  Live Webcam    — continuous local capture with real-time overlay
    (requires a webcam on the machine running Streamlit).

Both modes share the same session history, viewable as a table and
exportable to CSV or JSON directly from the sidebar.

Usage:
    pip install -r requirements-dashboard.txt
    streamlit run src/dashboard.py
"""

import json
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from .config import InspectionConfig
    from .vision_engine import VisionInspectorPipeline
except ImportError:
    from config import InspectionConfig
    from vision_engine import VisionInspectorPipeline


st.set_page_config(
    page_title="InspecVision AI",
    page_icon="🚀",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #

def _get_inspector(config: InspectionConfig) -> Optional[VisionInspectorPipeline]:
    """Create (or reuse) the pipeline, caching it across Streamlit reruns."""
    needs_new = (
        "inspector" not in st.session_state
        or st.session_state.get("inspector_config") != config
    )
    if needs_new:
        if "inspector" in st.session_state:
            st.session_state["inspector"].close()
        try:
            st.session_state["inspector"] = VisionInspectorPipeline(
                operator_id=st.session_state.get("operator_id", "Dashboard_User"),
                config=config,
            )
            st.session_state["inspector_config"] = config
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Run `python scripts/download_models.py` in your terminal, then reload.")
            return None
    return st.session_state["inspector"]


def _render_report_cards(report) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Lighting",
        report.lighting.status,
        f"{report.lighting.average_brightness:.1f} avg. brightness",
    )
    col2.metric(
        "Focus",
        report.structural.focus_status,
        f"{report.structural.sharpness_score:.0f} sharpness",
    )
    gesture_subtitle = (
        f"{report.gesture.confidence:.0%} confidence" if report.gesture.hand_detected else "no hand"
    )
    col3.metric("Gesture", report.gesture.gesture, gesture_subtitle)
    col4.metric(
        "Latency",
        f"{report.processing_latency_ms:.1f} ms",
        f"pinch: {'yes' if report.gesture.is_pinching else 'no'}",
    )


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #

st.sidebar.title("⚙️ Configuration")
operator_id = st.sidebar.text_input("Operator / Station ID", value="Dashboard_User")
st.session_state["operator_id"] = operator_id

with st.sidebar.expander("Thresholds", expanded=False):
    min_brightness = st.slider("Min brightness", 0, 255, 40)
    max_brightness = st.slider("Max brightness", 0, 255, 220)
    blur_threshold = st.slider("Blur threshold (Laplacian var.)", 0, 500, 100)
    gesture_min_confidence = st.slider("Min gesture confidence", 0.0, 1.0, 0.5)
    pinch_distance_threshold = st.slider("Pinch distance threshold", 0.01, 0.2, 0.06)

config = InspectionConfig(
    min_brightness=float(min_brightness),
    max_brightness=float(max_brightness),
    blur_threshold=float(blur_threshold),
    gesture_min_confidence=float(gesture_min_confidence),
    pinch_distance_threshold=float(pinch_distance_threshold),
)

inspector = _get_inspector(config)

st.sidebar.divider()
st.sidebar.subheader("📊 Session History")
if inspector and inspector.history:
    st.sidebar.write(f"{len(inspector.history)} frame(s) processed")
    if st.sidebar.button("Clear history"):
        inspector.history.clear()
        st.rerun()

    history_rows = [r.to_dict() for r in inspector.history]
    csv_df = pd.json_normalize(history_rows)
    st.sidebar.download_button(
        "⬇️ Download CSV",
        csv_df.to_csv(index=False),
        file_name="session_report.csv",
        mime="text/csv",
    )
    st.sidebar.download_button(
        "⬇️ Download JSON",
        json.dumps(history_rows, indent=2),
        file_name="session_report.json",
        mime="application/json",
    )
else:
    st.sidebar.write("No frames processed yet.")


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #

st.title("🚀 Industrial & Biometric Vision Engine")
st.caption("Lighting audit · focus/blur detection · pretrained gesture recognition")

if inspector is None:
    st.stop()

tab_upload, tab_webcam, tab_history = st.tabs(["🖼️ Upload Image", "📷 Live Webcam", "📄 History"])

# --- Upload Image tab ------------------------------------------------------ #
with tab_upload:
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        bgr_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        report = inspector.process_frame(bgr_frame)

        col_img, col_metrics = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_metrics:
            _render_report_cards(report)
            st.json(report.to_dict(), expanded=False)

# --- Live Webcam tab -------------------------------------------------------- #
with tab_webcam:
    st.info("Runs against your machine's default camera. Not available on remote deployments.")
    run_camera = st.toggle("Start camera", key="run_camera")
    frame_slot = st.empty()
    metrics_slot = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(config.camera_index)
        if not cap.isOpened():
            st.error(f"Could not open camera index {config.camera_index}.")
        else:
            try:
                while st.session_state.get("run_camera"):
                    ok, bgr_frame = cap.read()
                    if not ok:
                        st.error("Camera read failed.")
                        break

                    bgr_frame = cv2.flip(bgr_frame, 1)
                    report = inspector.process_frame(bgr_frame)

                    frame_slot.image(
                        cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                    with metrics_slot.container():
                        _render_report_cards(report)

                    time.sleep(0.03)
            finally:
                cap.release()

# --- History tab ------------------------------------------------------------ #
with tab_history:
    if inspector.history:
        history_rows = [r.to_dict() for r in inspector.history]
        st.dataframe(pd.json_normalize(history_rows), use_container_width=True)
    else:
        st.write("No frames processed yet — try the Upload or Webcam tabs.")
