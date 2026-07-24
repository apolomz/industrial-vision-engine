# 🚀 Industrial & Biometric Vision Engine

**An object-oriented Python suite for image-quality auditing, structural feature extraction, and real-time biometric gesture recognition — built for touchless, high-precision environments such as biomedical labs and industrial plants.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-00A98F?logo=google&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **v2.0** — migrated off the deprecated `mediapipe.solutions` API to the actively maintained **Tasks API**, and added pretrained gesture classification, blur detection, historical reporting, unit tests, and CI.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Sample Output](#-sample-output)
- [Testing & CI](#-testing--ci)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Author](#-author)

---

## 🔎 Overview

**InspecVision AI** unifies two complementary layers of Computer Vision into a single, production-style inspection pipeline:

1. **Classical matrix analysis** (OpenCV + NumPy) for deterministic, explainable image-quality checks — lighting, structural edges, and focus.
2. **Pretrained gesture inference** (MediaPipe Tasks — `GestureRecognizer`) for touchless, biometric human-machine interaction, layered with a custom thumb-index distance metric for fine-grained "pinch" detection.

The result is a lightweight but extensible engine capable of auditing a camera frame for lighting quality and blur, extracting structural edge information, and recognizing 7 pretrained hand gestures — then compiling everything into a timestamped, structured report ready for logging, dashboards, CSV analytics, or downstream automation.

---

## ✨ Key Features

- 💡 **Lighting Quality Audit** — HSV-space brightness analysis with configurable pass/fail thresholds.
- 🔬 **Blur / Focus Detection** — Laplacian-variance sharpness scoring, flagging out-of-focus frames (a real-world QA check on industrial lines).
- 🧩 **Structural Feature Extraction** — Gaussian-smoothed Canny edge maps with quantified edge density.
- ✋ **Pretrained Gesture Recognition** — 7-class classifier (`Thumbs_Up`, `Thumbs_Down`, `Victory`, `Pointing_Up`, `Closed_Fist`, `Open_Palm`, `ILoveYou`) via MediaPipe's Gesture Recognizer task, with per-gesture confidence.
- 🤏 **Custom Pinch Metric** — 3D Euclidean distance between thumb and index fingertip landmarks, layered on top of the pretrained classifier for a finer-grained "click" signal.
- ⏱️ **Performance Telemetry** — Per-frame processing latency captured automatically for every report.
- 📦 **Structured, Typed Reporting** — Dataclass-based reports serialized to a clean, stable JSON schema.
- 📊 **Session History + CSV Export** — Every processed frame is retained in-memory and exportable to CSV for trend analysis.
- 🎥 **Live Webcam Demo** — Real-time on-screen overlay of all metrics, with in-app CSV export.
- ✅ **Unit Tests + CI** — `pytest` coverage for the classical CV components, run automatically on every push via GitHub Actions.

---

## 🏗️ System Architecture

The project follows an object-oriented design with clear separation of concerns and low coupling:

| Module | Responsibility |
|---|---|
| **`InspectionConfig`** | Single source of truth for every tunable threshold (brightness range, Canny thresholds, blur threshold, pinch distance, gesture confidence, model path). |
| **`VisualFeatureExtractor`** | Classical matrix processing with OpenCV/NumPy: HSV-space lighting audit, Canny edge density, and Laplacian-variance sharpness/focus scoring. |
| **`GestureRecognitionEngine`** | Wraps MediaPipe's Tasks-API `GestureRecognizer` for pretrained gesture classification, plus a custom Euclidean pinch-distance metric computed from the returned hand landmarks. |
| **`VisionInspectorPipeline`** | Central orchestrator that synchronizes metric capture, measures processing latency, maintains an in-memory session history, and exports it to CSV. |

```
Frame (BGR) ──▶ VisualFeatureExtractor ──▶ lighting audit + edge density + sharpness
      │
      └────────▶ GestureRecognitionEngine ──▶ pretrained gesture + pinch distance
                          │
                          ▼
                VisionInspectorPipeline
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
         Structured JSON     Session history → CSV
```

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** — image I/O, color-space conversion, filtering, edge/blur detection
- **NumPy** — matrix operations and numerical analysis
- **MediaPipe Tasks API** — pretrained `GestureRecognizer` model
- **pytest** + **ruff** — testing and linting
- **GitHub Actions** — continuous integration

---

## 📂 Project Structure

```
industrial-vision-engine/
├── .github/workflows/ci.yml   # Lint + test on every push/PR
├── src/
│   ├── config.py               # InspectionConfig dataclass (all tunables)
│   ├── vision_engine.py        # Core module: extractor, gesture engine, pipeline
│   └── webcam_demo.py          # Real-time webcam demo with overlay + CSV export
├── scripts/
│   └── download_models.py      # Fetches the required .task model asset
├── tests/
│   └── test_vision_engine.py   # Unit tests for the classical CV components
├── examples/
│   └── sample_report.json      # Example of a generated report
├── models/                     # Downloaded .task model lives here (gitignored)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9 or higher
- A webcam (optional, only required for the live demo)
- Internet access on first run, to download the pretrained model (~30 MB)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/industrial-vision-engine.git
cd industrial-vision-engine

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the pretrained gesture-recognition model (one-time step)
python scripts/download_models.py
```

> ⚠️ Step 4 is required. MediaPipe's Tasks API ships models as separate `.task` files instead of bundling them in the pip package — skipping this step raises a clear `FileNotFoundError` telling you to run it.

---

## ▶️ Usage

### 1. Run the self-contained demo

Processes a synthetic in-memory frame and writes a JSON report to disk — no camera required:

```bash
python src/vision_engine.py
```

### 2. Run the live webcam demo

Opens your default camera and overlays lighting, focus, gesture, and latency in real time.

```bash
python src/webcam_demo.py
```

Controls: press **`q`** to quit (auto-exports session history to `session_report.csv`), or **`c`** to export at any time without quitting.

### 3. Use it as a library in your own code

```python
import cv2
from src.vision_engine import VisionInspectorPipeline

frame = cv2.imread("path/to/your/image.jpg")

inspector = VisionInspectorPipeline(operator_id="QA_Station_01")
report = inspector.process_frame(frame)
print(report.to_dict())

inspector.export_history_csv("session_report.csv")
inspector.close()
```

### 4. Tune thresholds for your environment

```python
from src.config import InspectionConfig
from src.vision_engine import VisionInspectorPipeline

config = InspectionConfig(min_brightness=60.0, blur_threshold=150.0, num_hands=2)
inspector = VisionInspectorPipeline(operator_id="QA_Station_01", config=config)
```

---

## 📊 Sample Output

```json
{
    "metadata": {
        "operator": "Eng_Sanchez_Univalle",
        "timestamp": "Fri Jul 24 09:12:03 2026",
        "processing_latency_ms": 22.87
    },
    "image_audit": {
        "average_brightness": 63.87,
        "status": "APPROVED"
    },
    "biometric_telemetry": {
        "hand_detected": true,
        "gesture": "Victory",
        "confidence": 0.9421,
        "is_pinching": false,
        "pinch_distance": 0.1873
    },
    "structural_analysis": {
        "edge_density_percentage": 4.31,
        "sharpness_score": 187.42,
        "focus_status": "SHARP"
    }
}
```

See [`examples/sample_report.json`](examples/sample_report.json) for the full file.

---

## ✅ Testing & CI

Unit tests cover `VisualFeatureExtractor` with synthetic frames (no model download required, so they run anywhere, including CI):

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check src tests
```

Every push and pull request to `main` runs the same lint + test suite via [GitHub Actions](.github/workflows/ci.yml) across Python 3.10 and 3.11.

---

## 🩹 Troubleshooting

### `AttributeError: module 'mediapipe' has no attribute 'solutions'`

If you're coming from an older version of this project (or another mediapipe-based script) and hit this error: it's not something you did wrong. Recent `mediapipe` pip releases (0.10.3x) ship the legacy `mediapipe.solutions` API broken or missing entirely on several platforms — it's a widely reported, currently open issue upstream (see `google-ai-edge/mediapipe` issues **#6200**, **#6204**, **#6261**).

**This project no longer uses that API.** Since v2.0, gesture recognition runs on MediaPipe's actively maintained **Tasks API** (`mediapipe.tasks.vision.GestureRecognizer`) instead, which sidesteps the bug entirely — and unlocks a pretrained 7-gesture classifier as a bonus. If you still see this error, make sure you're running the current `src/vision_engine.py` and not an older cached copy.

### `FileNotFoundError: Gesture model not found at 'models/gesture_recognizer.task'`

Run `python scripts/download_models.py` once after installing dependencies — see [Getting Started](#-getting-started).

---

## 🗺️ Roadmap

- [ ] Switch the webcam demo to `RunningMode.VIDEO` for temporally-smoothed tracking
- [ ] Support batch processing for folders of images
- [ ] Add a Streamlit/Gradio dashboard for non-technical demo access
- [ ] Extend the pinch metric into a full custom gesture vocabulary trained on project-specific data
- [ ] Dockerfile for a fully reproducible environment

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 👤 Author

Jhoan Sebastian Fernandez

Feel free to connect, open an issue, or suggest improvements via a pull request!
