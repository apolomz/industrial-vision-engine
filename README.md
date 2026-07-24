# 🚀 Industrial & Biometric Vision Engine

**An object-oriented Python suite for image-quality auditing, structural feature extraction, and real-time biometric gesture recognition — built for touchless, high-precision environments such as biomedical labs and industrial plants.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00A98F?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

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
- [Roadmap](#-roadmap)
- [License](#-license)
- [Author](#-author)

---

## 🔎 Overview

**InspecVision AI** unifies two complementary layers of Computer Vision into a single, production-style inspection pipeline:

1. **Classical matrix analysis** (OpenCV + NumPy) for deterministic, explainable image-quality checks.
2. **Deep-learning gestural inference** (MediaPipe Hands) for touchless, biometric human-machine interaction.

The result is a lightweight but extensible engine capable of auditing a camera frame for lighting quality, extracting structural edge information, and detecting hand gestures — then compiling everything into a single, timestamped JSON report ready for logging, dashboards, or downstream automation.

---

## ✨ Key Features

- 💡 **Lighting Quality Audit** — HSV-space brightness analysis with configurable pass/fail thresholds.
- 🧩 **Structural Feature Extraction** — Gaussian-smoothed Canny edge maps with quantified edge density.
- ✋ **Touchless Gesture Recognition** — 3D Euclidean distance between thumb and index fingertip landmarks to classify `PINCH_CLICK` vs. `OPEN_HAND`.
- ⏱️ **Performance Telemetry** — Per-frame processing latency captured automatically for every report.
- 📦 **Structured JSON Reporting** — Clean, machine-readable output ready for dashboards, logging pipelines, or QA systems.
- 🎥 **Live Webcam Demo** — Real-time on-screen overlay of all metrics for an instant, tangible demo.

---

## 🏗️ System Architecture

The project follows an object-oriented design with clear separation of concerns and low coupling:

| Class | Responsibility |
|---|---|
| **`VisualFeatureExtractor`** | Classical matrix processing with OpenCV/NumPy. Performs HSV-space lighting analysis and structural edge extraction via Gaussian filtering + Canny detection. |
| **`MediaPipeGestureEngine`** | Deep-learning inference engine that processes 3D biometric keypoints and evaluates relative Euclidean distances between anatomical landmarks (thumb/index) to classify gestures. |
| **`VisionInspectorPipeline`** | Central orchestrator that synchronizes metric capture, measures processing latency, and compiles a consolidated JSON report. |

```
Frame (BGR) ──▶ VisualFeatureExtractor ──▶ lighting audit + edge density
      │
      └────────▶ MediaPipeGestureEngine ──▶ gesture classification
                          │
                          ▼
                VisionInspectorPipeline
                          │
                          ▼
                 Structured JSON Report
```

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** — image I/O, color-space conversion, filtering, edge detection
- **NumPy** — matrix operations and numerical analysis
- **MediaPipe** — pretrained hand-landmark detection model

---

## 📂 Project Structure

```
industrial-vision-engine/
├── src/
│   ├── vision_engine.py   # Core module: extractor, gesture engine, pipeline
│   └── webcam_demo.py     # Real-time webcam demo with on-screen overlay
├── examples/
│   └── sample_report.json # Example of a generated report
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9 or higher
- A webcam (optional, only required for the live demo)

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
```

---

## ▶️ Usage

### 1. Run the self-contained demo

Processes a synthetic in-memory frame and writes a JSON report to disk — no camera required:

```bash
python src/vision_engine.py
```

### 2. Run the live webcam demo

Opens your default camera and overlays lighting status, active gesture, and latency in real time. Press `q` to quit:

```bash
python src/webcam_demo.py
```

### 3. Use it as a library in your own code

```python
import cv2
from src.vision_engine import VisionInspectorPipeline

frame = cv2.imread("path/to/your/image.jpg")

inspector = VisionInspectorPipeline(operator_id="QA_Station_01")
report = inspector.process_frame(frame)
inspector.close()

print(report)
```

---

## 📊 Sample Output

```json
{
    "metadata": {
        "operator": "Eng_Sanchez_Univalle",
        "timestamp": "Fri Jul 24 09:12:03 2026",
        "processing_latency_ms": 18.42
    },
    "image_audit": {
        "average_brightness": 63.87,
        "lighting_status": "APPROVED"
    },
    "biometric_telemetry": {
        "hand_detected": true,
        "gesture": "PINCH_CLICK",
        "normalized_distance": 0.0412
    },
    "structural_analysis": {
        "edge_density_percentage": 4.31
    }
}
```

See [`examples/sample_report.json`](examples/sample_report.json) for the full file.

---

## 🗺️ Roadmap

- [ ] Add unit tests (`pytest`) for `VisualFeatureExtractor` and `MediaPipeGestureEngine`
- [ ] Support batch processing for folders of images
- [ ] Add a Streamlit/Gradio dashboard for non-technical demo access
- [ ] Extend gesture vocabulary beyond pinch/open-hand
- [ ] Export historical reports to CSV for trend analysis

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 👤 Author

Jhoan Sebastian Fernandez

Feel free to connect, open an issue, or suggest improvements via a pull request!
