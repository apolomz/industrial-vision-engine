"""
Unit tests for the classical computer-vision components of the engine.

These tests deliberately avoid `GestureRecognitionEngine`, since it
requires the downloaded `.task` model asset. `VisualFeatureExtractor`
has no external dependencies beyond OpenCV/NumPy, so it is fully
testable with synthetic in-memory frames — including in CI.
"""

import cv2
import numpy as np
import pytest

from src.config import InspectionConfig
from src.vision_engine import VisualFeatureExtractor


@pytest.fixture
def extractor() -> VisualFeatureExtractor:
    return VisualFeatureExtractor(InspectionConfig())


def test_audit_lighting_overexposed_frame_is_rejected(extractor):
    bright_frame = np.full((50, 50, 3), 255, dtype=np.uint8)
    audit = extractor.audit_lighting(bright_frame)
    assert audit.average_brightness > 220
    assert audit.status == "REJECTED"


def test_audit_lighting_underexposed_frame_is_rejected(extractor):
    dark_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    audit = extractor.audit_lighting(dark_frame)
    assert audit.average_brightness < 40
    assert audit.status == "REJECTED"


def test_audit_lighting_mid_gray_frame_is_approved(extractor):
    mid_gray_frame = np.full((50, 50, 3), 128, dtype=np.uint8)
    audit = extractor.audit_lighting(mid_gray_frame)
    assert audit.status == "APPROVED"
    assert 40.0 <= audit.average_brightness <= 220.0


def test_canny_edge_map_shape_matches_grayscale_input(extractor):
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    edges = extractor.generate_canny_edge_map(frame)
    assert edges.shape == (100, 120)


def test_sharpness_flags_flat_frame_as_blurry(extractor):
    flat_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    structural = extractor.analyze_structure(flat_frame)
    assert structural.sharpness_score == pytest.approx(0.0, abs=1e-6)
    assert structural.focus_status == "BLURRY"


def test_sharpness_ranks_high_frequency_pattern_above_flat_frame(extractor):
    striped_frame = np.zeros((100, 100), dtype=np.uint8)
    striped_frame[:, ::2] = 255  # alternating columns -> high-frequency content
    striped_bgr = cv2.cvtColor(striped_frame, cv2.COLOR_GRAY2BGR)

    flat_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    sharp_score = extractor.measure_sharpness(striped_bgr)
    flat_score = extractor.measure_sharpness(flat_frame)

    assert sharp_score > flat_score


def test_default_config_thresholds_are_consistent():
    config = InspectionConfig()
    assert config.min_brightness < config.max_brightness
    assert 0.0 < config.gesture_min_confidence <= 1.0
    assert config.pinch_distance_threshold > 0.0
