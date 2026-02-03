
import pytest
import numpy as np
import cv2
from dptk.context import FrameContext
from dptk.transforms.features import detect_features, match_features

@pytest.fixture
def dummy_frame():
    # Make a frame with some noise/texture to find features
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    return frame

@pytest.fixture
def ctx(dummy_frame):
    return FrameContext(frame=dummy_frame, index=0, timestamp=0.0)

def test_detect_features_orb(ctx):
    op = detect_features(algorithm="ORB", nfeatures=100, metadata_key="feat")
    res = op(ctx)
    assert "feat" in res.metadata
    
    data = res.metadata["feat"]
    # ORB might not find features in random noise depending on the seed but usually finds something
    # or at least returns empty lists
    assert "keypoints" in data
    assert "descriptors" in data
    # Type checks
    assert isinstance(data["keypoints"], (list, tuple))
    # descriptors is numpy array or None
    if data["descriptors"] is not None:
         assert isinstance(data["descriptors"], np.ndarray)

def test_match_features_skip(ctx):
    # If we provide a template but have no scene features, it should return gracefully
    # Create random descriptors for template
    template_des = np.random.randint(0, 255, (10, 32), dtype=np.uint8)
    
    op = match_features(template_des, algorithm="ORB", metadata_input_key="nothing", metadata_output_key="matches")
    res = op(ctx)
    
    # Input key missing -> returns ctx
    assert "matches" not in res.metadata

def test_match_features_logic(ctx):
    # Force some features into metadata manually
    # Fake 5 descriptors
    fake_des = np.random.randint(0, 255, (5, 32), dtype=np.uint8)
    ctx.metadata["feat"] = {"descriptors": fake_des, "keypoints": []}
    
    template_des = np.random.randint(0, 255, (5, 32), dtype=np.uint8)
    
    op = match_features(template_des, algorithm="ORB", metadata_input_key="feat", metadata_output_key="matches")
    res = op(ctx)
    
    assert "matches" in res.metadata
    matches = res.metadata["matches"]
    assert isinstance(matches, list)
