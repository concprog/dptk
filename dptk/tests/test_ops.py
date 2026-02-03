
import pytest  # pyright: ignore[reportMissingImports]
import numpy as np
import cv2
from dptk.context import FrameContext
from dptk.transforms.ops import (
    resize, translate, rotate, rotate_bound, canny, 
    find_contours, analyze_contours
)

@pytest.fixture
def dummy_frame():
    # Create a 100x100 RGB image with a white square in the middle
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (25, 25), (75, 75), (255, 255, 255), -1)
    return frame

@pytest.fixture
def ctx(dummy_frame):
    return FrameContext(frame=dummy_frame, index=0, timestamp=0.0)

def test_resize(ctx):
    op = resize(width=50) # Maintain aspect ratio
    res = op(ctx)
    h, w = res.frame.shape[:2]
    assert w == 50
    assert h == 50 # Square input

    op2 = resize(width=200, height=100)
    res2 = op2(ctx)
    h, w = res2.frame.shape[:2]
    assert w == 200
    assert h == 100
    
def test_translate(ctx):
    op = translate(tx=10, ty=20)
    res = op(ctx)
    # Check if the square moved. Original center was (50, 50). New center should be roughly (60, 70)
    # The pixel at 50,50 should now be black (original was white)
    # The pixel at 60,70 should be white
    # The pixel at 25,25 was white (top-left of square), should now be black (background)
    assert np.all(res.frame[25, 25] == 0) 
    # The pixel at 60,70 is inside the new square, should be white
    assert np.all(res.frame[70, 60] == 255)

def test_rotate(ctx):
    op = rotate(angle=90)
    res = op(ctx)
    # Square rotated 90 degrees around center stays same visually but logic runs
    assert res.frame.shape == (100, 100, 3)
    
def test_rotate_bound(ctx):
    # Rotate 45 degrees, image size should increase to fit corners
    op = rotate_bound(angle=45)
    res = op(ctx)
    h, w = res.frame.shape[:2]
    assert h > 100 and w > 100

def test_canny(ctx):
    # FIXED: call factory first
    op = canny(threshold1=50, threshold2=150)
    res = op(ctx)
    # Canny in ops.py returns BGR 3-channel
    assert res.frame.shape[2] == 3
    # Should detect edges of the square
    assert np.any(res.frame > 0)

def test_contours_pipeline(ctx):
    # Pipeline: Canny -> Find Contours -> Analyze
    # Note: find_contours expects an image to process. In the implementation reviewed, 
    # it converts to gray. If we pass the binary edge map from Canny (which is BGR in ops.py),
    # converting to gray works fine.
    
    pipeline_step1 = canny()
    pipeline_step2 = find_contours(metadata_key="cnts")
    pipeline_step3 = analyze_contours(metadata_key="cnts", output_key="stats")
    
    ctx = pipeline_step1(ctx)
    ctx = pipeline_step2(ctx)
    ctx = pipeline_step3(ctx)
    
    assert "cnts" in ctx.metadata
    assert len(ctx.metadata["cnts"]) > 0
    
    assert "stats" in ctx.metadata
    stats = ctx.metadata["stats"]
    assert len(stats) > 0
    # Check area of the square (50x50 = 2500)
    # Canny edge might give outer/inner contours or slightly different area
    areas = [s["area"] for s in stats]
    assert any(a > 2000 for a in areas)
