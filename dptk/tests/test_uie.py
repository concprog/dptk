import pytest
import numpy as np
import cv2
from dptk.context import FrameContext
from dptk.transforms.uie import CLAHE, grayworld
from dptk.decorators import configure

@pytest.fixture
def colored_frame():
    # Create a 100x100 BGR image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Fill with some color to test changes
    img[:] = (50, 100, 150)
    return img

@pytest.fixture
def gray_frame():
    return np.zeros((100, 100), dtype=np.uint8)

def test_clahe_defaults(colored_frame):
    ctx = FrameContext(frame=colored_frame, index=0, timestamp=0.0)
    op = CLAHE
    res = op(ctx)
    
    # Should maintain shape
    assert res.frame.shape == colored_frame.shape
    assert res.frame.dtype == np.uint8
    # Content should change (CLAHE modifies L channel)
    assert not np.array_equal(res.frame, colored_frame)

def test_clahe_gray(gray_frame):
    ctx = FrameContext(frame=gray_frame, index=0, timestamp=0.0)
    op = CLAHE
    res = op(ctx)
    
    assert res.frame.shape == gray_frame.shape
    assert not np.array_equal(res.frame, gray_frame)

def test_clahe_configured(colored_frame):
    # Test configuring clipLimit
    op = configure(CLAHE, clipLimit=4.0, tileGridSize=(4,4))
    ctx = FrameContext(frame=colored_frame, index=0, timestamp=0.0)
    res = op(ctx)
    
    assert res.frame.shape == colored_frame.shape

def test_grayworld(colored_frame):
    # Grayworld alters color balance
    # Make a frame with strong color cast
    frame = np.full((100, 100, 3), (10, 10, 200), dtype=np.uint8) # Strong red in BGR? No (B, G, R) -> 200 is Red
    
    ctx = FrameContext(frame=frame, index=0, timestamp=0.0)
    op = grayworld
    res = op(ctx)
    
    assert res.frame.shape == frame.shape
    # We expect some shift in values, though exact math depends on algo
    assert not np.array_equal(res.frame, frame)

def test_grayworld_configured(colored_frame):
    op = configure(grayworld, alpha=1.5)
    ctx = FrameContext(frame=colored_frame, index=0, timestamp=0.0)
    res = op(ctx)
    assert res.frame.shape == colored_frame.shape
