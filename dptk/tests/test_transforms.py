import pytest
import numpy as np
import cv2
from dptk.context import FrameContext
from dptk.transforms.transforms import (
    four_point_transform, 
    warp_perspective, 
    warp_affine, 
    apply_homography
)

@pytest.fixture
def dummy_frame():
    # 100x100 white image
    return np.ones((100, 100, 3), dtype=np.uint8) * 255

@pytest.fixture
def ctx(dummy_frame):
    return FrameContext(frame=dummy_frame, index=0, timestamp=0.0)

def test_four_point_transform_static(ctx):
    # Draw a black square 10x10 at (0,0) to (10,10)
    cv2.rectangle(ctx.frame, (0,0), (10,10), (0,0,0), -1)
    
    pts = np.array([[0,0], [10,0], [10,10], [0,10]], dtype="float32")
    
    # Scale up to 50x50
    op = four_point_transform(pts=pts, maxWidth=50, maxHeight=50)
    ctx = op(ctx)
    
    assert ctx.frame.shape[:2] == (50, 50)
    # Check if mainly black (some interpolation artifacts might exist at edges)
    # Center pixel should be black
    assert np.all(ctx.frame[25, 25] == 0)

def test_four_point_transform_dynamic(ctx):
    # Store points in metadata
    pts = [[0,0], [20,0], [20,20], [0,20]] # 20x20 square
    ctx.metadata["roi"] = pts
    
    op = four_point_transform(pts_key="roi", maxWidth=40, maxHeight=40)
    ctx = op(ctx)
    
    assert ctx.frame.shape[:2] == (40, 40)

def test_warp_perspective_dynamic(ctx):
    src = np.float32([[0,0], [10,0], [0,10], [10,10]])
    dst = np.float32([[0,0], [20,0], [0,20], [20,20]])
    
    ctx.metadata["src_p"] = src
    ctx.metadata["dst_p"] = dst
    
    op = warp_perspective(src_key="src_p", dst_key="dst_p", size=(20, 20))
    ctx = op(ctx)
    
    assert ctx.frame.shape[:2] == (20, 20)

def test_homography_dynamic(ctx):
    # Simple shift
    src = np.float32([[0,0], [100,0], [100,100], [0,100]])
    dst = np.float32([[10,10], [110,10], [110,110], [10,110]])
    
    ctx.metadata["h_src"] = src
    ctx.metadata["h_dst"] = dst
    
    # RANSAC needs enough points
    op = apply_homography(src_key="h_src", dst_key="h_dst", ransac_thresh=5.0)
    ctx = op(ctx)
    
    # The image should be shifted. Original (0,0) is now at (10,10).
    # Since background was white, if we shift, the new areas might be black (or whatever border mode)
    # Default warpPerspective uses constant border=0 (black)
    
    # (0,0) in output should be black? 
    # Wait, we match src(0,0) -> dst(10,10).
    # So the pixel at (0,0) in source moves to (10,10) in dest.
    # The output image at (0,0) corresponds to source at (-10,-10), which is outside, so black.
    assert np.all(ctx.frame[0,0] == 0) 
