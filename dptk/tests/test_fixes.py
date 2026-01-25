
import pytest
import numpy as np
from dptk.context import FrameContext
from dptk.stream import Stream
from dptk.transforms.transforms import four_point_transform
import cv2

@pytest.fixture
def dummy_frame_context():
    # Create dummy frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    return FrameContext(frame=frame)

def test_import_yolo():
    """Verify dptk.sources.yolo can be imported (requires ultralytics)"""
    try:
        from dptk.sources import yolo
    except ImportError as e:
        pytest.fail(f"Failed to import dptk.sources.yolo: {e}")

def test_four_point_transform(dummy_frame_context):
    """Verify four_point_transform respects maxWidth/maxHeight"""
    # 4 points representing a 10x10 square at top left
    pts = np.array([[0,0], [10,0], [10,10], [0,10]], dtype="float32")
    
    # Test with explicit max size
    op = four_point_transform(pts, maxWidth=50, maxHeight=50)
    
    # Run transform
    res_ctx = op(dummy_frame_context)
    
    h, w = res_ctx.frame.shape[:2]
    assert w == 50, f"Expected width 50, got {w}"
    assert h == 50, f"Expected height 50, got {h}"

def test_stream_none_handling():
    """Verify Stream.pipe drops items when an operator returns None"""
    
    def none_op(ctx):
        return None
        
    def pass_op(ctx):
        ctx.metadata['processed'] = True
        return ctx
        
    source_data = [FrameContext(frame=np.zeros((10,10))), FrameContext(frame=np.zeros((10,10)))]
    stream = Stream(source_data)
    
    # Chain: pass -> none -> pass
    # Expected: 2 items in source, both dropped by none_op. Final output count = 0.
    pipeline = stream.pipe(pass_op, none_op, pass_op)
    
    count = 0
    for _ in pipeline:
        count += 1
        
    assert count == 0, f"Expected 0 items, got {count}"

    # Verify pass-through works without none_op
    stream2 = Stream(source_data)
    pipeline2 = stream2.pipe(pass_op)
    count2 = sum(1 for _ in pipeline2)
    assert count2 == 2, f"Expected 2 items, got {count2}"
