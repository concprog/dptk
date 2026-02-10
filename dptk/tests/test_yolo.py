import pytest
import numpy as np
from dptk.context import FrameContext
from dptk.transforms.yolo import crop_to_class, draw_boxes
from dptk.decorators import configure

# --- Robust Mocks for PyTorch/Ultralytics objects ---

class MockTensor:
    """Emulates a PyTorch tensor's behavior relevant to yolo.py"""
    def __init__(self, data):
        self._data = np.array(data)

    def __getitem__(self, key):
        # Allows slicing and masking, returning a new MockTensor
        return MockTensor(self._data[key])
    
    def cpu(self):
        return self
        
    def numpy(self):
        return self._data
        
    def item(self):
        # Extract scalar
        return self._data.item()

    def __len__(self):
        return len(self._data)
        
    def __iter__(self):
        return iter(self._data)

class MockBox:
    def __init__(self, xyxy, cls, conf, id=None):
        # In real YOLO Results.boxes, individual attributes are tensors of shape (1, ...)
        self.xyxy = MockTensor([xyxy]) # Shape (1, 4)
        self.cls = MockTensor([cls])   # Shape (1,)
        self.conf = MockTensor([conf]) # Shape (1,)
        self.id = MockTensor([id]) if id is not None else None

class MockBoxes(list):
    """
    Emulates Results.boxes which acts as a list of Box objects,
    but also exposes batch properties (.cls, .xyxy) as tensors.
    """
    def __init__(self, boxes_data):
        super().__init__(boxes_data)
        
    @property
    def cls(self):
        if not self:
             return MockTensor([])
        # Stack all classes into (N,) array
        all_cls = [b.cls.numpy()[0] for b in self]
        return MockTensor(all_cls)

    @property
    def xyxy(self):
        if not self:
             return MockTensor([]) 
        # Stack all coords into (N, 4) array
        all_xyxy = [b.xyxy.numpy()[0] for b in self]
        return MockTensor(all_xyxy)

class MockResult:
    def __init__(self, boxes_data, names):
        self.boxes = MockBoxes(boxes_data)
        self.names = names

@pytest.fixture
def mock_yolo_ctx():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 2 boxes:
    # Class 0: "person", bbox [10, 10, 50, 50]
    # Class 1: "car", bbox [60, 60, 90, 90]
    boxes = [
        MockBox([10, 10, 50, 50], 0, 0.9, 1),
        MockBox([60, 60, 90, 90], 1, 0.8, 2)
    ]
    # Name map includes a class "dog" (ID 2) which has NO boxes
    names = {0: "person", 1: "car", 2: "dog"}
    
    results = [MockResult(boxes, names)]
    
    return FrameContext(
        frame=frame,
        index=0,
        timestamp=0.0,
        metadata={"yolo": results}
    )

# --- Tests ---

def test_crop_missing_yolo_data():
    ctx = FrameContext(
        frame=np.zeros((100,100,3), dtype=np.uint8),
        index=0, 
        timestamp=0
    )
    # Default on_missing="drop"
    op = crop_to_class(target_label="person")
    assert op(ctx) is None

def test_crop_found_label(mock_yolo_ctx):
    op = crop_to_class(target_label="person")
    res = op(mock_yolo_ctx)
    
    assert res is not None
    # Crop should be 40x40 (50-10, 50-10)
    assert res.frame.shape == (40, 40, 3)

def test_crop_found_id(mock_yolo_ctx):
    # "car" is class 1
    op = crop_to_class(target_id=1)
    res = op(mock_yolo_ctx)
    
    assert res is not None
    # Crop should be 30x30 (90-60, 90-60)
    assert res.frame.shape == (30, 30, 3)

def test_crop_resize(mock_yolo_ctx):
    op = crop_to_class(target_label="person", resize_to=(20, 20))
    res = op(mock_yolo_ctx)
    
    assert res is not None
    assert res.frame.shape == (20, 20, 3)

def test_crop_missing_label_behavior(mock_yolo_ctx):
    # Target "dog" (ID 2). It exists in names, so search_id = 2.
    # But current boxes only have IDs 0 and 1.
    # So mask will be empty.
    
    # 1. on_missing="drop"
    op_drop = crop_to_class(target_label="dog", on_missing="drop")
    assert op_drop(mock_yolo_ctx) is None
    
    # 2. on_missing="pass" (return original ctx)
    op_pass = crop_to_class(target_label="dog", on_missing="pass")
    res = op_pass(mock_yolo_ctx)
    assert res is not None
    assert res.frame.shape == (100, 100, 3) # Original

def test_draw_boxes(mock_yolo_ctx):
    op = draw_boxes()
    res = op(mock_yolo_ctx)
    
    # Check if frame is modified (boxes drawn)
    # Original was all zeros. Drawn boxes are green (0, 255, 0).
    # So some pixels should be > 0
    assert np.any(res.frame > 0)
