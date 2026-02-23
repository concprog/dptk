import pytest
import numpy as np
from dptk.context import FrameContext
from dptk.transforms.yolo import crop_to_class, draw_boxes
from dptk.decorators import configure


class MockTensor:
    def __init__(self, data):
        self._data = np.array(data)

    def __getitem__(self, key):
        return MockTensor(self._data[key])

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def item(self):
        return self._data.item()

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class MockBox:
    def __init__(self, xyxy, cls, conf, id=None):
        self.xyxy = MockTensor([xyxy])
        self.cls = MockTensor([cls])
        self.conf = MockTensor([conf])
        self.id = MockTensor([id]) if id is not None else None


class MockBoxes(list):
    def __init__(self, boxes_data):
        super().__init__(boxes_data)

    @property
    def cls(self):
        if not self:
            return MockTensor([])
        all_cls = [b.cls.numpy()[0] for b in self]
        return MockTensor(all_cls)

    @property
    def xyxy(self):
        if not self:
            return MockTensor([])
        all_xyxy = [b.xyxy.numpy()[0] for b in self]
        return MockTensor(all_xyxy)


class MockResult:
    def __init__(self, boxes_data, names):
        self.boxes = MockBoxes(boxes_data)
        self.names = names


@pytest.fixture
def mock_yolo_ctx():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    boxes = [MockBox([10, 10, 50, 50], 0, 0.9, 1), MockBox([60, 60, 90, 90], 1, 0.8, 2)]
    names = {0: "person", 1: "car", 2: "dog"}

    results = [MockResult(boxes, names)]

    return FrameContext(frame=frame, index=0, timestamp=0.0, metadata={"yolo": results})


def test_crop_missing_yolo_data():
    ctx = FrameContext(
        frame=np.zeros((100, 100, 3), dtype=np.uint8), index=0, timestamp=0
    )
    op = crop_to_class(target_label="person")
    assert op(ctx) is None


def test_crop_found_label(mock_yolo_ctx):
    op = crop_to_class(target_label="person")
    res = op(mock_yolo_ctx)

    assert res is not None
    assert res.frame.shape == (40, 40, 3)


def test_crop_found_id(mock_yolo_ctx):
    op = crop_to_class(target_id=1)
    res = op(mock_yolo_ctx)

    assert res is not None
    assert res.frame.shape == (30, 30, 3)


def test_crop_resize(mock_yolo_ctx):
    op = crop_to_class(target_label="person", resize_to=(20, 20))
    res = op(mock_yolo_ctx)

    assert res is not None
    assert res.frame.shape == (20, 20, 3)


def test_crop_missing_label_behavior(mock_yolo_ctx):
    op_drop = crop_to_class(target_label="dog", on_missing="drop")
    assert op_drop(mock_yolo_ctx) is None

    op_pass = crop_to_class(target_label="dog", on_missing="pass")
    res = op_pass(mock_yolo_ctx)
    assert res is not None
    assert res.frame.shape == (100, 100, 3)


def test_draw_boxes(mock_yolo_ctx):
    op = draw_boxes()
    res = op(mock_yolo_ctx)
    assert np.any(res.frame > 0)
