import pytest
import numpy as np
import cv2
from dptk.context import FrameContext
from dptk.transforms.transforms import (
    four_point_transform,
    warp_perspective,
    warp_affine,
    apply_homography,
)


@pytest.fixture
def dummy_frame():
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


@pytest.fixture
def ctx(dummy_frame):
    return FrameContext(frame=dummy_frame, index=0, timestamp=0.0)


def test_four_point_transform_static(ctx):
    cv2.rectangle(ctx.frame, (0, 0), (10, 10), (0, 0, 0), -1)
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype="float32")
    op = four_point_transform(pts=pts, maxWidth=50, maxHeight=50)
    ctx = op(ctx)
    assert ctx.frame.shape[:2] == (50, 50)
    assert np.all(ctx.frame[25, 25] == 0)


def test_four_point_transform_dynamic(ctx):
    pts = [[0, 0], [20, 0], [20, 20], [0, 20]]
    ctx.metadata["roi"] = pts

    op = four_point_transform(pts_key="roi", maxWidth=40, maxHeight=40)
    ctx = op(ctx)

    assert ctx.frame.shape[:2] == (40, 40)


def test_warp_perspective_dynamic(ctx):
    src = np.float32([[0, 0], [10, 0], [0, 10], [10, 10]])
    dst = np.float32([[0, 0], [20, 0], [0, 20], [20, 20]])

    ctx.metadata["src_p"] = src
    ctx.metadata["dst_p"] = dst

    op = warp_perspective(src_key="src_p", dst_key="dst_p", size=(20, 20))
    ctx = op(ctx)

    assert ctx.frame.shape[:2] == (20, 20)


def test_homography_dynamic(ctx):
    src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    dst = np.float32([[10, 10], [110, 10], [110, 110], [10, 110]])

    ctx.metadata["h_src"] = src
    ctx.metadata["h_dst"] = dst

    op = apply_homography(src_key="h_src", dst_key="h_dst", ransac_thresh=5.0)
    ctx = op(ctx)
    assert np.all(ctx.frame[0, 0] == 0)
