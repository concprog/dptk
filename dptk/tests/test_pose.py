import pytest
import numpy as np
import cv2
from dptk.context import FrameContext
from dptk.transforms.pose import (
    undistort,
    detect_aruco,
    estimate_pose_single_markers,
    solve_pnp,
)


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def ctx(dummy_frame):
    return FrameContext(frame=dummy_frame, index=0, timestamp=0.0)


@pytest.fixture
def calibration_data():
    camera_matrix = np.eye(3, dtype=np.float32)
    dist_coeffs = np.zeros((5,), dtype=np.float32)
    return camera_matrix, dist_coeffs


def test_undistort(ctx, calibration_data):
    cm, dc = calibration_data
    op = undistort(cm, dc)
    res = op(ctx)
    assert res.frame.shape == (480, 640, 3)


def test_detect_aruco_empty(ctx):
    op = detect_aruco(metadata_key="aruco_data")
    res = op(ctx)
    assert "aruco_data" in res.metadata
    data = res.metadata["aruco_data"]
    assert len(data["corners"]) == 0
    assert data["ids"] is None


def test_estimate_pose_skips_empty(ctx, calibration_data):
    cm, dc = calibration_data
    ctx = detect_aruco(metadata_key="aruco_data")(ctx)
    op = estimate_pose_single_markers(
        cm, dc, metadata_input_key="aruco_data", metadata_output_key="pose_data"
    )
    res = op(ctx)
    assert "pose_data" not in res.metadata


def test_solve_pnp_random(ctx, calibration_data):
    cm, dc = calibration_data
    obj_points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]],
        dtype=np.float32,
    )
    img_points = np.array(
        [[10, 10], [20, 10], [10, 20], [15, 15], [20, 20], [15, 25]], dtype=np.float32
    )

    op = solve_pnp(obj_points, img_points, cm, dc, metadata_key="pnp_res")
    res = op(ctx)

    assert "pnp_res" in res.metadata
    pnp = res.metadata["pnp_res"]
    assert "rvec" in pnp
    assert "tvec" in pnp
    assert pnp["success"] in [True, False]
