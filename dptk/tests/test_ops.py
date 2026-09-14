import pytest  # pyright: ignore[reportMissingImports]
import numpy as np
import cv2
from dptk.context import FrameContext
from dptk.transforms.ops import (
    resize,
    translate,
    rotate,
    rotate_bound,
    canny,
    find_contours,
    analyze_contours,
    nlmeans_denoise,
    nlmeans_denoise_multi,
)
from dptk.decorators import configure
from dptk.stream import Stream, wait_till_complete


@pytest.fixture
def dummy_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (25, 25), (75, 75), (255, 255, 255), -1)
    return frame


@pytest.fixture
def ctx(dummy_frame):
    return FrameContext(frame=dummy_frame, index=0, timestamp=0.0)


def test_resize(ctx):
    op = resize(width=50)
    res = op(ctx)
    h, w = res.frame.shape[:2]
    assert w == 50
    assert h == 50

    op2 = resize(width=200, height=100)
    res2 = op2(ctx)
    h, w = res2.frame.shape[:2]
    assert w == 200
    assert h == 100


def test_translate(ctx):
    op = translate(tx=10, ty=20)
    res = op(ctx)
    assert np.all(res.frame[25, 25] == 0)
    assert np.all(res.frame[70, 60] == 255)


def test_rotate(ctx):
    op = rotate(angle=90)
    res = op(ctx)
    assert res.frame.shape == (100, 100, 3)


def test_rotate_bound(ctx):
    op = rotate_bound(angle=45)
    res = op(ctx)
    h, w = res.frame.shape[:2]
    assert h > 100 and w > 100


def test_canny(ctx):
    op = canny(threshold1=50, threshold2=150)
    res = op(ctx)
    assert res.frame.shape[2] == 3
    assert np.any(res.frame > 0)


def test_contours_pipeline(ctx):
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
    areas = [s["area"] for s in stats]
    assert any(a > 2000 for a in areas)


def _noisy_frames(n):
    rng = np.random.default_rng(0)
    return [
        FrameContext(
            frame=rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
            index=i,
            timestamp=float(i),
        )
        for i in range(n)
    ]


def test_nlmeans_denoise(dummy_frame):
    ctx = FrameContext(frame=dummy_frame, index=0, timestamp=0.0)
    res = configure(nlmeans_denoise, h=3, hColor=3)(ctx)
    assert res.frame.shape == dummy_frame.shape
    assert res.frame.dtype == np.uint8


def test_nlmeans_denoise_multi_keeps_frame_count():
    out = []
    stream = Stream(_noisy_frames(6)).pipe(configure(nlmeans_denoise_multi, h=3, hColor=3))
    stream.subscribe(lambda it: out.extend(it))
    wait_till_complete(stream)

    assert [c.index for c in out] == list(range(6))
    assert all(c.frame.shape == (32, 32, 3) and c.frame.dtype == np.uint8 for c in out)
