import numpy as np
import cv2
from dptk.transforms.ops import (
    remove_particles,
    remove_specks,
    _median5,
)
from dptk.decorators import configure


def _background(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 120, size=(h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (9, 9), 0)


def _window(pan=3, particle=True):
    """5 frames of a panning textured background with a bright square drifting 40 px/frame."""
    bg = _background()
    frames = []
    for i in range(5):
        f = np.roll(bg, pan * i, axis=1).copy()
        if particle:
            x = 10 + 40 * i
            cv2.rectangle(f, (x, 40), (x + 30, 70), (230, 235, 240), -1)
        frames.append(f)
    return frames


def test_median5_matches_numpy():
    rng = np.random.default_rng(1)
    arrs = [rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8) for _ in range(5)]
    expected = np.median(np.stack(arrs), axis=0).astype(np.uint8)
    assert np.array_equal(_median5(*arrs), expected)


def test_remove_particles_removes_moving_square():
    frames = _window()
    clean = _window(particle=False)[2]
    out = remove_particles.__wrapped__(frames, 2)

    region = (slice(40, 70), slice(90, 120))
    assert out.shape == frames[2].shape
    assert np.abs(out[region].astype(int) - clean[region].astype(int)).mean() < 12
    # far from the particle, pixels are untouched
    assert np.array_equal(out[:, :40], frames[2][:, :40])


def test_remove_particles_fill_modes():
    frames = _window()
    for fill in ("min", "inpaint"):
        res = remove_particles.__wrapped__(frames, 2, fill=fill)
        assert res[55, 105].max() < 160
    # configure binds window-op kwargs after (frames, centre)
    assert configure(remove_particles, fill="min").__batch__["size"] == 5


def test_remove_particles_keeps_large_and_saturated_regions():
    frames = _window(particle=False)
    for f in frames:
        cv2.rectangle(f, (20, 20), (120, 100), (240, 240, 240), -1)  # large bright
        cv2.circle(f, (140, 100), 5, (0, 0, 255), -1)  # small but saturated
    out = remove_particles.__wrapped__(frames, 2)
    assert np.array_equal(out[30:90, 30:110], frames[2][30:90, 30:110])
    assert out[100, 140].tolist() == [0, 0, 255]


def test_remove_specks_single_frame():
    frame = _background()
    cv2.circle(frame, (80, 60), 3, (240, 240, 240), -1)
    out = remove_specks.__wrapped__(frame)
    assert out[60, 80].max() < 150
    assert np.array_equal(out[:, :40], frame[:, :40])
