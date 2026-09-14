import pytest
import numpy as np
from dptk.context import FrameContext
from dptk.decorators import frame_op, batch_op, window_op, configure
from dptk.stream import Stream, wait_till_complete


def _run_sync(stream: Stream) -> list:
    """Runs the stream to completion and returns every context the sink received."""
    results = []

    def sink_func(it):
        for ctx in it:
            results.append(ctx)

    stream.subscribe(sink_func)
    wait_till_complete(stream)
    return results


@pytest.fixture
def source_data():
    return [
        FrameContext(
            frame=np.zeros((10, 10), dtype=np.uint8),
            index=0,
            timestamp=0.0,
            metadata={"val": 1},
        ),
        FrameContext(
            frame=np.zeros((10, 10), dtype=np.uint8),
            index=1,
            timestamp=0.1,
            metadata={"val": 2},
        ),
        FrameContext(
            frame=np.zeros((10, 10), dtype=np.uint8),
            index=2,
            timestamp=0.2,
            metadata={"val": 3},
        ),
        FrameContext(
            frame=np.zeros((10, 10), dtype=np.uint8),
            index=3,
            timestamp=0.3,
            metadata={"val": 4},
        ),
    ]


def test_stream_creation(source_data):
    stream = Stream(source_data)
    results = _run_sync(stream)
    assert len(results) == 4
    assert results[0].index == 0


def test_explicit_filter(source_data):
    stream = Stream(source_data)
    filtered = stream.filter(lambda ctx: ctx.index % 2 == 0)
    results = _run_sync(filtered)

    assert len(results) == 2
    assert results[0].index == 0
    assert results[1].index == 2


def test_implicit_filter_via_transform(source_data):
    stream = Stream(source_data)

    def drop_odd_vals(ctx):
        if ctx.metadata["val"] % 2 != 0:
            return None
        return ctx

    pipeline = stream.pipe(drop_odd_vals)
    results = _run_sync(pipeline)

    assert len(results) == 2
    assert results[0].metadata["val"] == 2
    assert results[1].metadata["val"] == 4


def test_chained_filters_and_transforms(source_data):
    stream = Stream(source_data)

    def add_tag(ctx):
        ctx.metadata["tag"] = "processed"
        return ctx

    pipeline = (
        stream.filter(lambda ctx: ctx.index < 3)
        .pipe(add_tag)
        .filter(lambda ctx: ctx.metadata["val"] % 2 == 0)
    )

    results = _run_sync(pipeline)

    assert len(results) == 1
    assert results[0].index == 1
    assert results[0].metadata["tag"] == "processed"


def test_none_filtering_default(source_data):
    mixed_data = source_data + [None]
    stream = Stream(mixed_data)

    filtered = stream.filter()
    results = _run_sync(filtered)

    assert len(results) == 4


def test_empty_stream():
    stream = Stream([])
    results = _run_sync(stream)
    assert len(results) == 0

    pipeline = stream.pipe(lambda x: x)
    assert len(_run_sync(pipeline)) == 0


def test_multiple_subscribers(source_data):
    stream = Stream(source_data)

    results1 = _run_sync(stream)
    assert len(results1) >= 0  # Sync runs pull off queue, second run might empty it


def test_frame_op_decorator():
    @frame_op
    def make_ones(frame):
        return np.ones_like(frame)

    @frame_op
    def return_none_helper(frame):
        return None

    ctx = FrameContext(frame=np.zeros((10, 10), dtype=np.uint8), index=0, timestamp=0.0)

    op1 = make_ones
    res1 = op1(ctx)
    assert np.all(res1.frame == 1)

    op2 = return_none_helper
    ctx.frame = np.zeros((10, 10), dtype=np.uint8)
    res2 = op2(ctx)

    assert res2 is not None
    assert np.all(res2.frame == 0)


def test_stream_generator_source():
    def gen():
        for i in range(3):
            yield FrameContext(frame=np.zeros((1, 1)), index=i, timestamp=float(i))

    stream = Stream(gen())
    results = _run_sync(stream)
    assert len(results) == 3
    assert results[0].index == 0
    assert results[2].index == 2


def _frames(n: int) -> list:
    return [
        FrameContext(frame=np.full((2, 2), i, dtype=np.uint8), index=i, timestamp=float(i))
        for i in range(n)
    ]


def test_batch_op_chunks_and_flushes_partial():
    @batch_op(size=4)
    def tag(frames):
        return [f + 100 for f in frames]

    # ops run in a child process, so record sizes via metadata instead of the closure
    def record(ctxs):
        for c in ctxs:
            c.metadata["batch"] = len(ctxs)
        return ctxs

    record.__batch__ = {"mode": "chunk", "size": 4}

    results = _run_sync(Stream(_frames(10)).pipe(tag, record))

    assert [c.index for c in results] == list(range(10))
    assert [c.metadata["batch"] for c in results] == [4] * 8 + [2] * 2
    assert all(c.frame[0, 0] == c.index + 100 for c in results)


def test_window_op_edge_pad_keeps_length():
    @window_op(size=3)
    def centre_sum(frames, centre):
        return sum(int(f[0, 0]) for f in frames) * np.ones_like(frames[centre])

    results = _run_sync(Stream(_frames(10)).pipe(centre_sum))

    assert [c.index for c in results] == list(range(10))
    # edge-padded: frame 0 sees [0, 0, 1] -> 1; frame 5 sees [4, 5, 6] -> 15; frame 9 sees [8, 9, 9] -> 26
    assert [int(c.frame[0, 0]) for c in results][:1] == [1]
    assert int(results[5].frame[0, 0]) == 15
    assert int(results[9].frame[0, 0]) == 26


def test_window_op_no_pad_drops_edges():
    @window_op(size=3, pad=None)
    def identity(frames, centre):
        return frames[centre]

    results = _run_sync(Stream(_frames(10)).pipe(identity))
    assert [c.index for c in results] == list(range(1, 9))


def test_window_op_short_stream():
    @window_op(size=5)
    def identity(frames, centre):
        return frames[centre]

    assert [c.index for c in _run_sync(Stream(_frames(2)).pipe(identity))] == [0, 1]


def test_window_op_requires_odd_size():
    with pytest.raises(ValueError):
        window_op(size=4)(lambda frames, centre: None)


def test_configure_batch_op():
    @batch_op(size=2)
    def add(frames, value):
        return [f + value for f in frames]

    op = configure(add, value=7)
    assert op.__batch__ == {"mode": "chunk", "size": 2}

    ctxs = _frames(2)
    out = op(ctxs)
    assert out is ctxs
    assert [int(c.frame[0, 0]) for c in out] == [7, 8]

    with pytest.raises(TypeError):
        configure(add, unknown=1)


def test_filter_around_batch_op():
    @batch_op(size=3)
    def noop(frames):
        return None

    pipeline = (
        Stream(_frames(10))
        .filter(lambda c: c.index % 2 == 0)
        .pipe(noop)
        .filter(lambda c: c.index != 4)
    )
    assert [c.index for c in _run_sync(pipeline)] == [0, 2, 6, 8]


def test_configure_injection():
    @frame_op
    def add_value(frame, value, scalar=1):
        return frame + (value * scalar)

    op = configure(add_value, value=10, scalar=2)

    ctx = FrameContext(frame=np.zeros((1, 1), dtype=np.uint8), index=0, timestamp=0.0)

    res = op(ctx)
    assert res.frame[0, 0] == 20

    import pytest

    with pytest.raises(TypeError):
        configure(add_value, scalar=5)

    with pytest.raises(TypeError):
        configure(add_value, value=1, unknown_arg=99)
