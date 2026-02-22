import pytest
import numpy as np
from dptk.context import FrameContext
from dptk.decorators import frame_op, configure
from dptk.stream import Stream

@pytest.fixture
def source_data():
    return [
        FrameContext(frame=np.zeros((10,10), dtype=np.uint8), index=0, timestamp=0.0, metadata={"val": 1}),
        FrameContext(frame=np.zeros((10,10), dtype=np.uint8), index=1, timestamp=0.1, metadata={"val": 2}),
        FrameContext(frame=np.zeros((10,10), dtype=np.uint8), index=2, timestamp=0.2, metadata={"val": 3}),
        FrameContext(frame=np.zeros((10,10), dtype=np.uint8), index=3, timestamp=0.3, metadata={"val": 4}),
    ]

def test_stream_creation(source_data):
    stream = Stream(source_data)
    results = list(stream)
    assert len(results) == 4
    assert results[0].index == 0

def test_explicit_filter(source_data):
    stream = Stream(source_data)
    filtered = stream.filter(lambda ctx: ctx.index % 2 == 0)
    results = list(filtered)
    
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
    results = list(pipeline)
    
    assert len(results) == 2
    assert results[0].metadata["val"] == 2
    assert results[1].metadata["val"] == 4

def test_chained_filters_and_transforms(source_data):
    stream = Stream(source_data)
    
    def add_tag(ctx):
        ctx.metadata["tag"] = "processed"
        return ctx

    pipeline = stream \
        .filter(lambda ctx: ctx.index < 3) \
        .pipe(add_tag) \
        .filter(lambda ctx: ctx.metadata["val"] % 2 == 0)
        
    results = list(pipeline)
    
    assert len(results) == 1
    assert results[0].index == 1
    assert results[0].metadata["tag"] == "processed"

def test_none_filtering_default(source_data):
    mixed_data = source_data + [None]
    stream = Stream(mixed_data)
    
    filtered = stream.filter() 
    results = list(filtered)
    
    assert len(results) == 4

def test_empty_stream():
    stream = Stream([])
    results = list(stream)
    assert len(results) == 0
    
    pipeline = stream.pipe(lambda x: x)
    assert len(list(pipeline)) == 0

def test_multiple_subscribers(source_data):
    stream = Stream(source_data)
    
    results1 = list(stream)
    assert len(results1) == 4
    
    results2 = list(stream)
    assert len(results2) == 4

def test_frame_op_decorator():
    @frame_op
    def make_ones(frame):
        return np.ones_like(frame)
        
    @frame_op
    def return_none_helper(frame):
        return None

    ctx = FrameContext(frame=np.zeros((10,10), dtype=np.uint8), index=0, timestamp=0.0)
    
    op1 = make_ones
    res1 = op1(ctx)
    assert np.all(res1.frame == 1)
    
    op2 = return_none_helper
    ctx.frame = np.zeros((10,10), dtype=np.uint8)
    res2 = op2(ctx)
    
    assert res2 is not None
    assert np.all(res2.frame == 0)

def test_stream_generator_source():
    def gen():
        for i in range(3):
            yield FrameContext(frame=np.zeros((1,1)), index=i, timestamp=float(i))
            
    stream = Stream(gen())
    results = list(stream)
    assert len(results) == 3
    assert results[0].index == 0
    assert results[2].index == 2
    
    assert len(list(stream)) == 0

def test_configure_injection():
    @frame_op
    def add_value(frame, value, scalar=1):
        return frame + (value * scalar)
        
    op = configure(add_value, value=10, scalar=2)
    
    ctx = FrameContext(
        frame=np.zeros((1,1), dtype=np.uint8),
        index=0,
        timestamp=0.0
    )
    
    res = op(ctx)
    assert res.frame[0,0] == 20
    
    import pytest
    with pytest.raises(TypeError):
        configure(add_value, scalar=5)
        
    with pytest.raises(TypeError):
        configure(add_value, value=1, unknown_arg=99)

