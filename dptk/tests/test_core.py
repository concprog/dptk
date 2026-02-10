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
    # Filter for even indices
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
        
    # Pipe the transform
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

    # 1. Filter index < 3 (keeps 0, 1, 2)
    # 2. Transform: add tag
    # 3. Filter val even (keeps 0 [val 1] -> drop, 1 [val 2] -> keep, 2 [val 3] -> drop)
    # Wait: 0 has val 1 (odd), 1 has val 2 (even), 2 has val 3 (odd)
    # So end result should be just index 1
    
    pipeline = stream \
        .filter(lambda ctx: ctx.index < 3) \
        .pipe(add_tag) \
        .filter(lambda ctx: ctx.metadata["val"] % 2 == 0)
        
    results = list(pipeline)
    
    assert len(results) == 1
    assert results[0].index == 1
    assert results[0].metadata["tag"] == "processed"

def test_none_filtering_default(source_data):
    # Verify that .filter() with no args removes None values if they somehow appear in source
    # Though Stream source expects FrameContext, let's explicitly inject None to test safe handling
    mixed_data = source_data + [None]
    stream = Stream(mixed_data)
    
    # filter() defaults to removing falsy/None?
    # Checking implementation: 
    # if predicate is None: if ctx is not None: yield ctx
    filtered = stream.filter() 
    results = list(filtered)
    
    assert len(results) == 4

def test_empty_stream():
    stream = Stream([])
    results = list(stream)
    assert len(results) == 0
    
    # Piping empty stream
    pipeline = stream.pipe(lambda x: x)
    assert len(list(pipeline)) == 0

def test_multiple_subscribers(source_data):
    # Stream (and source iterator logic) allows re-consumption ONLY if source is re-iterable (like list).
    # If source is a generator, it would be exhausted.
    stream = Stream(source_data)
    
    results1 = list(stream)
    assert len(results1) == 4
    
    results2 = list(stream)
    assert len(results2) == 4

def test_frame_op_decorator():
    # Verify @frame_op behavior:
    # 1. Updates frame if returns array
    # 2. Does NOT drop context if returns None (just keeps old frame)
    
    @frame_op
    def make_ones(frame):
        return np.ones_like(frame)
        
    @frame_op
    def return_none_helper(frame):
        return None

    ctx = FrameContext(frame=np.zeros((10,10), dtype=np.uint8), index=0, timestamp=0.0)
    
    # Case 1: Update
    op1 = make_ones
    res1 = op1(ctx)
    assert np.all(res1.frame == 1)
    
    # Case 2: Return None -> No change to frame, ctx NOT None
    op2 = return_none_helper
    # Reset frame
    ctx.frame = np.zeros((10,10), dtype=np.uint8)
    res2 = op2(ctx)
    
    assert res2 is not None
    assert np.all(res2.frame == 0) # Should be unchanged zeroes

def test_stream_generator_source():
    # Verify Stream works with generator input
    def gen():
        for i in range(3):
            yield FrameContext(frame=np.zeros((1,1)), index=i, timestamp=float(i))
            
    stream = Stream(gen())
    results = list(stream)
    assert len(results) == 3
    assert results[0].index == 0
    assert results[2].index == 2
    
    # Note: trying to iterate again would fail/be empty if we didn't recreate generator,
    # because 'gen()' call above created one iterator.

    # dptk Stream just holding 'self.source = source'. 
    # If source is iterator, it is one-time use. This is expected Python behavior.
    assert len(list(stream)) == 0

def test_configure_injection():
    # Test dptk.decorators.configure functionality
    
    @frame_op
    def add_value(frame, value, scalar=1):
        return frame + (value * scalar)
        
    # 1. Test correct binding
    # frame is implicitly handled, so we config 'value' and 'scalar'
    op = configure(add_value, value=10, scalar=2)
    
    ctx = FrameContext(
        frame=np.zeros((1,1), dtype=np.uint8),
        index=0,
        timestamp=0.0
    )
    
    res = op(ctx)
    assert res.frame[0,0] == 20  # 0 + (10 * 2)
    
    # 2. Test invalid binding (missing required arg)
    import pytest
    with pytest.raises(TypeError):
        # 'value' is missing
        configure(add_value, scalar=5)
        
    # 3. Test invalid binding (unknown arg)
    with pytest.raises(TypeError):
        configure(add_value, value=1, unknown_arg=99)

