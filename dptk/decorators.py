from typing import Callable, Optional, Any
import numpy as np
import functools
import inspect

from .context import FrameContext

def frame_op(func: Callable[..., Optional[np.ndarray]]) -> Callable[[FrameContext], FrameContext]:
    """
    Decorator marking a function as a frame-transform operation.
    The wrapped function strictly handles FrameContext execution using 
    default parameters only. For configuration, use configure().
    """
    @functools.wraps(func)
    def wrapper(ctx: FrameContext) -> FrameContext:
        """
        Receives FrameContext, operates on frame using defaults.
        """
        result = func(ctx.frame)
        
        if result is not None:
            ctx.frame = result
        return ctx
    
    wrapper.__wrapped__ = func
    
    return wrapper



def metadata_op(func: Callable[[dict], dict]) -> Callable[[FrameContext], FrameContext]:
    """
    """
    @functools.wraps(func)
    def wrapper(ctx: FrameContext) -> FrameContext:
        result = func(ctx.metadata)
        ctx.metadata = result if result is not None else ctx.metadata
        return ctx
    
    wrapper.__wrapped__ = func
    return wrapper

def configure(op: Callable, *args: Any, **kwargs: Any) -> Callable[[FrameContext], FrameContext]:
    """
    Binds configuration arguments to any @frame_op or @metadata_op decorated function.
    
    Returns a new operation that executes with the specified parameters.
    
    Args:
        op: The decorated function (e.g., clahe, resize)
        *args, **kwargs: Configuration arguments to inject
        
    Example:
        pipe(
            configure(clahe, clipLimit=3.0),
            configure(resize, width=640, height=480)
        )
    """
    if not hasattr(op, '__wrapped__'):
        raise TypeError(f"{op.__name__} is not decorated with @frame_op or @metadata_op")
    
    original_func = op.__wrapped__
    sig = inspect.signature(original_func)
    
    try:
        sig.bind(None, *args, **kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid configuration for {op.__name__}: {e}")
    
    def configured_execution(ctx: FrameContext) -> FrameContext:
        """Execution with bound configuration."""
        result = original_func(ctx.frame, *args, **kwargs)
        if result is not None:
            ctx.frame = result
        return ctx
    
    config_desc = ", ".join(
        [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    )
    configured_execution.__name__ = f"{op.__name__}({config_desc})"
    configured_execution.__doc__ = f"Configured instance of {op.__name__}\n\nOriginal:\n{op.__doc__ or ''}"
    
    return configured_execution