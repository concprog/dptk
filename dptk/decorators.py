from typing import Callable, Optional
from .context import FrameContext
import numpy as np

def frame_op(func: Callable[[np.ndarray], Optional[np.ndarray]]) -> Callable[[FrameContext], FrameContext]:
    """
    Decorator to convert a function operating on raw numpy arrays 
    into a Transform
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        result = func(ctx.frame)
        
        if result is not None:
            ctx.frame = result
            
        return ctx
    return wrapper

def metadata_op(func: Callable[[dict], dict]) -> Callable[[FrameContext], FrameContext]:
    """
    Decorator to convert a function operating on raw numpy arrays 
    into a Transform
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        ctx.metadata = func(ctx.metadata)
        return ctx
    return wrapper