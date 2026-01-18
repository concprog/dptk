from typing import Callable, Optional
from .context import FrameContext
import numpy as np

def frame_op(func: Callable[[np.ndarray], Optional[np.ndarray]]) -> Callable[[FrameContext], FrameContext]:
    """
    Decorator to convert a function operating on raw numpy arrays 
    into a Transform
    
    Handles:
    1. Extracting the frame from Context.
    2. Handling in-place (None return) vs out-of-place (new array return) ops.
    3. Updating the Context.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        result = func(ctx.frame)
        
        if result is not None:
            ctx.frame = result
            
        return ctx
    return wrapper