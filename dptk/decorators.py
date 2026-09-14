from typing import Callable, Optional, Any
import numpy as np
import copy
import functools
import inspect

from .context import FrameContext


def frame_op(
    func: Callable[..., Optional[np.ndarray]],
) -> Callable[[FrameContext], FrameContext]:
    """
    Marks a function as a core frame transformation operation.

    The decorated function should accept a numpy array as its first argument and
    operate on it using default parameters. To override defaults, use the `configure` function.

    Args:
        func: The function to be decorated.

    Returns:
        A wrapper function that accepts and returns a FrameContext.
    """

    @functools.wraps(func)
    def wrapper(ctx: FrameContext) -> FrameContext:
        result = func(ctx.frame)

        if result is not None:
            ctx.frame = result
        return ctx

    wrapper.__transform__ = True
    wrapper.__wrapped__ = func

    return wrapper


def batch_op(size: int) -> Callable[[Callable], Callable]:
    """
    Marks a function as a batch frame operation.

    The decorated function receives a list of `size` numpy arrays and returns a
    list of the same length. A `None` entry keeps the corresponding input frame.
    The runner collects `size` frames before each call and passes the remaining
    frames as a shorter list at the end of the stream.

    Args:
        size: Number of frames per call.

    Returns:
        A decorator producing a wrapper that accepts and returns a list of FrameContext.
    """

    def decorator(func: Callable[..., Optional[list]]) -> Callable:
        @functools.wraps(func)
        def wrapper(ctxs: list[FrameContext]) -> list[FrameContext]:
            _apply_batch_result(ctxs, func([c.frame for c in ctxs]))
            return ctxs

        wrapper.__transform__ = True
        wrapper.__wrapped__ = func
        wrapper.__batch__ = {"mode": "chunk", "size": size}
        return wrapper

    return decorator


def window_op(
    size: int, stride: int = 1, pad: Optional[str] = "edge"
) -> Callable[[Callable], Callable]:
    """
    Marks a function as a sliding-window frame operation.

    The decorated function receives a list of `size` numpy arrays and the index
    of the centre frame, and returns the new frame for that centre. The runner
    emits one FrameContext per `stride` input frames.

    Args:
        size: Number of frames in the window. Must be odd.
        stride: Number of input frames between two outputs.
        pad: `"edge"` repeats the first and last frame so the output has as many
            frames as the input. `None` emits only full windows.

    Returns:
        A decorator producing a wrapper that accepts a list of FrameContext and
        returns the centre FrameContext.
    """
    if size % 2 == 0:
        raise ValueError("window size must be odd")

    def decorator(func: Callable[..., Optional[np.ndarray]]) -> Callable:
        @functools.wraps(func)
        def wrapper(ctxs: list[FrameContext]) -> FrameContext:
            return _apply_window_result(
                ctxs, func([c.frame for c in ctxs], len(ctxs) // 2)
            )

        wrapper.__transform__ = True
        wrapper.__wrapped__ = func
        wrapper.__batch__ = {"mode": "window", "size": size, "stride": stride, "pad": pad}
        return wrapper

    return decorator


def _apply_batch_result(ctxs: list[FrameContext], frames: Optional[list]) -> None:
    """
    Writes the frames returned by a batch function back onto their contexts.
    """
    if frames is None:
        return
    if len(frames) != len(ctxs):
        raise ValueError(
            f"batch op returned {len(frames)} frames for {len(ctxs)} inputs"
        )
    for ctx, frame in zip(ctxs, frames):
        if frame is not None:
            ctx.frame = frame


def _apply_window_result(
    ctxs: list[FrameContext], frame: Optional[np.ndarray]
) -> FrameContext:
    """
    Returns a copy of the centre context carrying the frame returned by a window
    function. The contexts in the window are left unchanged, because later
    windows still read them.
    """
    centre = copy.copy(ctxs[len(ctxs) // 2])
    if frame is not None:
        centre.frame = frame
    return centre


def metadata_op(func: Callable[[dict], dict]) -> Callable[[FrameContext], FrameContext]:
    """
    Marks a function as a metadata transformation operation.

    Args:
        func: The function that accepts and returns a metadata dictionary.

    Returns:
        A wrapper function that accepts and returns a FrameContext.
    """

    @functools.wraps(func)
    def wrapper(ctx: FrameContext) -> FrameContext:
        result = func(ctx.metadata)
        ctx.metadata = result if result is not None else ctx.metadata
        return ctx

    wrapper.__wrapped__ = func
    return wrapper


def configure(
    op: Callable, *args: Any, **kwargs: Any
) -> Callable[[FrameContext], FrameContext]:
    """
    Binds specific configuration arguments to a decorated operation.

    Creates a new execution wrapper that applies the provided arguments
    to the original function decorated by `@frame_op` or `@metadata_op`.

    Args:
        op: The decorated function to configure.
        *args: Positional arguments to bind.
        **kwargs: Keyword arguments to bind.

    Returns:
        A new callable that executes the original function with the bound parameters.
    """
    if not hasattr(op, "__wrapped__"):
        raise TypeError(
            f"{op.__name__} is not decorated with @frame_op or @metadata_op, follow the transform docs instead"
        )

    original_func = op.__wrapped__
    sig = inspect.signature(original_func)
    batch = getattr(op, "__batch__", None)

    try:
        if batch and batch["mode"] == "window":
            sig.bind(None, None, *args, **kwargs)
        else:
            sig.bind(None, *args, **kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid configuration for {op.__name__}: {e}")

    if batch is None:

        def configured_execution(ctx: FrameContext) -> FrameContext:
            result = original_func(ctx.frame, *args, **kwargs)
            if result is not None:
                ctx.frame = result
            return ctx

    elif batch["mode"] == "chunk":

        def configured_execution(ctxs: list[FrameContext]) -> list[FrameContext]:
            _apply_batch_result(
                ctxs, original_func([c.frame for c in ctxs], *args, **kwargs)
            )
            return ctxs

    else:

        def configured_execution(ctxs: list[FrameContext]) -> FrameContext:
            return _apply_window_result(
                ctxs,
                original_func([c.frame for c in ctxs], len(ctxs) // 2, *args, **kwargs),
            )

    if batch is not None:
        configured_execution.__batch__ = batch

    config_desc = ", ".join(
        [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    )
    configured_execution.__name__ = f"{op.__name__}({config_desc})"
    configured_execution.__doc__ = (
        f"Configured instance of {op.__name__}\n\nOriginal:\n{op.__doc__ or ''}"
    )

    return configured_execution


def source(func: Callable) -> Callable:
    """
    Transforms a generator function into a multi-process Stream root node.

    The generator is executed within a background daemon thread, pushing its
    yielded items to the Stream's output queues.

    Args:
        func: The generator function to wrap.

    Returns:
        A wrapper that instantiates and returns a root Stream object.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from .stream import Stream

        def safe_generator():
            try:
                yield from func(*args, **kwargs)
            except Exception as e:
                print(f"[Source Error] {e}")

        stream = Stream(source_generator=safe_generator)
        stream._is_root = True
        return stream

    return wrapper


