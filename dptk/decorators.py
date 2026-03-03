from typing import Callable, Optional, Any, Iterable
import numpy as np
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

    try:
        sig.bind(None, *args, **kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid configuration for {op.__name__}: {e}")

    def configured_execution(ctx: FrameContext) -> FrameContext:
        result = original_func(ctx.frame, *args, **kwargs)
        if result is not None:
            ctx.frame = result
        return ctx

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


