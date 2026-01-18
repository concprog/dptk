from typing import Iterable, Callable, Iterator
from .context import FrameContext


class Stream:
    def __init__(self, source: Iterable[FrameContext]):
        self.source = source

    def pipe(self, *ops: Callable[[FrameContext], FrameContext]) -> "Stream":
        """
        Lazily chains operations.
        Returns a NEW Stream object wrapping a generator.
        """

        def generator() -> Iterator[FrameContext]:
            for ctx in self.source:
                processed_ctx = ctx
                for op in ops:
                    processed_ctx = op(processed_ctx)
                yield processed_ctx

        return Stream(generator())

    def subscribe(self, sink: Callable[[Iterable[FrameContext]], None]) -> None:
        """
        Triggers the consumption of the stream.
        Passes the iterable to the sink function.
        """
        sink(self.source)

    def filter(self, predicate: Callable | None = None) -> "Stream":
        """
        Filters the stream.
        If predicate is None (default), it removes all items that are None.
        """

        def generator() -> Iterator[FrameContext]:
            for ctx in self.source:
                # If predicate is explicitly None, use standard Python truthiness filtering
                # Since FrameContext is an object, 'if ctx' is always True.
                # We explicitly check for None here to enable the "Functional Fail State".
                if predicate is None:
                    if ctx is not None:
                        yield ctx
                elif predicate(ctx):
                    yield ctx

        return Stream(generator())

    # Allow the Stream to be treated as an iterable directly
    def __iter__(self):
        return iter(self.source)

