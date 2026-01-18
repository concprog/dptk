from typing import Iterable, Callable, Iterator
from .context import FrameContext

class Stream:
    def __init__(self, source: Iterable[FrameContext]):
        self.source = source

    def pipe(self, *ops: Callable[[FrameContext], FrameContext]) -> 'Stream':
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
    
    # Allow the Stream to be treated as an iterable directly
    def __iter__(self):
        return iter(self.source)