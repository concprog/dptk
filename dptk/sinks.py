import cv2
from typing import Callable, Iterable
from .context import FrameContext

def display(window_name: str = "Output") -> Callable[[Iterable[FrameContext]], None]:
    def consume(stream: Iterable[FrameContext]) -> None:
        for ctx in stream:
            cv2.imshow(window_name, ctx.frame)
            # Wait 1ms for UI refresh
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    return consume


def write(path: str, fps: float = 30.0) -> Callable[[Iterable[FrameContext]], None]:
    def consume(stream: Iterable[FrameContext]) -> None:
        writer = None
        
        for ctx in stream:
            if writer is None:
                h, w = ctx.frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            
            writer.write(ctx.frame)
        
        if writer:
            writer.release()
    return consume


def count() -> tuple[Callable[[Iterable[FrameContext]], None], Callable[[], int]]:
    counter = {'n': 0}
    
    def consume(stream: Iterable[FrameContext]) -> None:
        for _ in stream:
            counter['n'] += 1
            
    def get_count() -> int:
        return counter['n']
        
    return consume, get_count