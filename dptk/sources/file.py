from curses import meta
import cv2
from ..context import FrameContext
import time

def VideoSource(path: str) -> Iterable[FrameContext]:
    """
    A generator that yields FrameContext objects from a video file.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    try:
        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            yield FrameContext(
                frame=frame,
                index=index, 
                timestamp=time.time(),
                metadata={}
            )
            index += 1
    finally:
        cap.release()