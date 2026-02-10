from ..context import FrameContext
import glob
import time
import cv2
from typing import Iterable

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

def FolderSource(path: str, recursive: bool = False) -> Iterable[FrameContext]:
    """
    A generator that yields FrameContext objects from a folder of images.
    """
    walk = glob.glob(path + "/*.jpg", recursive=recursive)
    walk += glob.glob(path + "/*.png", recursive=recursive)
    walk += glob.glob(path + "/*.jpeg", recursive=recursive)
    for i, image in enumerate(walk):
        yield FrameContext(
            frame=cv2.imread(image),
            index=i,
            timestamp=time.time(),
            metadata={}
        )