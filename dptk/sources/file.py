from ..context import FrameContext
from ..decorators import threaded_source
import glob
import time
import cv2
from typing import Iterable

@threaded_source

def VideoSource(path: str) -> Iterable[FrameContext]:
    """
    Streams frames from a video file into FrameContext objects.
    
    Args:
        path: The file path to the video.
        
    Yields:
        Sequential FrameContext objects containing the video frames.
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

@threaded_source
def FolderSource(path: str, recursive: bool = False) -> Iterable[FrameContext]:
    """
    Streams images from a directory into FrameContext objects.
    
    Args:
        path: The path to the directory containing images.
        recursive: Whether to search subdirectories recursively.
        
    Yields:
        Sequential FrameContext objects containing the image frames.
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