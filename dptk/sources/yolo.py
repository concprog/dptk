from typing import Iterable
from ..context import FrameContext
from ..sources.file import VideoSource
from ultralytics import YOLO


def YoloSource(video_path: str, model_path: str = "last.pt", conf: float = 0.25):
    """
    Creates a stream generating frames from a video and annotates them with YOLO detections.
    
    Args:
        video_path: Path to the input video.
        model_path: Path to the YOLO weights.
        conf: Confidence threshold for bounding box predictions.
        
    Returns:
        A root stream yielding FrameContexts with YOLO results in their metadata.
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    base_stream = VideoSource(video_path)

    def apply_yolo(ctx: FrameContext) -> FrameContext:
        results = model(ctx.frame, conf=conf, verbose=False)
        ctx.metadata["yolo"] = results
        return ctx

    return base_stream.pipe(apply_yolo)


