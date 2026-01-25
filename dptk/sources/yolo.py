from typing import Iterable
from ..context import FrameContext
from ..sources.file import VideoSource
from ultralytics import YOLO


def YoloSource(video_path: str, model_path: str = "last.pt", conf: float = 0.25) -> Iterable[FrameContext]:
    """
    A source that wraps VideoSource and injects YOLO inference results
    into the FrameContext metadata.
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    base_stream = VideoSource(video_path)

    for ctx in base_stream:
        results = model(ctx.frame, conf=conf, verbose=False)
        ctx.metadata["yolo"] = results
        yield ctx

