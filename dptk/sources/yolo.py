from ultralytics import YOLO
from ..context import FrameContext
from ..sources.file import VideoSource

def YoloSource(
    video_path: str, 
    model_path: str = "last.pt", 
    conf: float = 0.25
):
    """
    A source that wraps VideoSource and injects YOLO inference results 
    into the FrameContext metadata.
    """
    # 1. Load the model once
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    # 2. Create the base video source generator
    base_stream = VideoSource(video_path)
    
    # 3. Iterate and enhance
    for ctx in base_stream:
        results = model(ctx.frame, conf=conf, verbose=False)
        ctx.metadata['yolo'] = results
        yield ctx