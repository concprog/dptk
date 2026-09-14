from ..sources.file import VideoSource
from ..transforms.yolo import yolo_detect


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
    return VideoSource(video_path).pipe(yolo_detect(model_path, conf=conf))
