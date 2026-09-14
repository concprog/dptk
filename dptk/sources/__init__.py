from .file import FolderSource, VideoSource

__all__ = ["FolderSource", "VideoSource", "YoloSource"]


def __getattr__(name):
    if name == "YoloSource":
        from .yolo import YoloSource
        return YoloSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
