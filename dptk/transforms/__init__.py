from .uie import CLAHE, grayworld, white_patch

from . import ops
from . import transforms
from . import features
from . import pose
from . import segmentation
from . import uie

__all__ = [
    "CLAHE",
    "grayworld",
    "white_patch",
    "crop_to_class",
    "draw_boxes",
    "ops",
    "transforms",
    "features",
    "pose",
    "segmentation",
    "uie",
    "yolo",
]

# ultralytics is an optional extra (dptk[ultralytics]); resolve yolo lazily.
_YOLO_NAMES = {"yolo", "crop_to_class", "draw_boxes"}


def __getattr__(name):
    if name in _YOLO_NAMES:
        from . import yolo
        return yolo if name == "yolo" else getattr(yolo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
