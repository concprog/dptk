from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(slots=True)
class FrameContext:
    """
    Data container representing a single video frame and its associated metadata.

    Attributes:
        frame: The image data as a numpy array.
        index: The sequential number of the frame in the stream.
        timestamp: The time the frame was captured or processed.
        metadata: Store for arbitrary data appended by transforms or models.
    """

    frame: np.ndarray
    index: int
    timestamp: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.frame, np.ndarray):
            raise TypeError(f"frame must be np.ndarray, got {type(self.frame)}")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")



