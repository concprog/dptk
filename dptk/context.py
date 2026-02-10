from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass(slots=True)
class FrameContext:
    """
    container for a single frame, with metadata and index
    """
    frame: np.ndarray
    index: int
    timestamp: float
    # Dictionary for arbitrary metadata (e.g., detections, model outputs)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.frame, np.ndarray):
            raise TypeError(f"frame must be np.ndarray, got {type(self.frame)}")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")