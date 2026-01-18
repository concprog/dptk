import dataclasses
from typing import Any
import numpy as np

@dataclasses.dataclass
class FrameContext:
    """
    container for a single frame
    """
    frame: np.ndarray
    index: int
    timestamp: float
    # Dictionary for arbitrary metadata (e.g., detections, model outputs)
    metadata: dict = dataclasses.field(default_factory=dict)