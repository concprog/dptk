import cv2
import numpy as np
from typing import Tuple

from ..context import FrameContext
from ..decorators import frame_op

@frame_op
def CLAHE(frame: np.ndarray, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    """
    c = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    if len(frame.shape) == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = c.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    else:
        # Grayscale
        return c.apply(frame)

@frame_op
def grayworld(frame: np.ndarray, alpha: float = 1.1) -> np.ndarray:
    """
    Simple Gray World white balance algorithm.
    """
    # Convert to float to avoid overflow/underflow during calculation
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    
    # Adjust a and b channels based on luminance (l is channel 0)
    # Original logic: result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    
    l_channel_norm = result[:, :, 0] / 255.0
    
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * l_channel_norm * alpha)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * l_channel_norm * alpha)
    
    # Clip and convert back
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

@frame_op
def ACE(frame: np.ndarray) -> np.ndarray:
    """
    Automatic Color Equalization (Placeholder).
    """
    # Placeholder
    return frame