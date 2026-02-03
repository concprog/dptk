import cv2
import numpy as np
from typing import Callable, Optional, Tuple, List
from ..context import FrameContext
from ..decorators import frame_op

# -----------------------------------------------------------------------------
# Edge Detection
# -----------------------------------------------------------------------------

def canny(threshold1: float = 100, threshold2: float = 200) -> Callable[[FrameContext], FrameContext]:
    """
    Applies Canny edge detection.
    
    Returns a 3-channel BGR image (converted from grayscale edges) for 
    compatibility with display sinks downstream.
    
    Args:
        threshold1: First threshold for the hysteresis procedure
        threshold2: Second threshold for the hysteresis procedure
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        # Convert back to BGR for pipeline consistency
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return _op


# -----------------------------------------------------------------------------
# Basic Transforms
# -----------------------------------------------------------------------------

def resize(width: Optional[int] = None, height: Optional[int] = None, 
           inter: int = cv2.INTER_AREA) -> Callable[[FrameContext], FrameContext]:
    """
    Resizes the image to the specified width and height.
    If only one is specified, maintains aspect ratio.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        (h, w) = frame.shape[:2]
        if width is None and height is None:
            return frame
        
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)
            
        return cv2.resize(frame, dim, interpolation=inter)
    
    return _op


def translate(tx: float, ty: float) -> Callable[[FrameContext], FrameContext]:
    """
    Translates the image by tx (horizontal) and ty (vertical) pixels.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        (h, w) = frame.shape[:2]
        return cv2.warpAffine(frame, M, (w, h))
    
    return _op


def rotate(angle: float, center: Optional[Tuple[int, int]] = None, 
           scale: float = 1.0) -> Callable[[FrameContext], FrameContext]:
    """
    Rotates the image by angle degrees around the specified center.
    If center is None, rotates around the image center.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        (h, w) = frame.shape[:2]
        c = center if center is not None else (w // 2, h // 2)
            
        M = cv2.getRotationMatrix2D(c, angle, scale)
        return cv2.warpAffine(frame, M, (w, h))
    
    return _op


def rotate_bound(angle: float) -> Callable[[FrameContext], FrameContext]:
    """
    Rotates the image by angle degrees, ensuring the entire image is visible
    (no clipping) by dynamically resizing the canvas.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        (h, w) = frame.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(frame, M, (nW, nH))
    
    return _op


# -----------------------------------------------------------------------------
# Contours & Analysis
# -----------------------------------------------------------------------------

def find_contours(
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
    metadata_key: str = "contours"
) -> Callable[[FrameContext], FrameContext]:
    """
    Finds contours and stores them in context metadata.
    Does NOT modify the frame image.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        # Fix: Check frame shape, not gray shape
        if len(ctx.frame.shape) == 3:
            gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = ctx.frame
        
        cnts, hierarchy = cv2.findContours(gray, mode, method)
        ctx.metadata[metadata_key] = cnts
        ctx.metadata[f"{metadata_key}_hierarchy"] = hierarchy
        return ctx
    
    return wrapper


def draw_contours(
    metadata_key: str = "contours",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Callable[[FrameContext], FrameContext]:
    """
    Draws contours found in metadata onto the frame.
    Modifies frame in-place.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_key in ctx.metadata:
            cnts = ctx.metadata[metadata_key]
            cv2.drawContours(ctx.frame, cnts, -1, color, thickness)
        return ctx
    
    return wrapper


def analyze_contours(
    metadata_key: str = "contours", 
    output_key: str = "shape_analysis", 
    draw: bool = True
) -> Callable[[FrameContext], FrameContext]:
    """
    Analyzes contours for moments, area, perimeter, and bounding rect.
    Stores results in metadata and optionally visualizes (centroids/bboxes)
    on the frame.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_key not in ctx.metadata:
            return ctx
            
        cnts = ctx.metadata[metadata_key]
        results = []
        
        for c in cnts:
            # Moments
            M = cv2.moments(c)
            cx, cy = 0, 0
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            # Shape properties
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            x, y, w, h = cv2.boundingRect(c)
            
            data = {
                "moments": M,
                "centroid": (cx, cy),
                "area": area,
                "perimeter": perimeter,
                "bbox": (x, y, w, h)
            }
            results.append(data)
            
            if draw:
                # Draw centroid
                cv2.circle(ctx.frame, (cx, cy), 5, (255, 255, 255), -1)
                # Draw bbox
                cv2.rectangle(ctx.frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        ctx.metadata[output_key] = results
        return ctx
    
    return wrapper