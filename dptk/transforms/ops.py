import cv2
import numpy as np
from typing import Callable, Optional, Tuple, List
from ..context import FrameContext
from ..decorators import frame_op

# -----------------------------------------------------------------------------
# Edge Detection
# -----------------------------------------------------------------------------

@frame_op
def canny(threshold1: float = 100, threshold2: float = 200) -> Callable[[np.ndarray], np.ndarray]:
    """
    Applies Canny edge detection. Returns a single-channel gray image (or maybe 3-channel for visualization?).
    Usually pipeline expects BGR, so we might want to convert back to BGR to keep channel consistency
    if downstream ops expect it. However, Canny returns grayscale.
    Let's return 3-channel BGR for compatibility with display sinks.
    """
    def _op(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        # Convert back to BGR for display compatibility
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return _op

# -----------------------------------------------------------------------------
# Basic Transforms
# -----------------------------------------------------------------------------

def resize(width: int | None = None, height: int | None = None, inter: int = cv2.INTER_AREA):
    """
    Resizes the image to the specified width and height.
    If only one is specific, maintains aspect ratio.
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


def translate(tx: float, ty: float):
    """
    Translates the image by tx and ty.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        (h, w) = frame.shape[:2]
        return cv2.warpAffine(frame, M, (w, h))
    return _op


def rotate(angle: float, center: Tuple[int, int] | None = None, scale: float = 1.0):
    """
    Rotates the image by angle degrees.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        (h, w) = frame.shape[:2]
        c = center
        if c is None:
            c = (w // 2, h // 2)
            
        M = cv2.getRotationMatrix2D(c, angle, scale)
        return cv2.warpAffine(frame, M, (w, h))
    return _op


def rotate_bound(angle: float):
    """
    Rotates the image by angle degrees, ensuring the entire image is visible
    (no clipping) by resizing the canvas.
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
):
    """
    Finds contours and stores them in context metadata.
    Does NOT modify the frame image.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        # Usually need thresholding or canny before finding contours
        # This assumes input might be suitable, or we do a basic binary version?
        # Let's do a basic Otsu threshold to be safe for a generic 'find_contours'
        # Or expect user to pipe `canny` before? canny returns BGR so we convert to gray.
        # But `cv2.findContours` works on binary images. 
        # If user did Canny -> Gray -> BGR (in my op above) -> Gray (here), it works.
        
        # We'll check if it's already binary/gray or not?
        # Ideally, user pipes: source -> canny/threshold -> find_contours
        
        # Let's perform a simple auto-check or just simple threshold if not edges
        # But simplest reference implementation usually assumes prepared image.
        # However, to be robust, let's just run findContours on single channel.
        
        if len(gray.shape) > 2:
             gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        else:
             gray = ctx.frame.copy() 
             
        # Often folks want to find contours on EDGE or THRESHOLD output.
        # If the input frame is already "edges" (from canny op), it's good.
        # If it's a raw photo, this will be garbage.
        # Implicit assumption: User puts this after an edge/thresh op. 
        # OR we can add a 'preprocess' arg.
        
        cnts, hierarchy = cv2.findContours(gray, mode, method)
        ctx.metadata[metadata_key] = cnts
        ctx.metadata[f"{metadata_key}_hierarchy"] = hierarchy
        return ctx
    return wrapper

def draw_contours(
    metadata_key: str = "contours",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
):
    """
    Draws contours found in metadata onto the frame.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_key in ctx.metadata:
            cnts = ctx.metadata[metadata_key]
            cv2.drawContours(ctx.frame, cnts, -1, color, thickness)
        return ctx
    return wrapper

def analyze_contours(metadata_key: str = "contours", output_key: str = "shape_analysis", draw: bool = True):
    """
    Analyzes contours for moments, area, perimeter, and bounding rect.
    Stores results in metadata and optionally visualizes (centroids/bboxes).
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
