import cv2
import numpy as np
from ..context import FrameContext
from ..decorators import frame_op
from typing import List, Tuple, Optional

# -----------------------------------------------------------------------------
# Transform Helpers
# -----------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders a list of 4 coordinates: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(pts: np.ndarray, maxWidth: int | None = None, maxHeight: int | None = None):
    """
    Applies a perspective transform to obtain a top-down view of the image region
    defined by the 4 points.
    
    Args:
        pts: List of 4 points [(x,y), ...].
    """
    # This needs to be a closure that accepts the frame
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute width of new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        w = int(max(widthA, widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        h = int(max(heightA, heightB))

        if maxWidth is not None:
             w = maxWidth
        if maxHeight is not None:
             h = maxHeight
        
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (w, h))
    return _op


def warp_perspective(src_pts: np.ndarray, dst_pts: np.ndarray, size: Tuple[int, int]):
    """
    General perspective warp given source and destination points.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(frame, M, size)
    return _op

def warp_affine(src_pts: np.ndarray, dst_pts: np.ndarray, size: Tuple[int, int]):
    """
    Affine warp (3 points).
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        M = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(frame, M, size)
    return _op

def apply_homography(src_pts: np.ndarray, dst_pts: np.ndarray, ransac_thresh: float = 5.0):
    """
    Computes and applies homography matrix using RANSAC.
    Note: Requires sufficient points.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        # This usually transforms the 'src' image to match 'dst' perspective if we had two images.
        # Here we just treat 'frame' as the source image to be warped.
        # If dst_pts don't define a rectangular bounds, the output size might be arbitrary.
        # We'll default to current frame size unless otherwise specified.
        # But commonly Homography maps points -> points. 
        # For image warping:
        (h, w) = frame.shape[:2]
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is None:
            return frame
        return cv2.warpPerspective(frame, H, (w, h))
    return _op
