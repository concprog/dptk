import cv2
import numpy as np
from ..context import FrameContext
from typing import List, Tuple, Optional, Callable, Dict, Any

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

def four_point_transform(
    pts: Optional[np.ndarray] = None, 
    pts_key: Optional[str] = None,
    maxWidth: Optional[int] = None, 
    maxHeight: Optional[int] = None
) -> Callable[[FrameContext], FrameContext]:
    """
    Applies a perspective transform to obtain a top-down view of the image region
    defined by the 4 points.
    
    Args:
        pts: List of 4 points [(x,y), ...].
        pts_key: Metadata key to retrieve points from if pts is None.
        maxWidth: Optional explicit width for output
        maxHeight: Optional explicit height for output
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        points = pts
        if points is None and pts_key is not None:
             points = ctx.metadata.get(pts_key)
             
        if points is None:
            return ctx
            
        points = np.array(points, dtype="float32") # Ensure array
        
        rect = order_points(points)
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
        ctx.frame = cv2.warpPerspective(ctx.frame, M, (w, h))
        return ctx
        
    return wrapper


def warp_perspective(
    src_pts: Optional[np.ndarray] = None, 
    dst_pts: Optional[np.ndarray] = None, 
    src_key: Optional[str] = None,
    dst_key: Optional[str] = None,
    size: Optional[Tuple[int, int]] = None
) -> Callable[[FrameContext], FrameContext]:
    """
    General perspective warp given source and destination points.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        s_pts = src_pts
        if s_pts is None and src_key:
            s_pts = ctx.metadata.get(src_key)
            
        d_pts = dst_pts
        if d_pts is None and dst_key:
            d_pts = ctx.metadata.get(dst_key)
            
        if s_pts is None or d_pts is None:
            return ctx
            
        s_pts = np.array(s_pts, dtype="float32")
        d_pts = np.array(d_pts, dtype="float32")
        
        out_size = size
        if out_size is None:
            out_size = (ctx.frame.shape[1], ctx.frame.shape[0])

        M = cv2.getPerspectiveTransform(s_pts, d_pts)
        ctx.frame = cv2.warpPerspective(ctx.frame, M, out_size)
        return ctx
    return wrapper

def warp_affine(
    src_pts: Optional[np.ndarray] = None, 
    dst_pts: Optional[np.ndarray] = None,
    src_key: Optional[str] = None,
    dst_key: Optional[str] = None,
    size: Optional[Tuple[int, int]] = None
) -> Callable[[FrameContext], FrameContext]:
    """
    Affine warp (3 points).
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        s_pts = src_pts
        if s_pts is None and src_key:
            s_pts = ctx.metadata.get(src_key)
            
        d_pts = dst_pts
        if d_pts is None and dst_key:
            d_pts = ctx.metadata.get(dst_key)
            
        if s_pts is None or d_pts is None:
            return ctx

        s_pts = np.array(s_pts, dtype="float32")
        d_pts = np.array(d_pts, dtype="float32")

        out_size = size
        if out_size is None:
             out_size = (ctx.frame.shape[1], ctx.frame.shape[0])

        M = cv2.getAffineTransform(s_pts, d_pts)
        ctx.frame = cv2.warpAffine(ctx.frame, M, out_size)
        return ctx
    return wrapper

def apply_homography(
    src_pts: Optional[np.ndarray] = None, 
    dst_pts: Optional[np.ndarray] = None,
    src_key: Optional[str] = None,
    dst_key: Optional[str] = None,
    ransac_thresh: float = 5.0
) -> Callable[[FrameContext], FrameContext]:
    """
    Computes and applies homography matrix using RANSAC.
    Note: Requires sufficient points.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        s_pts = src_pts
        if s_pts is None and src_key:
             s_pts = ctx.metadata.get(src_key)
             
        d_pts = dst_pts
        if d_pts is None and dst_key:
             d_pts = ctx.metadata.get(dst_key)
        
        if s_pts is None or d_pts is None:
             return ctx
             
        # s_pts/d_pts might be list of points, need numpy
        s_pts = np.array(s_pts)
        d_pts = np.array(d_pts)
        
        (h, w) = ctx.frame.shape[:2]
        H, _ = cv2.findHomography(s_pts, d_pts, cv2.RANSAC, ransac_thresh)
        
        if H is None:
            return ctx
            
        ctx.frame = cv2.warpPerspective(ctx.frame, H, (w, h))
        return ctx
    return wrapper
