import cv2
import numpy as np
from ..context import FrameContext
from typing import List, Tuple, Optional, Callable, Dict, Any



def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Creates an ordered set of four rectangular coordinate pairings spanning from origin.
    
    Args:
        pts: Input array of shape (4, 2) containing corner coordinates.
        
    Returns:
        Ordered (top-left, top-right, bottom-right, bottom-left) coordinate sequence matrix.
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
    Constructs a transform mapping extracting a subregion of the frame view directly from its four explicit corners.
    
    Args:
        pts: Explicit literal lists definitions of point tuples.
        pts_key: Path pointer to retrieve points array.
        maxWidth: Explicit dimension structures.
        maxHeight: Explicit pixel limit definitions.
        
    Returns:
        Transform handler yielding corrected orthographic perspective context arrays bounds.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        points = pts
        if points is None and pts_key is not None:
             points = ctx.metadata.get(pts_key)
             
        if points is None:
            return ctx
            
        points = np.array(points, dtype="float32")
        
        rect = order_points(points)
        (tl, tr, br, bl) = rect
        
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
    Maps regions based on generic perspective.
    
    Args:
        src_pts: Original points boundary mappings.
        dst_pts: Target destinations maps matrices vectors.
        src_key: Variables paths references IDs.
        dst_key: Target configurations mapping string references.
        size: Fixed explicit tuple target output.
        
    Returns:
        Applied frame warp function.
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
    Generates translational and rotational alignment warps using three explicit reference point pairings.
    
    Args:
        src_pts: Base coordinate positions.
        dst_pts: Target transformation destination coordinates.
        src_key: String key for source metadata access.
        dst_key: String key for destination metadata access.
        size: Shape configuration array bounding limits values bounds identifiers bounds lengths structures objects pointers.
        
    Returns:
        Affine function mapping outputs sequences mappings variables.
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
    Computes an optimal perspective transformation correlating two planar structures using robust out-of-bounds error rejections.
    
    Args:
        src_pts: List of original coordinates.
        dst_pts: Matched translation coordinates.
        src_key: Metadata access string key for source.
        dst_key: Metadata access string key for destination.
        ransac_thresh: Error threshold boundary definition point distance for matching.
        
    Returns:
        Transformation function modifying contexts with valid warped projections.
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
             
        s_pts = np.array(s_pts)
        d_pts = np.array(d_pts)
        
        (h, w) = ctx.frame.shape[:2]
        H, _ = cv2.findHomography(s_pts, d_pts, cv2.RANSAC, ransac_thresh)
        
        if H is None:
            return ctx
            
        ctx.frame = cv2.warpPerspective(ctx.frame, H, (w, h))
        return ctx
    return wrapper
