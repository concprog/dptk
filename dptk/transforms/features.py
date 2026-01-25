import cv2
import numpy as np
from ..context import FrameContext
from typing import Optional, List, Any

# -----------------------------------------------------------------------------
# DETECTOR
# -----------------------------------------------------------------------------

def detect_features(algorithm: str = "ORB", nfeatures: int = 500, metadata_key: str = "features"):
    """
    Detects keypoints and computes descriptors.
    Algorithms available: 'ORB', 'SIFT' (if available), 'AKAZE'.
    """
    # Initialize detector once
    if algorithm == "ORB":
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif algorithm == "AKAZE":
        detector = cv2.AKAZE_create()
    elif algorithm == "SIFT":
        detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def wrapper(ctx: FrameContext) -> FrameContext:
        kps, des = detector.detectAndCompute(ctx.frame, None)
        ctx.metadata[metadata_key] = {
            "keypoints": kps,
            "descriptors": des,
            "algorithm": algorithm
        }
        return ctx
    return wrapper

# -----------------------------------------------------------------------------
# MATCHING
# -----------------------------------------------------------------------------

def match_features(
    template_descriptors: np.ndarray, 
    algorithm: str = "ORB",
    norm_type: int | None = None,
    cross_check: bool = True,
    metadata_input_key: str = "features",
    metadata_output_key: str = "matches"
):
    """
    Matches features from current frame against a provided template descriptor set.
    """
    if norm_type is None:
        if algorithm == "ORB":
            norm_type = cv2.NORM_HAMMING
        else:
            norm_type = cv2.NORM_L2
            
    matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)
    
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_input_key not in ctx.metadata:
            return ctx
            
        scene_des = ctx.metadata[metadata_input_key]["descriptors"]
        if scene_des is None or len(scene_des) == 0:
            return ctx
            
        matches = matcher.match(template_descriptors, scene_des)
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        ctx.metadata[metadata_output_key] = matches
        return ctx
    return wrapper

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

def draw_keypoints(metadata_key: str = "features", color: tuple = (0, 255, 0)):
    """
    Draws detected keypoints on the frame.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_key in ctx.metadata:
            data = ctx.metadata[metadata_key]
            kps = data["keypoints"]
            # cv2.drawKeypoints returns a new image, so we update ctx.frame
            ctx.frame = cv2.drawKeypoints(ctx.frame, kps, None, color=color, flags=0)
        return ctx
    return wrapper

def draw_matches(
    template_img: np.ndarray,
    template_kps: list,
    metadata_features_key: str = "features",
    metadata_matches_key: str = "matches",
    max_matches: int = 10
):
    """
    Draws matches between template and current frame.
    Note: This changes the frame dimension (usually side-by-side).
    This might break downstream ops expecting original resolution.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_features_key not in ctx.metadata or metadata_matches_key not in ctx.metadata:
            return ctx
            
        scene_kps = ctx.metadata[metadata_features_key]["keypoints"]
        matches = ctx.metadata[metadata_matches_key]
        
        # Draw top N matches
        out_img = cv2.drawMatches(
            template_img, template_kps,
            ctx.frame, scene_kps,
            matches[:max_matches],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        ctx.frame = out_img
        return ctx
    return wrapper
