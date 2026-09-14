import cv2
import numpy as np
from ..context import FrameContext


def detect_features(
    algorithm: str = "ORB", nfeatures: int = 500, metadata_key: str = "features"
):
    """
    Creates an operation to detect keypoints and compute descriptors.

    Args:
        algorithm: 'ORB', 'SIFT', or 'AKAZE'.
        nfeatures: Maximum number of features to retain.
        metadata_key: Dictionary key for output data.

    Returns:
        A callable operation that injects feature data into FrameContext.
    """
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
            "algorithm": algorithm,
        }
        return ctx

    return wrapper


def match_features(
    template_descriptors: np.ndarray,
    algorithm: str = "ORB",
    norm_type: int | None = None,
    cross_check: bool = True,
    metadata_input_key: str = "features",
    metadata_output_key: str = "matches",
):
    """
    Creates an operation that matches current frame features to a predefined template.

    Args:
        template_descriptors: Descriptor array of the target object.
        algorithm: Type of detector used for the template to select proper mathcher norms.
        norm_type: Explicit distance measurement norm if automatic is undesired.
        cross_check: If True restricts matches to robust mutual nearest neighbors.
        metadata_input_key: Where to find frame descriptors.
        metadata_output_key: Output dictionary key for match results.

    Returns:
        An executable wrapper returning a populated context.
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
        matches = sorted(matches, key=lambda x: x.distance)

        ctx.metadata[metadata_output_key] = matches
        return ctx

    return wrapper


def draw_keypoints(metadata_key: str = "features", color: tuple = (0, 255, 0)):
    """
    Creates an operation to draw detected keypoints directly on the current image.

    Args:
        metadata_key: Context dictionary key where feature dict is stored.
        color: RGB tuple to use for the drawing markers.

    Returns:
        Execution wrapper function updating ctx.frame.
    """

    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_key in ctx.metadata:
            data = ctx.metadata[metadata_key]
            kps = data["keypoints"]
            ctx.frame = cv2.drawKeypoints(ctx.frame, kps, None, color=color, flags=0)
        return ctx

    return wrapper


def draw_matches(
    template_img: np.ndarray,
    template_kps: list,
    metadata_features_key: str = "features",
    metadata_matches_key: str = "matches",
    max_matches: int = 10,
):
    """
    Creates an operation replacing the frame with a side-by-side match visualization.

    Args:
        template_img: Source template frame.
        template_kps: Saved coordinates matching the descriptors for the template.
        metadata_features_key: Origin of target image keypoints.
        metadata_matches_key: Output of previous match_features invocation.
        max_matches: Max connector lines to draw.

    Returns:
        A wrapper directly replacing ctx.frame.
    """

    def wrapper(ctx: FrameContext) -> FrameContext:
        if (
            metadata_features_key not in ctx.metadata
            or metadata_matches_key not in ctx.metadata
        ):
            return ctx

        scene_kps = ctx.metadata[metadata_features_key]["keypoints"]
        matches = ctx.metadata[metadata_matches_key]

        out_img = cv2.drawMatches(
            template_img,
            template_kps,
            ctx.frame,
            scene_kps,
            matches[:max_matches],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        ctx.frame = out_img
        return ctx

    return wrapper
