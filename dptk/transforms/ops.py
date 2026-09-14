import cv2
import numpy as np
from typing import Callable, Optional, Tuple
from ..context import FrameContext
from ..decorators import frame_op, window_op


def canny(
    threshold1: float = 100, threshold2: float = 200
) -> Callable[[FrameContext], FrameContext]:
    """
    Applies Canny edge detection.

    Converts to grayscale for processing before restoring an analogous
    3-channel BGR format for compatibility.

    Args:
        threshold1: First threshold for the hysteresis procedure.
        threshold2: Second threshold for the hysteresis procedure.

    Returns:
        Transformed edge-map context.
    """

    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        edges = cv2.Canny(gray, threshold1, threshold2)

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return _op


def resize(
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter: int = cv2.INTER_AREA,
) -> Callable[[FrameContext], FrameContext]:
    """
    Resizes the image to specific dimensions.

    Args:
        width: Optional forced width (calculates from height if None).
        height: Optional forced height (calculates from width if None).
        inter: CV2 interpolation identifier.

    Returns:
        Frame op returning resized image.
    """

    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
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
    Translates image laterally.

    Args:
        tx: Horizontal pixel offset.
        ty: Vertical pixel offset.

    Returns:
        Op outputting warped affine.
    """

    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = frame.shape[:2]
        return cv2.warpAffine(frame, M, (w, h))

    return _op


def rotate(
    angle: float, center: Optional[Tuple[int, int]] = None, scale: float = 1.0
) -> Callable[[FrameContext], FrameContext]:
    """
    Rotates image contents fixed to canvas dimensions.

    Args:
        angle: Global angle measure.
        center: Pivot of the rotation.
        scale: Scaling coefficient.

    Returns:
        Warped rotational operation.
    """

    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        c = center if center is not None else (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(c, angle, scale)
        return cv2.warpAffine(frame, M, (w, h))

    return _op


def rotate_bound(angle: float) -> Callable[[FrameContext], FrameContext]:
    """
    Rotates an image dynamically resizing bounds to avert clipping.

    Args:
        angle: Global angle measure.

    Returns:
        Correctly expanded affine matrix frame context.
    """

    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cX, cY = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(frame, M, (nW, nH))

    return _op


@frame_op
def normalize(frame: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(frame)
    cv2.normalize(r, r, 1, 255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g, g, 1, 255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(b, b, 1, 255, norm_type=cv2.NORM_MINMAX)
    frame = cv2.merge([r, g, b])
    return frame


@frame_op
def normalize_intensity(frame: np.ndarray) -> np.ndarray:
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))
    cv2.normalize(s, s, 1, 225, norm_type=cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)


@frame_op
def normalize_a(frame: np.ndarray) -> np.ndarray:
    L, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_RGB2Lab))
    cv2.normalize(a, a, 1, 255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(b, b, 1, 255, norm_type=cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_Lab2RGB)


@frame_op
def dilate_erode(
    frame: np.ndarray, kernel_size: int = 7, iterations: int = 1
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=iterations)
    frame = cv2.erode(frame, kernel, iterations=iterations)
    return frame


@frame_op
def gblur(frame: np.ndarray) -> np.ndarray:
    cv2.GaussianBlur(frame, (7, 7), 0, frame)
    return frame


@frame_op
def remove_color_cast(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h_new = (h + 90) % 180
    hsv_new = cv2.merge([h_new, s, v])
    rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    ave_color = cv2.mean(rgb_new)[0:3]
    color_img = np.full_like(frame, ave_color)
    blend = cv2.addWeighted(frame, 0.6, color_img, 0.4, 0)
    return blend


@frame_op
def color_transfer_frame(frame: np.ndarray, preserve_paper: bool = False) -> np.ndarray:
    """
    Applies color transfer to a single frame using hardcoded ImageNet statistics.

    Parameters:
    ----------
    frame : np.ndarray
        Input image in RGB format.
    preserve_paper : bool, optional
        If True, uses the scaling factor from the Reinhard et al. paper.
        If False, uses the reciprocal factor for potentially different aesthetics.

    Returns:
    -------
    np.ndarray
        The color-transferred image in RGB format (uint8).
    """
    lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc = (
        160.199,
        26.70,
        127.60,
        9.023,
        131.327,
        15.125,
    )

    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab).astype("float32")
    (L, a, b) = cv2.split(lab)
    # (lMeanTar, lStdTar) = (L.mean(), L.std())
    # (aMeanTar, aStdTar) = (a.mean(), a.std())
    # (bMeanTar, bStdTar) = (b.mean(), b.std())

    # L = cv2.dilate(L, np.ones((3,3)))
    L = cv2.GaussianBlur(L, (5, 5), 0)
    (lMeanTar, lStdTar) = cv2.meanStdDev(L)
    (aMeanTar, aStdTar) = cv2.meanStdDev(a)
    (bMeanTar, bStdTar) = cv2.meanStdDev(b)

    # Subtract target means
    L -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # Scale by standard deviations
    if preserve_paper:
        # Paper method: scale by (target_std / source_std)
        L = (lStdTar / lStdSrc) * L
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
    else:
        # Reciprocal method: scale by (source_std / target_std)
        L = (lStdSrc / lStdTar) * L
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

    L += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    L = np.clip(L, 0, 255)
    L = cv2.GaussianBlur(L, (5, 5), 0)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([L, a, b])
    transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_Lab2RGB)

    return transfer


@frame_op
def nlmeans_denoise(
    frame: np.ndarray,
    h: float = 5,
    hColor: float = 10,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
) -> np.ndarray:
    """
    Non-local means denoising for colour frames (cv2.fastNlMeansDenoisingColored).

    Args:
        h: Filter strength for the luminance component.
        hColor: Filter strength for the colour components.
        templateWindowSize: Odd patch size used to compute weights.
        searchWindowSize: Odd window size for the weighted average; cost scales linearly.
    """
    return cv2.fastNlMeansDenoisingColored(
        frame, None, h, hColor, templateWindowSize, searchWindowSize
    )


@window_op(size=3)
def nlmeans_denoise_multi(
    frames: list,
    centre: int,
    h: float = 5,
    hColor: float = 10,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
) -> np.ndarray:
    """
    Temporal non-local means denoising (cv2.fastNlMeansDenoisingColoredMulti).

    Denoises the centre frame using the neighbouring frames of the window as
    extra samples. The window is 3 frames wide, so output lags input by one frame.

    Args:
        h: Filter strength for the luminance component.
        hColor: Filter strength for the colour components.
        templateWindowSize: Odd patch size used to compute weights.
        searchWindowSize: Odd window size for the weighted average; cost scales linearly.
    """
    return cv2.fastNlMeansDenoisingColoredMulti(
        frames, centre, len(frames), None, h, hColor, templateWindowSize, searchWindowSize
    )


@frame_op
def satboost(frame: np.ndarray, alpha=1.1, thresh=0.5) -> np.ndarray:
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))
    if s.mean() / 255.0 > thresh:
        s = np.clip(s * alpha, 0, 255).astype(np.uint8)
    # v = cv2.normalize(v, v, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)


def find_contours(
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
    metadata_key: str = "contours",
) -> Callable[[FrameContext], FrameContext]:
    """
    Triggers object contour detection injecting metadata output.

    Args:
        mode: Outer or exhaustive representation mode for retrieval.
        method: Segment approximation definition.
        metadata_key: Subdict string literal.

    Returns:
        Frame wrapper adding hierarchy to contextual output fields.
    """

    def wrapper(ctx: FrameContext) -> FrameContext:
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
    thickness: int = 2,
) -> Callable[[FrameContext], FrameContext]:
    """
    Superimposes previously detected polygons directly atop the current matrix values.

    Args:
        metadata_key: Read position string for poly points.
        color: Standard 3 channel pixel marker array tuple.
        thickness: Render radius limit logic override.
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
    draw: bool = True,
) -> Callable[[FrameContext], FrameContext]:
    """
    Derives attributes (moments, areas) across any pre-extracted bounds structures.

    Args:
        metadata_key: Parent contour source dict identifier string.
        output_key: Result destination dict identifier string.
        draw: Switch deciding explicit centroid+box rendering routines or stealth.

    Returns:
        Mutated context handling nested moments output metadata object shapes.
    """

    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_key not in ctx.metadata:
            return ctx

        cnts = ctx.metadata[metadata_key]
        results = []

        for c in cnts:
            M = cv2.moments(c)
            cx, cy = 0, 0
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            x, y, w, h = cv2.boundingRect(c)

            data = {
                "moments": M,
                "centroid": (cx, cy),
                "area": area,
                "perimeter": perimeter,
                "bbox": (x, y, w, h),
            }
            results.append(data)

            if draw:
                cv2.circle(ctx.frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.rectangle(ctx.frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        ctx.metadata[output_key] = results
        return ctx

    return wrapper
