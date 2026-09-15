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


_dis_cache: dict = {}


def _dis_flow(preset: int) -> "cv2.DISOpticalFlow":
    """Returns a shared DIS optical flow instance for the given preset."""
    if preset not in _dis_cache:
        _dis_cache[preset] = cv2.DISOpticalFlow_create(preset)
    return _dis_cache[preset]


def _mesh(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Returns cached float32 pixel coordinate grids for cv2.remap."""
    key = ("mesh", shape)
    if key not in _dis_cache:
        h, w = shape
        _dis_cache[key] = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
    return _dis_cache[key]


def _median5(a, b, c, d, e) -> np.ndarray:
    """Element-wise median of five arrays using a min/max sorting network."""
    a, b = cv2.min(a, b), cv2.max(a, b)
    c, d = cv2.min(c, d), cv2.max(c, d)
    a, c = cv2.min(a, c), cv2.max(a, c)
    b, d = cv2.min(b, d), cv2.max(b, d)
    # a is the minimum and d the maximum of the first four, so the median of
    # all five is the median of b, c and e.
    b, c = cv2.min(b, c), cv2.max(b, c)
    return cv2.max(b, cv2.min(c, e))


def _particle_mask(
    mask: np.ndarray, frame: np.ndarray, max_area: int, max_sat: int, grow: int
) -> np.ndarray:
    """
    Keeps only candidate pixels that look like particles: saturation at most
    max_sat and part of a connected component no larger than max_area pixels.
    The result is dilated by grow pixels to cover particle fringes.
    """
    sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1]
    mask = cv2.bitwise_and(mask, cv2.threshold(sat, max_sat, 255, cv2.THRESH_BINARY_INV)[1])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    keep = (stats[:, cv2.CC_STAT_AREA] <= max_area).astype(np.uint8) * 255
    keep[0] = 0
    out = np.take(keep, labels)
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        out = cv2.dilate(out, k)
    return out


def _feather_blend(
    frame: np.ndarray, fill: np.ndarray, mask: np.ndarray, feather: int
) -> np.ndarray:
    """Blends fill into frame under a Gaussian-softened binary mask."""
    if feather > 1:
        k = feather | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    w = mask.astype(np.float32) * (1.0 / 255.0)
    return cv2.blendLinear(frame, fill, 1.0 - w, w)


@window_op(size=5)
def remove_particles(
    frames: list,
    centre: int,
    tophat_size: int = 21,
    tophat_thresh: int = 25,
    temporal_thresh: int = 20,
    max_area: int = 2500,
    max_sat: int = 90,
    fill: str = "median",
    feather: int = 7,
    grow: int = 2,
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
) -> np.ndarray:
    """
    Removes bright drifting particles (marine snow, dust, bubbles) from the
    centre frame using its motion-compensated neighbours.

    Neighbouring frames are warped onto the centre with DIS optical flow. A
    pixel is treated as a particle when it is a small bright spot (white
    top-hat) or brighter than the temporal median of the warped window
    (transient). Candidate
    regions larger than max_area or more saturated than max_sat are kept as
    scene content. Masked pixels are replaced by the temporal median (or
    minimum) of the warped window. The window is 5 frames wide, so output lags
    input by two frames.

    Args:
        tophat_size: Structuring element diameter; spots smaller than this are candidates.
        tophat_thresh: Minimum top-hat response (grey levels) for the spatial cue.
        temporal_thresh: Minimum excess brightness over the temporal median.
        max_area: Largest component area (px) still considered a particle.
        max_sat: Largest mean HSV saturation still considered a particle.
        fill: "median", "min" or "inpaint" for the replacement source.
        feather: Blur kernel for the mask edge; 0 or 1 disables.
        grow: Mask dilation radius in pixels.
        dis_preset: cv2.DISOPTICAL_FLOW_PRESET_* controlling flow quality vs speed.
    """
    ref = frames[centre]
    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    # Flow is estimated on lightly filtered greys so particles do not steer it.
    flow_src = [cv2.blur(g, (5, 5)) for g in grey]
    gx, gy = _mesh(grey[centre].shape)
    dis = _dis_flow(dis_preset)

    warped, warped_grey = [], []
    for i, f in enumerate(frames):
        if i == centre:
            continue
        flow = dis.calc(flow_src[centre], flow_src[i], None)
        mx, my = gx + flow[..., 0], gy + flow[..., 1]
        warped.append(cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        warped_grey.append(
            cv2.remap(grey[i], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        )

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_size, tophat_size))
    tophat = cv2.morphologyEx(grey[centre], cv2.MORPH_TOPHAT, k)
    spatial = cv2.threshold(tophat, tophat_thresh, 255, cv2.THRESH_BINARY)[1]

    # Median-frame subtraction: a pixel that is brighter than the temporal
    # median is transient even when one neighbour duplicates the centre.
    grey_med = _median5(warped_grey[0], warped_grey[1], grey[centre], warped_grey[2], warped_grey[3])
    excess = cv2.subtract(grey[centre], grey_med)
    temporal = cv2.threshold(excess, temporal_thresh, 255, cv2.THRESH_BINARY)[1]

    mask = _particle_mask(cv2.bitwise_or(spatial, temporal), ref, max_area, max_sat, grow)
    if not mask.any():
        return ref

    if fill == "median":
        src = _median5(warped[0], warped[1], ref, warped[2], warped[3])
    elif fill == "min":
        src = warped[0]
        for w in warped[1:]:
            src = cv2.min(src, w)
    elif fill == "inpaint":
        src = cv2.inpaint(ref, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f"unknown fill {fill!r}; expected 'median', 'min' or 'inpaint'")

    return _feather_blend(ref, src, mask, feather)


@frame_op
def remove_specks(
    frame: np.ndarray,
    size: int = 15,
    thresh: int = 25,
    max_area: int = 600,
    max_sat: int = 90,
    feather: int = 5,
    grow: int = 2,
) -> np.ndarray:
    """
    Removes small bright specks from a single frame.

    Detects spots brighter than their surroundings and smaller than size via a
    white top-hat, discards candidates that are too large or too saturated to
    be particles, and replaces the rest with the morphological opening of the
    frame. Spatial only: particles larger than size are left untouched.

    Args:
        size: Structuring element diameter; spots smaller than this are candidates.
        thresh: Minimum top-hat response (grey levels).
        max_area: Largest component area (px) still considered a speck.
        max_sat: Largest mean HSV saturation still considered a speck.
        feather: Blur kernel for the mask edge; 0 or 1 disables.
        grow: Mask dilation radius in pixels.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, k)
    cand = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)[1]
    mask = _particle_mask(cand, frame, max_area, max_sat, grow)
    if not mask.any():
        return frame
    opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, k)
    return _feather_blend(frame, opened, mask, feather)


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
