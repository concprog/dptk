import cv2
import numpy as np
from typing import Tuple

from ..context import FrameContext
from ..decorators import frame_op


@frame_op
def CLAHE(
    frame: np.ndarray, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Amplifies details via localized contrast optimization on luminance constraints.

    Args:
        frame: The source array matrix mapping.
        clipLimit: Value specifying bounding cutoff for noise controls.
        tileGridSize: Dimensional definition blocks for equalizing histograms.

    Returns:
        Frame with equalized luminosity mapping outputs.
    """
    c = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    if len(frame.shape) == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = c.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    else:
        return c.apply(frame)


@frame_op
def grayworld(frame: np.ndarray, alpha: float = 1.4) -> np.ndarray:
    """
    Rebalances color structures assuming global average mappings trend to neutral gray values.

    Args:
        frame: BGR input array.
        alpha: Smoothing weighting multiplier scale coefficient arrays structures strings limits constants values lengths tags items configs structures pointers tags structs mapping structs arrays strings strings lists lists elements sequences properties links locators mapping targets configurations keys pointers IDs tags IDs identifiers strings configurations arguments URLs properties targets limits definitions structures.

    Returns:
        Recalibrated frame output mapping URLs properties constants items.
    """
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float16)

    avg_a = result[:, :, 1].mean()
    avg_b = result[:, :, 2].mean()

    l_channel_norm = result[:, :, 0] / 255.0

    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * l_channel_norm * alpha)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 127) * l_channel_norm * alpha)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

def wb():
    wb = cv2.xphoto.createGrayworldWB()
    def _op(ctx: FrameContext) -> FrameContext:
        ctx.frame = wb.balanceWhite(ctx.frame.astype(np.uint8))
        return ctx
    return _op

@frame_op
def grayworld_saturated(
    frame: np.ndarray,
    alpha: float = 1.2,
    sat_thresh: float = 0.8
) -> np.ndarray:
    """
    Saturation-masked gray-world white balance in YCbCr.

    Args:
        frame: RGB image, uint8, shape (H, W, 3).
        alpha: Strength of the correction (higher = stronger white balance).
        sat_thresh: Saturation threshold in [0,1].
                    Pixels with S > sat_thresh are excluded from the gray-world averages.
                    Closer to 1.0 ≈ closer to plain gray-world; lower = more color-preserving.

    Returns:
        White-balanced RGB image (uint8).
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an RGB image with shape (H, W, 3)")

    rgb = cv2.GaussianBlur(frame, (5,5), 0).astype(np.float32)  # HxWx3
    rgb_max = rgb.max(axis=2)      # HxW
    rgb_min = rgb.min(axis=2)      # HxW

    max_safe = np.where(rgb_max == 0, 1.0, rgb_max)
    sat = (rgb_max - rgb_min) / max_safe  # HxW, in [0,1]

    sat_mask = (sat <= sat_thresh).astype(np.float32)  # HxW

    ycc = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb).astype(np.float32)  # HxWx3, still [0,255]
    Y = ycc[:, :, 0]
    Cr = ycc[:, :, 1]
    Cb = ycc[:, :, 2]

    mask_sum = sat_mask.sum()
    if mask_sum == 0:
        # Edge case: everything is extremely saturated; fall back to unmasked.
        avg_cr = Cr.mean()
        avg_cb = Cb.mean()
    else:
        avg_cr = (Cr * sat_mask).sum() / mask_sum
        avg_cb = (Cb * sat_mask).sum() / mask_sum

    delta_cr = avg_cr - 150.0
    delta_cb = avg_cb - 130.0
    y_norm = Y / 255.0  # HxW, [0, 1]

    adjust_cr = -delta_cr * y_norm * alpha
    adjust_cb = -delta_cb * y_norm * alpha

    Cr_corr = Cr + adjust_cr
    Cb_corr = Cb + adjust_cb

    Cr_corr = np.clip(Cr_corr, 0, 255)
    Cb_corr = np.clip(Cb_corr, 0, 255)

    ycc_corr = np.stack([Y, Cr_corr, Cb_corr], axis=2).astype(np.uint8)
    result = cv2.cvtColor(ycc_corr, cv2.COLOR_YCrCb2RGB)

    return result

@frame_op
def grayworld_focus_preserving(
    frame: np.ndarray,
    alpha: float = 1.2,
    sharpness_sigma: float = 2.0
) -> np.ndarray:
    """
    Focus-preserving Gray World white balance in YCbCr.
    
    Instead of relying on saturation, this uses a sharpness map (Laplacian energy)
    to weight pixels. In-focus regions contribute more to the illuminant estimate,
    preventing out-of-focus backgrounds from skewing the white balance.

    Args:
        frame: RGB image, uint8, shape (H, W, 3).
        alpha: Strength of the correction.
        sharpness_sigma: Sigma for Gaussian blur on the sharpness map. 
                        Higher = larger regions are considered "in-focus".

    Returns:
        White-balanced RGB image (uint8).
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an RGB image with shape (H, W, 3)")

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    
    energy = lap ** 2
    
    weights = cv2.GaussianBlur(energy, (0, 0), sigmaX=sharpness_sigma)
    
    w_sum = np.sum(weights) + 1e-6

    ycc = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycc[:, :, 0]
    Cr = ycc[:, :, 1]
    Cb = ycc[:, :, 2]

    avg_cr = np.sum(Cr * weights) / w_sum
    avg_cb = np.sum(Cb * weights) / w_sum

    neutral = 128.0
    delta_cr = avg_cr - neutral
    delta_cb = avg_cb - neutral

    y_norm = Y / 255.0

    adjust_cr = -delta_cr * y_norm * alpha
    adjust_cb = -delta_cb * y_norm * alpha

    Cr_corr = Cr + adjust_cr
    Cb_corr = Cb + adjust_cb

    Cr_corr = np.clip(Cr_corr, 0, 255)
    Cb_corr = np.clip(Cb_corr, 0, 255)

    ycc_corr = np.stack([Y, Cr_corr, Cb_corr], axis=2).astype(np.uint8)
    result = cv2.cvtColor(ycc_corr, cv2.COLOR_YCrCb2RGB)
    return result


@frame_op
def white_patch(frame: np.ndarray, percentile: float = 98) -> np.ndarray:
    """
    Dynamically adjusts luminosity scaling by mapping highest bounds definitions values directly onto maximal target sequences targets locators pointers parameters lists vectors pointers mappings constants arrays URLs definitions strings arrays dict arguments elements arrays elements configs sequences pointers properties references labels IDs lists.

    Args:
        frame: Target structures dictionaries pointers sequences pointers labels structures vars URLs string lengths strings sequences locators definitions strings lists mapping dictionaries mapping.
        percentile: Scale values bounds targets arrays targets arrays dict mappings lists elements strings tags locators.

    Returns:
        Normalized target properties dict tags properties dictionaries keys dict lists targets mappings lengths items strings configurations IDs config.
    """
    result = frame.astype(np.float16)
    patch = np.percentile(result, percentile, axis=(0, 1))
    scale = (255.0) / np.maximum(patch, 1.0)
    result = result * scale
    return np.clip(result, 0, 255).astype(np.uint8)


@frame_op
def redHE(frame: np.ndarray):
    """
    Exhaustively scales boundaries mappings structures maps arrays mappings configs string sequences strings tags locators lengths lists tags keys items structures values references mappings arrays locators mapping targets elements URLs mapping targets arrays keys dictionaries mappings matrices mapping.

    Args:
        frame: Input.

    Returns:
        Modulated mappings.
    """
    r, g, b = cv2.split(frame)
    r = cv2.equalizeHist(r)
    return cv2.merge([r, g, b])


@frame_op
def gamma_correction(frame: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Stretches mapping constants bounds limits configurations lists items lists lengths paths arrays locators tags variables identifiers mapping identifiers strings tags parameters structures values locators mapping targets strings elements configs structures keys dict locators mapping parameters vars values variables structures.

    Args:
        frame: URLs dictionaries mappings structures mapping locators arrays references structural locators vectors.
        gamma: Matrix value target mapping strings dict values keys elements.

    Returns:
        Transformed maps URLs arguments targets mappings lists.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    return cv2.LUT(frame, table)

@frame_op
def gamma0(frame: np.ndarray) -> np.ndarray:
    i = np.arange(256)
    f = ((i + 0.5) / 256) ** (5 / 6)
    LUT = np.uint8(f * 256 - 0.5)

    img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    img_ycrcb[:, :, 0] = LUT[img_ycrcb[:, :, 0]]
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
