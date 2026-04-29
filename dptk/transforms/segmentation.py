import numpy as np
import cv2
from typing import Callable, Literal, Any, Dict
from ..context import FrameContext

def farneback_seg(
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
    magnitude_threshold: float = 3.0,
    morph_kernel_size: int = 5,
    dilate_iterations: int = 2,
    result_key: str = "farneback",
    draw_viz: bool = False,
    draw_mode: Literal["segmentation", "flow"] = "segmentation",
) -> Callable[[FrameContext], FrameContext]:
    """
    Factory that creates a Farneback dense optical flow segmentation transform.

    Args:
        pyr_scale: Image scale (<1) to build pyramids for each image.
        levels: Number of pyramid layers including the initial image.
        winsize: Averaging window size.
        iterations: Number of iterations the algorithm does at each pyramid level.
        poly_n: Size of the pixel neighborhood used to find polynomial expansion.
        poly_sigma: Standard deviation of the Gaussian that is used to smooth derivatives.
        flags: Operation flags.
        magnitude_threshold: Flow magnitude threshold for moving objects.
        morph_kernel_size: Size of the morphological opening operation kernel.
        dilate_iterations: Number of iterations for the dilation operation.
        result_key: Key to store results in ctx.metadata.
        draw_viz: Whether to overwrite ctx.frame with the visualization.
        draw_mode: 'segmentation' to show the binary mask, 'flow' to show HSV colored flow.

    Returns:
        Transform function processing incoming FrameContext.
    """
    
    # Use a dictionary to maintain state across closure executions
    state: Dict[str, Any] = {
        "prev_gray": None,
        "hsv_template": None
    }
    
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    def transform(ctx: FrameContext) -> FrameContext:
        frame = ctx.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize state on first frame
        if state["prev_gray"] is None:
            state["prev_gray"] = gray
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255  # Max saturation
            state["hsv_template"] = hsv
            
            # Populate empty results for consistency
            ctx.metadata[f"{result_key}_flow"] = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)
            ctx.metadata[f"{result_key}_mag"] = np.zeros_like(gray, dtype=np.float32)
            ctx.metadata[f"{result_key}_ang"] = np.zeros_like(gray, dtype=np.float32)
            ctx.metadata[f"{result_key}_seg"] = np.zeros_like(gray, dtype=np.uint8)
            
            if draw_viz:
                if draw_mode == "segmentation":
                    ctx.frame = cv2.cvtColor(np.zeros_like(gray, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
                elif draw_mode == "flow":
                    # empty flow viz is black
                    ctx.frame = np.zeros_like(frame)
                    
            return ctx
            
        # Compute dense optical flow
        flow_input = np.zeros_like(gray, dtype=np.float32) # Dummy for type checker if needed, but cv2 parses None fine at runtime
        flow = cv2.calcOpticalFlowFarneback(
            state["prev_gray"], gray, None, # type: ignore
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        )
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Motion segmentation by thresholding magnitude
        _, thresh = cv2.threshold(mag, magnitude_threshold, 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        
        # Cleanup binary mask using morphological operations
        segmented = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        segmented = cv2.dilate(segmented, kernel, iterations=dilate_iterations)
        
        # Update metadata
        ctx.metadata[f"{result_key}_flow"] = flow
        ctx.metadata[f"{result_key}_mag"] = mag
        ctx.metadata[f"{result_key}_ang"] = ang
        ctx.metadata[f"{result_key}_seg"] = segmented
        
        # Visualization
        if draw_viz:
            if draw_mode == "segmentation":
                # Convert 1-channel segmented mask back to 3-channel for standard output
                ctx.frame = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
            elif draw_mode == "flow":
                hsv = state["hsv_template"].copy()
                hsv[..., 0] = ang * 180 / np.pi / 2
                mag_norm = np.zeros_like(mag, dtype=np.float32)
                hsv[..., 2] = cv2.normalize(mag, mag_norm, 0, 255, cv2.NORM_MINMAX)
                ctx.frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Update previous frame for next iteration
        state["prev_gray"] = gray
        return ctx

    return transform
