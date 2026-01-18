import numpy as np
import cv2
from ..context import FrameContext


def crop_to_class(
    target_label: str | None = None,
    target_id: int | None = None,
    resize_to: tuple | None = None,
    on_missing: str = "drop",
):
    """
    Functional approach: Returns None if class is not found, allowing
    the stream to filter it out naturally.

    Args:
        on_missing: "drop" (returns None) or "black" (returns black frame).
    """

    def wrapper(ctx: FrameContext) -> FrameContext | None:
        # 1. Check metadata
        if "yolo" not in ctx.metadata:
            return None  # Signal: No data, drop frame

        results = ctx.metadata["yolo"][0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            if on_missing == "drop":
                return None
            else:
                # Black frame fallback logic here if needed
                return ctx

        search_id = target_id
        if target_label is not None:
            label_map = {v: k for k, v in results.names.items()}
            search_id = label_map.get(target_label)

        if search_id is None:
            return None

        class_ids = boxes.cls.cpu().numpy()
        mask = class_ids == search_id

        if not np.any(mask):
            if on_missing == "drop":
                return None  # Functional Fail State
            else:
                return ctx

        target_boxes = boxes.xyxy[mask].cpu().numpy()
        areas = (target_boxes[:, 2] - target_boxes[:, 0]) * (
            target_boxes[:, 3] - target_boxes[:, 1]
        )
        best_idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, target_boxes[best_idx])

        h, w = ctx.frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 > x1) and (y2 > y1):
            cropped = ctx.frame[y1:y2, x1:x2]
            if resize_to:
                cropped = cv2.resize(cropped, resize_to)
            ctx.frame = cropped
            return ctx

        # If crop bounds were invalid for some reason
        if on_missing == "drop":
            return None
        return ctx

    return wrapper
