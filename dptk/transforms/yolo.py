import numpy as np
import cv2
from ..context import FrameContext
from typing import Optional, List, Callable
from ultralytics import YOLO


def yolo_detect(
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    classes: list[int] | None = None,
    device: str | None = None,
    verbose: bool = False,
    result_key: str = "yolo",
) -> Callable[[FrameContext], FrameContext]:
    """
    Factory that creates a YOLO detection transform.
    
    Args:
        model_path: Path to YOLO model weights (.pt file).
        conf: Confidence threshold for detections.
        iou: NMS IoU threshold.
        classes: Filter by class IDs (None = all classes).
        device: Device to run on ('cpu', '0', '1', etc.).
        verbose: Show inference logs.
        result_key: Key to store results in ctx.metadata.
    
    Returns:
        Transform function processing incoming FrameContext structures.
    """
    print(f"[yolo_detect] Loading model from {model_path}...")
    model = YOLO(model_path)
    
    model_args = {
        "conf": conf,
        "iou": iou,
        "classes": classes,
        "device": device,
        "verbose": verbose,
    }
    
    def transform(ctx: FrameContext) -> FrameContext:
        """
        Executes YOLO inference against a pre-warmed context object structure.
        """
        results = model(ctx.frame, **model_args)
        
        ctx.metadata[result_key] = results
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "class_id": int(box.cls),
                        "class_name": r.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                        "bbox_norm": box.xywhn[0].tolist(),
                    })
        
        ctx.metadata[f"{result_key}_detections"] = detections
        ctx.metadata[f"{result_key}_count"] = len(detections)
        
        return ctx
    
    transform.model = model
    transform.model_path = model_path
    
    return transform

def crop_to_class(
    target_label: str | None = None,
    target_id: int | None = None,
    resize_to: tuple | None = None,
    on_missing: str = "drop",
):
    """
    Extracts isolated bounding frames from previously collected object data.

    Args:
        target_label: Object string identifier filter target.
        target_id: Object numerical identifier filter constraint.
        resize_to: Optional output bounds sequence structures definition variables arrays.
        on_missing: Missing state flag.
        
    Returns:
        Frame contexts bounded filters logic items bounds sequences.
    """

    def wrapper(ctx: FrameContext) -> FrameContext | None:
        if "yolo" not in ctx.metadata:
            return None

        results = ctx.metadata["yolo"][0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            if on_missing == "drop":
                return None
            else:
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
                return None
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

        if on_missing == "drop":
            return None
        return ctx

    return wrapper

def draw_boxes():
    """
    Embeds raw box data structures arrays logic constraints directly to output matrices references bounding structs.
    
    Returns:
        Render wrapper output mappings sequences definitions items vectors structures symbols pointers strings objects dict vars.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if "yolo" not in ctx.metadata:
            return ctx
            
        results = ctx.metadata["yolo"][0]
        boxes = results.boxes

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cls_id = int(box.cls[0].item())
                label = results.names[cls_id]
                conf = float(box.conf[0].item())
                
                text_parts = [label, f"{conf:.2f}"]
                
                if box.id is not None:
                    track_id = int(box.id[0].item())
                    text_parts.insert(0, f"#{track_id}")
                    
                text = " ".join(text_parts)
                
                cv2.rectangle(ctx.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_label = max(y1, h + 5)
                
                cv2.rectangle(ctx.frame, (x1, y_label - h - 5), (x1 + w, y_label), (0, 255, 0), -1)
                cv2.putText(ctx.frame, text, (x1, y_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
        return ctx
    return wrapper
