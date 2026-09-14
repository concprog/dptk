# dptk

`dptk` is a lightweight, lazy library for building video and perception processing pipelines. It is designed to handle video streams and perception tasks efficiently by leveraging parallel processing and a functional, DAG-based architecture.

## Installation

Install the package directly from the GitHub repository:

```bash
pip install git+https://github.com/Dreadnought-Robotics/dptk.git
```

## Core Concepts

At the heart of `dptk` is the **FrameContext**, which acts as the atomic unit of data.

`dptk.context.FrameContext` is a simple dataclass that wraps:
1.  **`frame`**: The raw data (usually a NumPy array, such as an image).
2.  **`metadata`**: A dictionary for storing side-channel information (e.g., detected objects, timestamps, scores).
3.  **`index`** & **`timestamp`**: Temporal information.

This structure allows every stage of the pipeline to read/write shared data without changing the function signatures of your operations.

## Architecture

### Streams & Parallelism
The `Stream` class ensures your pipeline is **lazy and highly parallel**. When you define a pipeline, you are building a directed acyclic graph (DAG).
*   **Isolation**: Under the hood, each branch of the stream runs isolated in its own `multiprocess.Process` to bypass Python's GIL.
*   **Threading**: Sources constructed via decorators spawn background daemonic threads to publish data into the processing graph.

### Transforms
A **Transform** is any callable that accepts a `FrameContext` and returns a `FrameContext` (or `None`).

#### The `@frame_op` Decorator
For simple image operations, `dptk` provides the `@frame_op` decorator. It handles the boilerplate of unpacking the context, modifying the frame, and returning the context. This allows you to write functions that operate directly on images/NumPy arrays while seamlessly integrating into the pipeline.

```python
from dptk.decorators import frame_op
import numpy as np
import cv2

@frame_op
def grayworld(frame: np.ndarray, alpha: float = 1.1) -> np.ndarray:
    """
    Rebalances color structures assuming global average mappings trend to neutral gray values.

    Args:
        frame: BGR input array.
        alpha: Smoothing weighting multiplier.

    Returns:
        Recalibrated frame output.
    """
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float16)

    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])

    l_channel_norm = result[:, :, 0] / 255.0

    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * l_channel_norm * alpha)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * l_channel_norm * alpha)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
```

#### Batch and window ops
Some functions accept several frames at once: a YOLO model runs one forward pass for a list of images, and `cv2.fastNlMeansDenoisingColoredMulti` denoises a frame with its neighbours. Two decorators declare that shape. The stream collects frames for the op inside the worker process, so no extra copies are made.

*   **`@batch_op(size)`** — the function receives a list of `size` frames and returns a list of the same length. The last, shorter batch at the end of a stream is processed too.
*   **`@window_op(size, stride=1, pad="edge")`** — the function receives a sliding window of `size` frames (odd) and the index of the centre frame, and returns the new centre frame. With `pad="edge"` the output has as many frames as the input; with `pad=None` only full windows are emitted.

```python
from dptk.decorators import batch_op, window_op

@batch_op(size=8)
def detect(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [draw(f, model(frames)[i]) for i, f in enumerate(frames)]

@window_op(size=3)
def denoise(frames: list[np.ndarray], centre: int) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColoredMulti(frames, centre, len(frames), None, 5, 10)
```

`configure()` works on both. `yolo_detect(..., batch=8)` and `ops.nlmeans_denoise_multi` are ready-made examples.

#### Queue transport
Frames move between pipes through a queue selected by `DPTK_TRANSPORT`:

| value | backend | notes |
|---|---|---|
| `dejaq` (default) | shared-memory ring buffer | pickle-5 out-of-band buffers, one copy per hop |
| `fifo` | faster-fifo | needs `pip install dptk[fastfifo]` |
| `mp` | `multiprocess.Queue` | pipe-based, slowest for large frames |

`DPTK_QUEUE_BYTES` sets the ring size for the shared-memory backends (default 64 MB). Every pipe and every sink owns one ring in `/dev/shm`, so keep the total below the size of that filesystem.

## Usage Example

The following example demonstrates a parallel pipeline setup. It processes a video stream for image enhancement (`pipeline`) while simultaneously running object detection (`gate`) on the same source stream.

```python
from dptk.stream import Stream
from dptk.sources.file import VideoSource
from dptk.transforms.uie import CLAHE, gamma_correction, grayworld, redHE, white_patch
from dptk.sinks import display, count, write
from dptk import configure, run
from dptk.transforms.yolo import crop_to_class, yolo_detect


def main():
    # 1. Define the source
    stream = VideoSource("/home/ssen4/Projects/newcontrol/input_video_sauvc.mp4")

    # 2. Define the Image Enhancement branch
    pipeline = stream.pipe(
        configure(CLAHE, clipLimit=1.5, tileGridSize=(2, 2)),
        configure(gamma_correction, 1.2),
        configure(grayworld, 1.0),
    )

    # 3. Define the Object Detection branch (Parallel)
    gate = stream.pipe(
        yolo_detect("models/last.pt"),
        crop_to_class(target_label="gate", resize_to=(640, 640)),
    ).filter()

    # 4. Subscribe Sinks (Write to file)
    gate.subscribe(write("models/gate_output.mp4"))
    pipeline.subscribe(write("models/uie_output.mp4"))

    # 5. Execute
    # Blocks main thread, runs both branches in parallel
    run(pipeline, gate)


if __name__ == "__main__":
    main()
```

## ROS 2 Integration

`dptk` supports native ROS 2 streams via `RosSource` and `RosPublisherSink`.

```python
import rclpy
from dptk import run, configure
from dptk.ros import RosSource, RosPublisherSink
from dptk.transforms.uie import CLAHE
from dptk.transforms.yolo import yolo_detect, crop_to_class

def main():
    rclpy.init()
    node = rclpy.create_node('dptk_ros_pipeline')
    
    stream = RosSource("/camera/image_raw")
    
    pipeline = stream.pipe(
        configure(CLAHE, clipLimit=1.5, tileGridSize=(2, 2)),
    )
    
    gate = stream.pipe(
        yolo_detect("last.pt"),
        crop_to_class(target_label="gate"),
    ).filter()
    
    pipeline.subscribe(RosPublisherSink(node, "/processed/image"))
    
    # Blocks, gracefully cascades shutdown to multiprocesses on SIGINT
    run(pipeline, gate)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

---

## Available Transforms

All transforms implement `Callable[[FrameContext], FrameContext]`.

### `dptk.transforms.ops` (Basic CV)
*   `resize(width, height)`: Resizes frame.
*   `rotate(angle)`: Rotates frame.
*   `canny(t1, t2)`: Edge detection.
*   `find_contours`, `draw_contours`, `analyze_contours`: Contour pipeline.
*   `nlmeans_denoise`, `nlmeans_denoise_multi`: Non-local means denoising, single-frame and temporal (window of 3).

### `dptk.transforms.transforms` (Geometric)
*   `four_point_transform(pts | pts_key)`: Perspective crop.
*   `warp_perspective(src | src_key, dst | dst_key)`: General wrap.
*   `warp_affine(src | src_key, dst | dst_key)`: Affine warp.
*   `apply_homography(src | src_key, dst | dst_key)`: RANSAC homography.

### `dptk.transforms.pose` (3D/Calibration)
*   `undistort`: Lens correction.
*   `detect_aruco`, `draw_aruco`: Marker detection.
*   `solve_pnp`: 3D-2D pose estimation.
*   `project_points`: Project 3D points back to image.

### `dptk.transforms.features` (Keypoints)
*   `detect_features(algorithm)`: ORB/SIFT/AKAZE.
*   `match_features(template)`: Feature matching.
*   `draw_keypoints`, `draw_matches`: Visualization.

### `dptk.transforms.uie` (Enhancement)
*   `CLAHE`: Contrast enhancement.
*   `grayworld`: White balance.
*   `gamma_correction`: Gamma adjustment.

### `dptk.transforms.yolo` (YOLO utils)
*   `crop_to_class`: Crops detection regions.
*   `draw_boxes`: Visualizes detections.

---

## Library Structure

```
dptk/
├── __init__.py
├── context.py          # Core FrameContext container
├── stream.py           # Stream orchestration and `run()` engine
├── decorators.py       # @frame_op decorator
├── sinks.py            # Sinks (display, write)
├── ros/                # ROS 2 Integration
│   ├── __init__.py
│   ├── msgs.py
│   ├── source.py       # RosSource
│   └── sink.py         # RosPublisherSink
├── sources/
│   ├── __init__.py
│   ├── file.py         # VideoSource
│   └── yolo.py         # YoloSource
└── transforms/
    ├── __init__.py
    ├── features.py     # Feature detection (ORB, SIFT, Matching)
    ├── ops.py          # Basic CV ops (resize, rotate, contours)
    ├── pose.py         # Pose & Calibration (ArUco, PnP, undistort)
    ├── transforms.py   # Geometric transforms (warp, homography)
    ├── uie.py          # Image enhancement (CLAHE, Gray World)
    └── yolo.py         # YOLO utilities (crop_to_class, draw_boxes)
```