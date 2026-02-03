# dptk

`dptk` is a lightweight, functional-style library for building video and perception processing pipelines. It emphasizes lazy evaluation, modularity, and a clear separation between data sources, transformations, and sinks.

## Core Concepts

At the heart of `dptk` is the **FrameContext**, which acts as the atomic unit of data.

`dptk.context.FrameContext` is a simple dataclass that wraps:
1.  **`frame`**: The raw data (usually a NumPy array, such as an image).
2.  **`metadata`**: A dictionary for storing side-channel information (e.g., detected objects, timestamps, scores).
3.  **`index`** & **`timestamp`**: Temporal information.

This structure allows every stage of the pipeline to read/write shared data without changing the function signatures of your operations.

## Streams

The `Stream` class (`dptk.stream.Stream`) ensures your pipeline is **lazy**. When you define a pipeline, you are building a recipe; no data is processed until you explicitly ask for it.

### Creating a Stream
Streams start from a **Source**. A source is simply any Python iterable that yields `FrameContext` objects.

```python
from dptk.stream import Stream
from dptk.sources.file import VideoSource

# This does NOT read the file yet.
stream = Stream(VideoSource("video.mp4"))
```

### Pipelining
You modify the stream using `.pipe()`. This method takes one or more callables (transforms) and returns a *new* `Stream`. The original stream remains untouched (functional pattern).

```python
# Chains operations lazily
processed_stream = stream.pipe(
    resize(width=640),
    grayscale()
)
```

### Filtering
You can filter frames out of the stream using `.filter()`.
-   **Explicit predicate**: `.filter(lambda ctx: ctx.index % 2 == 0)`
-   **Implicit filtering**: If a transform returns `None`, the stream automatically drops that frame. This is the "Functional Fail State" pattern.

```python
# Keep only frames where 'person' was detected
# (Assuming a previous YoloSource or transform populated metadata)
filtered_stream = stream.filter(lambda ctx: "yolo" in ctx.metadata)
```

## Transforms

A **Transform** is any callable that accepts a `FrameContext` and returns a `FrameContext` (or `None`).

### Writing Transforms
While you can write manual functions, `dptk` provides the `@frame_op` decorator to simplify common image operations. It handles the boilerplate of unpacking the context, modifying the frame, and returning the context.

```python
from dptk.decorators import frame_op

@frame_op
def invert(frame):
    return 255 - frame

# Usage
pipeline = stream.pipe(invert)
```

For more complex logic, specifically when you need to read or write **metadata**, you should write a factory function that returns a wrapper.

```python
def filter_by_score(threshold):
    def wrapper(ctx):
        # Read metadata
        score = ctx.metadata.get("score", 0)
        # Return None to drop the frame
        if score < threshold:
            return None
        return ctx
    return wrapper
```

## Sinks

A **Sink** is the end of the line. It consumes the stream, pulling data through the pipeline. Sinks are any function that accepts an iterable of `FrameContext`.

Common sinks include:
-   `dptk.sinks.display`: Shows frames in a window.
-   `dptk.sinks.write`: Saves frames to a video file.

## Execution

Because streams are lazy, simply defining the pipeline does nothing. To execute the pipeline, you must **subscribe** a sink or iterate over the stream.

```python
# Option 1: Use .subscribe() (Recommended for full pipelines)
pipeline.subscribe(display("My Window"))

# Option 2: Iterate manually (Useful for debugging or custom loops)
for ctx in pipeline:
    print(f"Processed frame {ctx.index}")
```
## Library structure:

```
dptk/
├── __init__.py
├── context.py          # Core FrameContext container
├── stream.py           # Stream orchestration
├── decorators.py       # @frame_op decorator
├── sinks.py            # Sinks (display, write)
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

## Available Transforms:

All transforms implement `Callable[[FrameContext], FrameContext]`.
Many invoke the `@frame_op` decorator internally.

### `dptk.transforms.ops` (Basic CV)
- `resize(width, height)`: Resizes frame.
- `rotate(angle)`: Rotates frame.
- `canny(t1, t2)`: Edge detection.
- `find_contours`, `draw_contours`, `analyze_contours`: Contour pipeline.

### `dptk.transforms.transforms` (Geometric)
- `four_point_transform(pts | pts_key)`: Perspective crop.
- `warp_perspective(src | src_key, dst | dst_key)`: General wrap.
- `warp_affine(src | src_key, dst | dst_key)`: Affine warp.
- `apply_homography(src | src_key, dst | dst_key)`: RANSAC homography.

### `dptk.transforms.pose` (3D/Calibration)
- `undistort`: Lens correction.
- `detect_aruco`, `draw_aruco`: Marker detection.
- `solve_pnp`: 3D-2D pose estimation.
- `project_points`: Project 3D points back to image.

### `dptk.transforms.features` (Keypoints)
- `detect_features(algorithm)`: ORB/SIFT/AKAZE.
- `match_features(template)`: Feature matching.
- `draw_keypoints`, `draw_matches`: Visualization.

### `dptk.transforms.uie` (Enhancement)
- `CLAHE`: Contrast enhancement.
- `grayworld`: White balance.

### `dptk.transforms.yolo` (YOLO utils)
- `crop_to_class`: Crops detection regions.
- `draw_boxes`: Visualizes detections.

## Summary

1.  **Wrap** data in `FrameContext`.
2.  **Create** a `Stream` from a source.
3.  **Chain** operations using `.pipe()`.
4. **Filter** out unwanted frames using `.filter()`.
5.  **Execute** by attaching a Sink.