This is a summary of the `dptk` codebase, a Python-based toolkit for building flexible and reusable data processing pipelines.

* * *

### Project Structure

```
/home/ssen4/Projects/dptk/
├───.env
├───.gitignore
├───.python-version
├───main.py
├───pyproject.toml
├───README.md
├───uv.lock
├───.git/...
├───.venv/...
└───dptk/
    ├───__init__.py
    ├───context.py
    ├───decorators.py
    ├───sinks.py
    ├───stream.py
    ├───sources/
    │   ├───__init__.py
    │   ├───file.py
    │   └───yolo.py
    └───transforms/
        ├───__init__.py
        ├───ops.py
        ├───uie.py
        └───yolo.py
```

* * *

### Core Modules

#### `dptk/stream.py`

The `Stream` class is the core of the library, providing a fluent interface for creating and managing data processing pipelines. It supports chaining operations, filtering data, and subscribing sinks to consume the results.

```python
class Stream:
    def __init__(self, source: Iterable[FrameContext]):
        self.source = source

    def pipe(self, *ops: Callable[[FrameContext], FrameContext]) -> "Stream":
        """
        Lazily chains operations.
        Returns a NEW Stream object wrapping a generator.
        """

        def generator() -> Iterator[FrameContext]:
            for ctx in self.source:
                processed_ctx = ctx
                for op in ops:
                    processed_ctx = op(processed_ctx)
                yield processed_ctx

        return Stream(generator())

    def subscribe(self, sink: Callable[[Iterable[FrameContext]], None]) -> None:
        """
        Triggers the consumption of the stream.
        Passes the iterable to the sink function.
        """
        sink(self.source)

    def filter(self, predicate: Callable | None = None) -> "Stream":
        """
        Filters the stream.
        If predicate is None (default), it removes all items that are None.
        """

        def generator() -> Iterator[FrameContext]:
            for ctx in self.source:
                # If predicate is explicitly None, use standard Python truthiness filtering
                # Since FrameContext is an object, 'if ctx' is always True.
                # We explicitly check for None here to enable the "Functional Fail State".
                if predicate is None:
                    if ctx is not None:
                        yield ctx
                elif predicate(ctx):
                    yield ctx

        return Stream(generator())

    # Allow the Stream to be treated as an iterable directly
    def __iter__(self):
        return iter(self.source)
```

#### `dptk/context.py`

The `FrameContext` class is a simple data container that encapsulates a single frame of data as it passes through the pipeline. It holds the frame data, index, timestamp, and a dictionary for arbitrary metadata.

```python
import dataclasses
from typing import Any
import numpy as np

@dataclasses.dataclass
class FrameContext:
    """
    container for a single frame
    """
    frame: np.ndarray
    index: int
    timestamp: float
    metadata: dict = dataclasses.field(default_factory=dict)
```

#### `dptk/decorators.py`

This module provides decorators to simplify the creation of new pipeline operations. The `frame_op` decorator, for example, converts a function that operates on a raw NumPy array into a `Stream` compatible transform.

```python
from typing import Callable, Optional
from .context import FrameContext
import numpy as np

def frame_op(func: Callable[[np.ndarray], Optional[np.ndarray]]) -> Callable[[FrameContext], FrameContext]:
    """
    Decorator to convert a function operating on raw numpy arrays 
    into a Transform
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        result = func(ctx.frame)
        
        if result is not None:
            ctx.frame = result
            
        return ctx
    return wrapper
```

* * *

### Sources

Sources are the starting point of a data processing pipeline. They are responsible for generating the initial stream of `FrameContext` objects.

#### `dptk/sources/file.py`

The `VideoSource` function reads a video file and yields a stream of `FrameContext` objects, one for each frame in the video.

```python
from ..context import FrameContext

import time
import cv2
from typing import Iterable

def VideoSource(path: str) -> Iterable[FrameContext]:
    """
    A generator that yields FrameContext objects from a video file.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    try:
        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            yield FrameContext(
                frame=frame,
                index=index, 
                timestamp=time.time(),
                metadata={}
            )
            index += 1
    finally:
        cap.release()
```

#### `dptk/sources/yolo.py`

The `YoloSource` function is a specialized source that wraps `VideoSource` and injects YOLO (You Only Look Once) inference results into the `FrameContext` metadata.

```python
from ultralytics import YOLO
from ..sources.file import VideoSource

def YoloSource(video_path: str, model_path: str = "last.pt", conf: float = 0.25):
    """
    A source that wraps VideoSource and injects YOLO inference results
    into the FrameContext metadata.
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    base_stream = VideoSource(video_path)

    for ctx in base_stream:
        results = model(ctx.frame, conf=conf, verbose=False)
        ctx.metadata["yolo"] = results
        yield ctx
```

* * *

### Transforms

Transforms are operations that modify the `FrameContext` objects as they pass through the pipeline.

#### `dptk/transforms/uie.py`

This module contains image enhancement transforms, such as `CLAHE` (Contrast Limited Adaptive Histogram Equalization) and `grayworld` color balancing.

```python
import cv2
import numpy as np

from ..context import FrameContext
from ..decorators import frame_op

@frame_op
def CLAHE(frame):
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    l = c.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

@frame_op
def grayworld(frame):
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result
```

#### `dptk/transforms/yolo.py`

This module provides transforms for working with YOLO inference results. The `crop_to_class` function, for example, crops the frame to the bounding box of a specified class.

```python
import numpy as np
import cv2
from ..context import FrameContext

def crop_to_class(
    target_label: str | None = None,
    target_id: int | None = None,
    resize_to: tuple | None = None,
    on_missing: str = "drop",
):
    def wrapper(ctx: FrameContext) -> FrameContext | None:
        if "yolo" not in ctx.metadata:
            return None

        results = ctx.metadata["yolo"][0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return None if on_missing == "drop" else ctx

        # ... cropping logic ...

        return ctx
    return wrapper
```

* * *

### Sinks

Sinks are the end point of a data processing pipeline. They are responsible for consuming the final stream of `FrameContext` objects.

#### `dptk/sinks.py`

This module provides several common sinks, such as `display` (to show the frames in a window), `write` (to save the frames to a video file), and `count` (to count the number of frames).

```python
import cv2
from typing import Callable, Iterable
from .context import FrameContext

def display(window_name: str = "Output") -> Callable[[Iterable[FrameContext]], None]:
    def consume(stream: Iterable[FrameContext]) -> None:
        for ctx in stream:
            cv2.imshow(window_name, ctx.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    return consume

def write(path: str, fps: float = 30.0) -> Callable[[Iterable[FrameContext]], None]:
    def consume(stream: Iterable[FrameContext]) -> None:
        writer = None
        for ctx in stream:
            if writer is None:
                h, w = ctx.frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            writer.write(ctx.frame)
        if writer:
            writer.release()
    return consume
```

* * *

### Example Usage

Here is an example of how you might use the `dptk` library to build a simple pipeline that reads a video, applies a `CLAHE` transform, and then displays the result:

```python
from dptk.stream import Stream
from dptk.sources.file import VideoSource
from dptk.transforms.uie import CLAHE
from dptk.sinks import display

def main():
    stream = Stream(VideoSource("my_video.mp4"))
    
    pipeline = stream.pipe(
        CLAHE()
    )
    
    pipeline.subscribe(display("Enhanced Video"))

if __name__ == "__main__":
    main()
```