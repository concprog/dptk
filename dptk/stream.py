import copy
import signal
import threading
import multiprocess
from collections import deque
from typing import Iterable, Iterator, Callable, Optional, List

from os import cpu_count
from concurrent.futures import ThreadPoolExecutor, Future, wait

from .context import FrameContext
from .queue import make_queue, put_sentinel

_SENTINEL = "__DPTK_STREAM_SENTINEL__"
_sink_executor = ThreadPoolExecutor(max_workers=cpu_count()-2)


def _stage(op: Callable, it: Iterator[FrameContext]) -> Iterator[FrameContext]:
    """
    Wraps one op as a generator stage over the incoming frames.

    The stage type is selected by the op's `__batch__` attribute: none for a
    per-frame op, `"chunk"` for a batch op, `"window"` for a sliding-window op.
    """
    batch = getattr(op, "__batch__", None)
    if batch is None:
        return _map_stage(op, it)
    if batch["mode"] == "chunk":
        return _chunk_stage(op, it, batch["size"])
    return _window_stage(
        op, it, batch["size"], batch.get("stride", 1), batch.get("pad"), batch["centre"]
    )


def _map_stage(op: Callable, it: Iterator[FrameContext]) -> Iterator[FrameContext]:
    """
    Applies `op` to each frame. Drops frames for which `op` returns None.
    """
    for ctx in it:
        result = op(ctx)
        if result is not None:
            yield result


def _chunk_stage(
    op: Callable, it: Iterator[FrameContext], size: int
) -> Iterator[FrameContext]:
    """
    Collects `size` frames, then yields every frame returned by `op(list)`.
    The last, shorter chunk is processed as well.
    """
    buf: List[FrameContext] = []
    for ctx in it:
        buf.append(ctx)
        if len(buf) == size:
            yield from op(buf)
            buf = []
    if buf:
        yield from op(buf)


def _window_stage(
    op: Callable,
    it: Iterator[FrameContext],
    size: int,
    stride: int,
    pad: Optional[str],
    centre: int,
) -> Iterator[FrameContext]:
    """
    Slides a window of `size` frames over the stream and yields `op(list)` every
    `stride` frames. With `pad="edge"` the first frame is repeated `centre`
    times and the last frame `size - 1 - centre` times so the output has as
    many frames as the input.
    """
    lead, trail = centre, size - 1 - centre
    window: deque = deque(maxlen=size)
    count = 0

    def emit():
        nonlocal count
        count += 1
        if (count - 1) % stride == 0:
            return op(list(window))
        return None

    for ctx in it:
        if pad == "edge" and not window:
            window.extend(copy.copy(ctx) for _ in range(lead))
        window.append(ctx)
        if len(window) == size:
            result = emit()
            if result is not None:
                yield result

    if pad == "edge" and window:
        last = window[-1]
        for _ in range(trail):
            window.append(copy.copy(last))
            if len(window) == size:
                result = emit()
                if result is not None:
                    yield result

class Stream:
    """
    Directed acyclic graph node representing a stream of frame contexts.

    A Stream can act as a root source processing a generator, or a child node
    applying transformations to data received from a parent Stream.
    """

    def __init__(
        self,
        source_generator: Optional[Iterable | Callable[[], Iterable]] = None,
        ops: tuple = (),
        parent: Optional["Stream"] = None,
    ):
        """
        Initializes a new Stream instance.
        """
        self.parent = parent
        self.ops = ops
        self.children: List["Stream"] = []

        self._input_queue = make_queue()

        self._output_queues: list = []
        self._sink_futures: List[Future] = []

        self._worker: Optional[multiprocess.Process | threading.Thread] = None
        self._generator = source_generator

        self._running = False
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._attach_lock = threading.Lock()

        if self.parent:
            self.parent._attach(self._input_queue)
            self.parent.children.append(self)

    def _attach(self, queue) -> None:
        """
        Registers an output queue.

        A root stream accepts new queues at any time. A finished stream sends
        the sentinel to the new queue at once, so a late consumer still
        terminates. A transform stream that is running cannot accept new queues,
        because its worker process holds its own copy of the queue list.
        """
        with self._attach_lock:
            worker = self._worker
            if isinstance(worker, multiprocess.Process) and worker.is_alive():
                raise RuntimeError("cannot attach to a running transform stream")
            self._output_queues.append(queue)
            if self._finished.is_set() or (worker is not None and not worker.is_alive()):
                queue.put(_SENTINEL)

    def _ensure_running(self):
        """
        Recursively triggers the execution of this stream and all its parent streams.
        """
        if self._running:
            return

        self._running = True

        if self.parent:
            self.parent._ensure_running()

        if self._worker is None:
            if self.parent is None:
                self._worker = threading.Thread(target=self._source_loop, daemon=True)
            else:
                self._worker = multiprocess.Process(
                    target=self._run_transform_process,
                    args=(self._input_queue, self._output_queues, self.ops),
                )

            self._worker.start()

    def _source_loop(self):
        """
        Executes the root generator source and distributes items to connected output queues.
        """
        try:
            iterable = (
                self._generator() if callable(self._generator) else self._generator
            )
            for item in iterable:
                if self._stop.is_set():
                    break
                for q in self._output_queues:
                    q.put(item)
        finally:
            with self._attach_lock:
                self._finished.set()
                for q in self._output_queues:
                    q.put(_SENTINEL)

    @staticmethod
    def _run_transform_process(
        in_queue: multiprocess.Queue, out_queues: List[multiprocess.Queue], ops: tuple
    ):
        """
        Consumes frames from the input queue, applies transformations sequentially,
        and distributes the results to all output queues.
        """
        # The parent process controls shutdown through the sentinel.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            it: Iterator[FrameContext] = iter(in_queue.get, _SENTINEL)
            for op in ops:
                it = _stage(op, it)
            for ctx in it:
                for q in out_queues:
                    q.put(ctx)
        finally:
            for q in out_queues:
                q.put(_SENTINEL)

    def pipe(self, *ops: Callable) -> "Stream":
        """
        Creates and connects a new child stream to this stream, applying the provided operations.
        """
        return Stream(ops=ops, parent=self)

    def filter(self, predicate: Callable | None = None) -> "Stream":
        """
        Creates a new child stream that filters out frames failing the predicate.
        If no predicate is provided, filters out None objects.
        """
        def op(ctx):
            if predicate is None:
                return ctx
            return ctx if predicate(ctx) else None
        return self.pipe(op)

    def subscribe(self, sink: Callable) -> None:
        """
        Attaches a final consumption function to the stream and initiates execution.
        """
        sink_queue = make_queue()
        self._attach(sink_queue)

        queue_iterator = iter(sink_queue.get, _SENTINEL)
        self._sink_futures.append(_sink_executor.submit(sink, queue_iterator))

        self._ensure_running()

    def __enter__(self):
        """
        Context manager entry point.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point to ensure workers are joined.
        """
        self.join()

    def stop(self):
        """
        Requests an early stop of this stream.

        The source loop stops after its current item. Each output queue receives
        the sentinel, so every child stream and sink ends its iteration normally.
        Frames still queued are discarded.
        """
        self._stop.set()
        for q in self._output_queues:
            put_sentinel(q, _SENTINEL)  # a full queue is skipped; join() terminates the worker

    def join(self):
        """
        Waits for the stream worker to complete execution and terminates it if halted.
        """
        if isinstance(self._worker, multiprocess.Process) and self._worker.is_alive():
            self._worker.join(timeout=1.0)
            if self._worker.is_alive():
                self._worker.terminate()

    def graph(self) -> List["Stream"]:
        """
        Returns all streams connected to this one.

        The list starts at the root source and continues through every child
        stream, so parents always come before their children.
        """
        root = self
        while root.parent:
            root = root.parent
        out, stack = [], [root]
        while stack:
            s = stack.pop()
            out.append(s)
            stack.extend(s.children)
        return out


def wait_till_complete(*streams: Stream):
    """
    Blocks the main thread until all provided streams (and their upstream parents)
    are finished processing.

    This replaces the need for manual while-loops or rclpy.spin() calls in main().
    Processing is complete when every sink has consumed its stream. On
    KeyboardInterrupt, each stream is stopped in graph order and the sinks get
    up to 10 seconds to finish, so output files are closed correctly.
    """
    graph = list(dict.fromkeys(s for st in streams for s in st.graph()))
    for s in graph:
        s._ensure_running()
    sinks = [f for s in graph for f in s._sink_futures]

    try:
        wait(sinks)
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        for s in graph:
            s.stop()
        wait(sinks, timeout=10.0)
    finally:
        for s in graph:
            s.join()
