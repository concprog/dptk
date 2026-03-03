import threading
import multiprocess
from typing import Iterable, Callable, Iterator, Optional, List

from os import cpu_count
from concurrent.futures import ThreadPoolExecutor
import time

_SENTINEL = "__DPTK_STREAM_SENTINEL__"
_sink_executor = ThreadPoolExecutor(max_workers=cpu_count()-2)

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

        self._input_queue: multiprocess.Queue = multiprocess.Queue(maxsize=64)

        self._output_queues: List[multiprocess.Queue] = []

        self._worker: Optional[multiprocess.Process | threading.Thread] = None
        self._generator = source_generator

        self._running = False

        if self.parent:
            self.parent._output_queues.append(self._input_queue)

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
                for q in self._output_queues:
                    q.put(item)
        finally:
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
        for ctx in iter(in_queue.get, _SENTINEL):
            processed_ctx = ctx
            for op in ops:
                processed_ctx = op(processed_ctx)
                if processed_ctx is None:
                    break

            if processed_ctx is not None:
                for q in out_queues:
                    q.put(processed_ctx)

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
        if predicate is None:
            op = lambda ctx: ctx if ctx is not None else None
        else:
            op = lambda ctx: ctx if predicate(ctx) else None
        return self.pipe(op)

    def subscribe(self, sink: Callable) -> None:
        """
        Attaches a final consumption function to the stream and initiates execution.
        """
        sink_queue = multiprocess.Queue(maxsize=64)
        self._output_queues.append(sink_queue)

        queue_iterator = iter(sink_queue.get, _SENTINEL)
        _sink_executor.submit(sink, queue_iterator)
        
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

    def join(self):
        """
        Waits for the stream worker to complete execution and terminates if halted.
        """
        if self._worker:
            if isinstance(self._worker, threading.Thread):
                pass
            elif isinstance(self._worker, multiprocess.Process):
                if self._worker.is_alive():
                    self._worker.join(timeout=1.0)
                    if self._worker.is_alive():
                        self._worker.terminate()


def wait_till_complete(*streams: Stream):
    """
    Blocks the main thread until all provided streams (and their upstream parents)
    are finished processing.

    This replaces the need for manual while-loops or rclpy.spin() calls in main().
    """
    # 1. Ensure everything is running
    for s in streams:
        s._ensure_running()

    try:
        # 2. Block until the root source queues are closed and workers join.
        roots = set()
        for s in streams:
            current = s
            while current.parent:
                current = current.parent
            roots.add(current)

        while any(r._worker and r._worker.is_alive() for r in roots):
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\\nStopping pipeline...")
        # Inject the sentinel into all root queues to forcefully terminate downstream child processes
        for r in roots:
            for q in r._output_queues:
                try:
                    q.put(_SENTINEL)
                except Exception:
                    pass
    finally:
        # Cleanup
        for r in roots:
            r.join()
