import signal
import threading
import multiprocess
from typing import Iterable, Callable, Optional, List

from os import cpu_count
from concurrent.futures import ThreadPoolExecutor, Future, wait

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
        self.children: List["Stream"] = []

        self._input_queue: multiprocess.Queue = multiprocess.Queue(maxsize=64)

        self._output_queues: List[multiprocess.Queue] = []
        self._sink_futures: List[Future] = []

        self._worker: Optional[multiprocess.Process | threading.Thread] = None
        self._generator = source_generator

        self._running = False
        self._stop = threading.Event()

        if self.parent:
            self.parent._output_queues.append(self._input_queue)
            self.parent.children.append(self)

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
            for ctx in iter(in_queue.get, _SENTINEL):
                processed_ctx = ctx
                for op in ops:
                    processed_ctx = op(processed_ctx)
                    if processed_ctx is None:
                        break

                if processed_ctx is not None:
                    for q in out_queues:
                        q.put(processed_ctx)
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
        sink_queue = multiprocess.Queue(maxsize=64)
        self._output_queues.append(sink_queue)

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
            try:
                q.put(_SENTINEL, timeout=1.0)
            except Exception:
                pass  # queue is full; join() terminates the worker instead
            q.cancel_join_thread()

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
