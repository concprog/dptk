"""
Queues that carry frames between the processes of a pipeline.

Three backends are available. `DPTK_TRANSPORT` selects one:

- `"dejaq"` (default): shared-memory ring buffer, pickle-5 out-of-band buffers.
- `"fifo"`: faster-fifo, shared-memory ring buffer, standard pickle. Needs the
  `fastfifo` extra.
- `"mp"`: `multiprocess.Queue`, pipe-based.

`DPTK_QUEUE_BYTES` sets the ring size in bytes for the shared-memory backends.
Each queue maps its full ring in `/dev/shm`, so keep the size small enough for
the number of queues in the pipeline (one per pipe, one per sink).
"""

import os
import pickle
import warnings
from functools import partial

import multiprocess

TRANSPORT = os.environ.get("DPTK_TRANSPORT", "dejaq")
QUEUE_BYTES = int(float(os.environ.get("DPTK_QUEUE_BYTES", 64e6)))


def make_queue(maxsize: int = 64):
    """
    Creates a frame queue with the backend selected by `DPTK_TRANSPORT`.

    Args:
        maxsize: Item limit for the `multiprocess.Queue` backend. The
            shared-memory backends are limited by `QUEUE_BYTES` instead.

    Returns:
        A queue object with `put(obj)` and `get()`.
    """
    if TRANSPORT == "dejaq":
        try:
            from dejaq import DejaQueue

            return DejaQueue(buffer_bytes=QUEUE_BYTES)
        except ImportError:
            warnings.warn("dejaq not installed, using multiprocess.Queue")
    elif TRANSPORT == "fifo":
        try:
            from faster_fifo import Queue as FifoQueue

            return FifoQueue(
                max_size_bytes=QUEUE_BYTES,
                dumps=partial(pickle.dumps, protocol=5),
                loads=pickle.loads,
            )
        except ImportError:
            warnings.warn("faster-fifo not installed, using multiprocess.Queue")
    return multiprocess.Queue(maxsize=maxsize)


def put_sentinel(queue, sentinel, timeout: float = 1.0) -> None:
    """
    Puts `sentinel` on `queue` without blocking longer than `timeout`, and
    releases the queue's feeder thread so interpreter exit does not wait on
    unread items. A full queue is left as is.
    """
    try:
        queue.put(sentinel, timeout=timeout)
    except Exception:
        pass
    if hasattr(queue, "cancel_join_thread"):
        queue.cancel_join_thread()
