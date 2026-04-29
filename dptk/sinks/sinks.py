import cv2
from typing import Callable, Iterable
from ..context import FrameContext

def display(
    window_name: str = "Output", waitKey: int = 1
) -> Callable[[Iterable[FrameContext]], None]:
    """
    Creates a sink that displays frames in an OpenCV window.

    Args:
        window_name: The title of the display window.
        waitKey: The delay in milliseconds for cv2.waitKey.

    Returns:
        A callable that consumes a stream of FrameContext objects.
    """

    def consume(stream: Iterable[FrameContext]) -> None:
        for ctx in stream:
            cv2.imshow(window_name, ctx.frame)
            if cv2.waitKey(waitKey) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    return consume

def write(path: str, fps: float = 30.0) -> Callable[[Iterable[FrameContext]], None]:
    """
    Creates a sink that writes frames to a video file.

    Args:
        path: The output file path (e.g., 'output.mp4').
        fps: The frames per second for the output video.

    Returns:
        A callable that consumes a stream of FrameContext objects.
    """

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

def count() -> tuple[Callable[[Iterable[FrameContext]], None], Callable[[], int]]:
    """
    Creates a sink that counts the number of frames passing through it.

    Returns:
        A tuple containing the consumer callable and a function to retrieve the current count.
    """
    counter = {"n": 0}
    def consume(stream: Iterable[FrameContext]) -> None:
        for _ in stream:
            counter["n"] += 1
        print(counter["n"])

    return consume

def replace_images_in_folder():
    def consume(stream: Iterable[FrameContext]) -> None:
        for ctx in stream:
            path = ctx.metadata.get("path", None)
            if path is not None:
                cv2.imwrite(path, ctx.frame)
    return consume