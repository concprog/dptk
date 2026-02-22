from dptk.stream import Stream
from dptk.sources.file import VideoSource
from dptk.transforms.uie import CLAHE, gamma_correction, grayworld, redHE, white_patch
from dptk.sinks import display, count, write
from dptk import configure, run
from dptk.transforms.yolo import crop_to_class, yolo_detect


def main():
    stream = VideoSource("/home/ssen4/Projects/newcontrol/input_video_sauvc.mp4")

    pipeline = stream.pipe(
        configure(CLAHE, clipLimit=1.5, tileGridSize=(2, 2)),
        configure(gamma_correction, 1.2),
        configure(grayworld, 1.0),
    )

    gate = stream.pipe(
        yolo_detect("last.pt"),
        crop_to_class(target_label="gate"),
    ).filter()

    gate.subscribe(count())
    pipeline.subscribe(write("../uie_output.mp4"))

    run(pipeline, gate)


if __name__ == "__main__":
    main()
