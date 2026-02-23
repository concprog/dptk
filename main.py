from dptk.sources.file import VideoSource
from dptk.transforms.uie import CLAHE, gamma_correction, grayworld, redHE, white_patch
from dptk.transforms.yolo import crop_to_class, yolo_detect
from dptk.sinks import write
from dptk import configure, run


def main():
    stream = VideoSource("/home/ssen4/Projects/newcontrol/input_video_sauvc.mp4")

    pipeline = stream.pipe(
        configure(CLAHE, clipLimit=1.5, tileGridSize=(2, 2)),
        configure(gamma_correction, 1.2),
        configure(grayworld, 1.0),
    )

    gate = pipeline.pipe(
        yolo_detect("models/last.pt"),
        crop_to_class(target_label="gate", resize_to=(640, 640)),
    ).filter()

    gate.subscribe(write("models/gate_output.mp4"))
    pipeline.subscribe(write("models/uie_output.mp4"))

    run(pipeline, gate)


if __name__ == "__main__":
    main()
