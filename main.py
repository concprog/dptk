from dptk.sources.file import VideoSource
from dptk.transforms.ops import gblur, normalize, satboost, remove_color_cast, color_transfer_frame
from dptk.transforms.uie import CLAHE, gamma_correction, grayworld, grayworld_saturated, wb
from dptk.transforms.yolo import crop_to_class, yolo_detect
from dptk.sinks import write
from dptk import configure, wait_till_complete


def main():
    stream = VideoSource("/home/ssen4/Projects/newcontrol/input_video_sauvc.mp4")

    pipeline = stream.pipe(
        # remove_color_cast,
        normalize,
        color_transfer_frame,
        normalize,
        configure(CLAHE, clipLimit=1.5, tileGridSize=(2, 2)),
        # normalize,
        # configure(satboost, 0.7),
        # grayworld,
        # gamma_correction,
        # wb(),
        # configure(grayworld_saturated, alpha=1.2),
    )

    # gate = pipeline.pipe(
    #     yolo_detect("models/last.pt"),
    #     crop_to_class(target_label="gate", resize_to=(640, 640)),
    # ).filter()

    pipeline.subscribe(write("models/uie_output.mp4"))
    # gate.subscribe(write("models/gate_output.mp4"))

    wait_till_complete(pipeline)


if __name__ == "__main__":
    main()
