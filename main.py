from dptk.sinks.sinks import replace_images_in_folder
from dptk.sources import FolderSource, VideoSource
from dptk.transforms.ops import (
    dilate_erode,
    draw_contours,
    find_contours,
    gblur,
    normalize,
    normalize_a,
    normalize_intensity,
    satboost,
    remove_color_cast,
    color_transfer_frame,
    analyze_contours,
)
from dptk.transforms.segmentation import farneback_seg
from dptk.transforms.uie import (
    CLAHE,
    gamma0,
    gamma_correction,
    grayworld,
    grayworld_saturated,
    wb,
)
from dptk.sinks import write
from dptk import configure, wait_till_complete


def main():
    # stream = VideoSource("/home/ssen4/Projects/newcontrol/input_video_sauvc.mp4")
    stream = VideoSource("rosbagg1.mp4")
    # stream = FolderSource(
    #     "/home/ssen4/Projects/newcontrol/datasets/usefuleshit/dnt_sauvc_yolo_complete_anno/images/train"
    # )

    pipeline = stream.pipe(
        # remove_color_cast,
        normalize,
        color_transfer_frame,
        normalize,
        configure(CLAHE, clipLimit=1.3, tileGridSize=(2, 2)),
        # configure(satboost, 0.7),
        # gamma0,
        # gamma_correction,
    )

    # gate = pipeline.pipe(
    #    yolo_detect("models/last.pt"),
    #    crop_to_class(target_label="gate", resize_to=(640, 640)),
    #    # gblur,
    #    farneback_seg(draw_viz=True),
    #    dilate_erode,
    #    find_contours,
    #    draw_contours
    #    # analyze_contours,
    # ).filter()

    pipeline.subscribe(write("models/uie_output.mp4"))
    # gate.subscribe(write("models/gate_output.mp4"))

    wait_till_complete(pipeline)


if __name__ == "__main__":
    main()
