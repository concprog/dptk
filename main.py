from dptk.stream import Stream
from dptk.sources.file import VideoSource
from dptk.transforms.uie import CLAHE, grayworld, redHE
from dptk.sinks import display
from dptk import configure
from dptk.transforms.yolo import crop_to_class, yolo_detect

def main():
    stream = Stream(VideoSource("/home/ssen4/Projects/newcontrol/input_video_sauvc.mp4"))
    
    pipeline = stream.pipe(
        configure(grayworld, 0.9),
        configure(CLAHE, clipLimit=1.5, tileGridSize=(8,8)),
        redHE
    )

    # gate = pipeline.pipe(
    #     yolo_detect("last.pt")
    #     crop_to_class(target_label="gate")
    # ).filter()

    gate.subscribe(display("Gate"))
    pipeline.subscribe(display("Enhanced Video"))

if __name__ == "__main__":
    main()
