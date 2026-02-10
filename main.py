from dptk.stream import Stream
from dptk.sources.file import VideoSource
from dptk.transforms.uie import CLAHE, grayworld
from dptk.sinks import display
from dptk import configure

def main():
    stream = Stream(VideoSource("/home/ssen4/Projects/newcontrol/input_video_sauvc.mp4"))
    
    pipeline = stream.pipe(
        configure(grayworld, 0.9),
        configure(CLAHE, clipLimit=2.0, tileGridSize=(8,8)),
    )
    
    pipeline.subscribe(display("Enhanced Video"))

if __name__ == "__main__":
    main()
