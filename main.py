from dptk.stream import Stream
from dptk.sources.file import VideoSource
from dptk.transforms.uie import CLAHE
from dptk.sinks import display

def main():
    stream = Stream(VideoSource("/home/ssen4/Projects/newcontrol/input_video_sauvc.mp4"))
    
    pipeline = stream.pipe(
        CLAHE
    )
    
    pipeline.subscribe(display("Enhanced Video"))

if __name__ == "__main__":
    main()
