import cv2 as cv
import numpy as np
from typing import Generator, Tuple

class WritePropertyError(Exception):
    pass

class Video:
    def __init__(self, path: str):
        self.path = path
        self.video = cv.VideoCapture(self.path)
        self._fps = self.video.get(cv.CAP_PROP_FPS)
        self._height = self.video.get(cv.CAP_PROP_FRAME_HEIGHT)
        self._width = self.video.get(cv.CAP_PROP_FRAME_WIDTH)
    
    @property    
    def fps(self) -> float:
        return self._fps
    
    @fps.setter
    def fps(self, value):
        raise WritePropertyError("FPS is read-only")
    
    @property
    def frameSize(self) -> Tuple[float, float]:
        return (self._width, self._height)
    
    @frameSize.setter
    def frameSize(self, value):
        raise WritePropertyError("Frame Size is read-only")
    
        
    def readFrame(self) -> Generator[np.ndarray, None, None]:
        while self.video.isOpened():
            ret, frame = self.video.read()
            
            if not ret:
                print("Broken Frame. Exiting...")
                break
            
            yield frame
            
        self.video.release()
            
    
    def saveFrame(self):
        pass
    
video = Video(path="yolov5\\Record-2.mp4")


