import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Any


class FrameFetcher:
    def __init__(self):
        self.video_path: Optional[Path] = None
        self.video: Optional[Any] = None
        self.frame_count: Optional[int] = None

    def init_video(self, video_path: str | Path, frame_count: Optional[int]):
        video_path = Path(video_path)
        if self.video_path != video_path:
            self.video_path = video_path
            self.video = cv2.VideoCapture(str(self.video_path))
            if frame_count is not None:
                self.frame_count = frame_count
            else:
                self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def fetch(self, frame_indexes: list[int]) -> list[np.ndarray]:
        if self.video is None:
            raise RuntimeError("Need to initialize video before fetching frames")
        min_frame_index = min(frame_indexes)
        max_frame_index = max(frame_indexes)

        if min_frame_index < 0 or max_frame_index >= self.frame_count:
            raise RuntimeError("Frame index out of range")

        self.video.set(cv2.CAP_PROP_POS_FRAMES, min_frame_index)

        index2frame = dict()
        frame_indexes_set = set(frame_indexes)
        for index in range(min_frame_index, max_frame_index + 1):
            success, frame = self.video.read()
            if not success:
                raise RuntimeError(f"Error while fetching frame '{index}' from '{self.video_path}'")
            if index in frame_indexes_set:
                index2frame[index] = frame

        frames = [index2frame[index] for index in frame_indexes]
        return frames
