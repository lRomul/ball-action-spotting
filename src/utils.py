import cv2  # type: ignore
import time
import random
import numpy as np
from pathlib import Path


def get_video_info(video_path: str | Path) -> dict[str, int | float]:
    video = cv2.VideoCapture(str(video_path))
    video_info = dict(
        frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(video.get(cv2.CAP_PROP_FPS)),
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return video_info


def set_random_seed(index: int):
    seed = int(time.time() * 1000.0) + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
