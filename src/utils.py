import re
import time
import random
from pathlib import Path

import numpy as np
import cv2  # type: ignore
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


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


def get_best_model_path(dir_path, return_score=False, more_better=True):
    dir_path = Path(dir_path)
    model_scores = []
    for model_path in dir_path.glob('*.pth'):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', str(model_path))
        if score is not None:
            score = float(score.group(0)[1:-4])
            model_scores.append((model_path, score))

    if not model_scores:
        if return_score:
            return None, -np.inf
        else:
            return None

    model_score = sorted(model_scores, key=lambda x: x[1], reverse=more_better)
    best_model_path = model_score[0][0]
    if return_score:
        best_score = model_score[0][1]
        return best_model_path, best_score
    else:
        return best_model_path


def post_processing(frame_indexes: list[int],
                    predictions: np.ndarray,
                    gauss_sigma: float,
                    height: float,
                    distance: int) -> tuple[list[int], list[float]]:
    predictions = gaussian_filter(predictions, gauss_sigma)
    peaks, _ = find_peaks(predictions, height=height, distance=distance)
    confidences = predictions[peaks].tolist()
    action_frame_indexes = (peaks + frame_indexes[0]).tolist()
    return action_frame_indexes, confidences
