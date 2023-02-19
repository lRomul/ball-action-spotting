from collections import defaultdict

import numpy as np
from scipy.stats import norm  # type: ignore

from src.ball_action import constants


def make_gauss_density(scale: float) -> tuple[np.ndarray, np.ndarray]:
    tail_length = 4 * round(scale)
    x = np.linspace(-tail_length, tail_length, 2 * tail_length + 1).astype(int)
    y = norm.pdf(x, 0, scale)
    y /= y.max()
    return x, y


class VideoTarget:
    def __init__(self, video_data: dict, gauss_scale: float):
        relative_indexes, gauss_pdf = make_gauss_density(gauss_scale)
        self.relative_indexes = [int(x) for x in relative_indexes]
        self.gauss_pdf = [float(y) for y in gauss_pdf]

        self.frame_index2class_target: dict[str, defaultdict] = {
            cls: defaultdict(float) for cls in constants.classes
        }

        self.action_index2frame_index: dict[int, int] = dict()
        actions_sorted_by_frame_index = sorted(
            video_data["frame_index2action"].items(), key=lambda x: x[0]
        )
        for action_index, (frame_index, action) in enumerate(actions_sorted_by_frame_index):
            frame_index2target = self.frame_index2class_target[action]
            for relative_index, value in zip(self.relative_indexes, self.gauss_pdf):
                current_index = frame_index + relative_index
                frame_index2target[current_index] = max(value, frame_index2target[current_index])
            self.action_index2frame_index[action_index] = frame_index

    def target(self, frame_index: int) -> np.ndarray:
        target = np.zeros(constants.num_classes, dtype=np.float32)
        for cls in constants.classes:
            target[constants.class2target[cls]] = self.frame_index2class_target[cls][frame_index]
        return target

    def targets(self, frame_indexes: list[int]) -> np.ndarray:
        targets = [self.target(idx) for idx in frame_indexes]
        return np.stack(targets, axis=0)

    def get_frame_index_by_action_index(self, action_index: int) -> int:
        return self.action_index2frame_index[action_index]
