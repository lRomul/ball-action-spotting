from collections import defaultdict

import numpy as np
from scipy.stats import norm

from src.ball_action import constants


def make_gauss_density(scale: float):
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

        self.frame_index2class_target = {cls: defaultdict(float) for cls in constants.classes}

        for frame_index, action in video_data["frame_index2action"].items():
            frame_index2target = self.frame_index2class_target[action]
            for relative_index, value in zip(self.relative_indexes, self.gauss_pdf):
                current_index = frame_index + relative_index
                frame_index2target[current_index] = max(value, frame_index2target[current_index])

    def target(self, frame_index):
        target = np.zeros(constants.num_classes, dtype=np.float32)
        for cls in constants.classes:
            target[constants.class2target[cls]] = self.frame_index2class_target[cls][frame_index]
        return target

    def targets(self, frame_indexes):
        return [self.target(idx) for idx in frame_indexes]
