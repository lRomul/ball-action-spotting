import abc
from collections import defaultdict

import torch
import numpy as np


class VideoTarget:
    def __init__(self, video_data: dict, classes: list[str]):
        self.classes = classes
        self.num_classes = len(classes)
        self.class2target = {cls: trg for trg, cls in enumerate(classes)}
        self.frame_index2class_target: dict[str, defaultdict] = {
            cls: defaultdict(float) for cls in classes
        }

        self.action_index2frame_index: dict[int, int] = dict()
        actions_sorted_by_frame_index = sorted(
            video_data["frame_index2action"].items(), key=lambda x: x[0]
        )
        for action_index, (frame_index, action) in enumerate(actions_sorted_by_frame_index):
            self.action_index2frame_index[action_index] = frame_index
            if action in classes:
                self.frame_index2class_target[action][frame_index] = 1.0

    def target(self, frame_index: int) -> np.ndarray:
        target = np.zeros(self.num_classes, dtype=np.float32)
        for cls in self.classes:
            target[self.class2target[cls]] = self.frame_index2class_target[cls][frame_index]
        return target

    def targets(self, frame_indexes: list[int]) -> np.ndarray:
        targets = [self.target(idx) for idx in frame_indexes]
        return np.stack(targets, axis=0)

    def get_frame_index_by_action_index(self, action_index: int) -> int:
        return self.action_index2frame_index[action_index]

    def num_actions(self) -> int:
        return len(self.action_index2frame_index)


def center_crop_targets(targets: np.ndarray, crop_size: int) -> np.ndarray:
    num_crop_targets = targets.shape[0] - crop_size
    left = num_crop_targets // 2
    right = num_crop_targets - left
    return targets[left:-right]


class TargetsToTensorProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, targets: np.ndarray) -> torch.Tensor:
        pass


class MaxWindowTargetsProcessor(TargetsToTensorProcessor):
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, targets: np.ndarray) -> torch.Tensor:
        targets = targets.astype(np.float32, copy=False)
        targets = center_crop_targets(targets, self.window_size)
        target = np.amax(targets, axis=0)
        target_tensor = torch.from_numpy(target)
        return target_tensor
