import abc
from typing import Optional
from collections import defaultdict

import torch
import numpy as np

from src.ball_action import constants


class VideoTarget:
    def __init__(self, video_data: dict):
        self.frame_index2class_target: dict[str, defaultdict] = {
            cls: defaultdict(float) for cls in constants.classes
        }

        self.action_index2frame_index: dict[int, int] = dict()
        self.action_index2frame_index_by_action: dict[str, dict[int, int]] = {
            cls: dict() for cls in constants.classes
        }
        actions_sorted_by_frame_index = sorted(
            video_data["frame_index2action"].items(), key=lambda x: x[0]
        )
        for action_index, (frame_index, action) in enumerate(actions_sorted_by_frame_index):
            self.action_index2frame_index[action_index] = frame_index
            if action in constants.classes:
                by_action_index = len(self.action_index2frame_index_by_action[action])
                self.action_index2frame_index_by_action[action][by_action_index] = frame_index
                self.frame_index2class_target[action][frame_index] = 1.0

    def target(self, frame_index: int) -> np.ndarray:
        target = np.zeros(constants.num_classes, dtype=np.float32)
        for cls in constants.classes:
            target[constants.class2target[cls]] = self.frame_index2class_target[cls][frame_index]
        return target

    def targets(self, frame_indexes: list[int]) -> np.ndarray:
        targets = [self.target(idx) for idx in frame_indexes]
        return np.stack(targets, axis=0)

    def get_frame_index_by_action_index(self,
                                        action_index: int,
                                        action: Optional[str] = None) -> int:
        if action is None:
            return self.action_index2frame_index[action_index]
        else:
            return self.action_index2frame_index_by_action[action][action_index]

    def num_actions(self, action: Optional[str] = None) -> int:
        if action is None:
            return len(self.action_index2frame_index)
        else:
            return len(self.action_index2frame_index_by_action[action])


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
