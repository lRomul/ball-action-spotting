import abc
import random

import numpy as np

import torch
from torch.utils.data import Dataset
import kornia  # type: ignore

from src.ball_action.target import VideoTarget
from src.frame_fetcher import FrameFetcher
from src.utils import set_random_seed


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    frames = frames.astype(np.float32) / 255.0
    tensor_frames = torch.from_numpy(frames)
    tensor_frames = tensor_frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
    tensor_frames = kornia.color.bgr_to_grayscale(tensor_frames)
    tensor_frames = tensor_frames.squeeze(1)
    return tensor_frames


def targets_to_tensor(targets: np.ndarray) -> torch.Tensor:
    targets = targets.astype(np.float32, copy=False)
    target_tensor = torch.from_numpy(targets)
    return target_tensor


class ActionBallDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self,
                 videos_data: list[dict],
                 frame_stack_size: int,
                 frame_stack_step: int,
                 target_gauss_scale: float):
        self.frame_stack_size = frame_stack_size
        self.frame_stack_step = frame_stack_step
        self.target_gauss_scale = target_gauss_scale

        self.behind_frames = self.frame_stack_size // 2
        self.ahead_frames = self.frame_stack_size - self.behind_frames - 1
        self.behind_frames *= self.frame_stack_step
        self.ahead_frames *= self.frame_stack_step

        self.videos_data = videos_data
        self.num_videos = len(self.videos_data)
        self.num_videos_actions = [len(v["frame_index2action"]) for v in self.videos_data]
        self.num_actions = sum(self.num_videos_actions)
        self.videos_target = [
            VideoTarget(data, gauss_scale=self.target_gauss_scale) for data in self.videos_data
        ]

        self.frame_fetcher = FrameFetcher()

    def sample_frame_indexes(self, frame_index: int):
        return list(
            range(
                frame_index - self.behind_frames,
                frame_index + self.ahead_frames + 1,
                self.frame_stack_step,
            )
        )

    @abc.abstractmethod
    def get_frames_targets(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        pass

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames, targets = self.get_frames_targets(index)
        input_tensor = frames_to_tensor(frames)
        target_tensor = targets_to_tensor(targets)
        return input_tensor, target_tensor


class TrainActionBallDataset(ActionBallDataset):
    def __init__(self,
                 videos_data: list[dict],
                 frame_stack_size: int,
                 frame_stack_step: int,
                 target_gauss_scale: float,
                 epoch_size: int):
        super().__init__(videos_data, frame_stack_size, frame_stack_step, target_gauss_scale)
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return self.epoch_size

    def get_frames_targets(self, index) -> tuple[np.ndarray, np.ndarray]:
        set_random_seed(index)

        video_index = random.randrange(0, self.num_videos)

        video_data = self.videos_data[video_index]
        video_frame_count = video_data["frame_count"]
        frame_index = random.randrange(
            self.behind_frames,
            video_frame_count - self.ahead_frames
        )
        frame_indexes = self.sample_frame_indexes(frame_index)
        self.frame_fetcher.init_video(video_data["video_path"], video_frame_count)
        frames = self.frame_fetcher.fetch(frame_indexes)

        targets = self.videos_target[video_index].targets(frame_indexes)

        return frames, targets
