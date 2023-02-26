import abc
import random
from typing import Optional

import numpy as np

import torch
from torch.utils.data import Dataset

from src.nvdec_frame_fetcher import NvDecFrameFetcher
from src.ball_action.target import VideoTarget
from src.utils import set_random_seed


def normalize_tensor_frames(frames: torch.Tensor) -> torch.Tensor:
    frames = frames.to(torch.float32) / 255.0
    return frames


def targets_to_tensor(targets: np.ndarray) -> torch.Tensor:
    targets = targets.astype(np.float32, copy=False)
    target_tensor = torch.from_numpy(targets)
    return target_tensor


class ActionBallDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self,
                 videos_data: list[dict],
                 frame_stack_size: int,
                 frame_stack_step: int,
                 target_gauss_scale: float,
                 gpu_id: int = 0):
        self.frame_stack_size = frame_stack_size
        self.frame_stack_step = frame_stack_step
        self.target_gauss_scale = target_gauss_scale
        self.gpu_id = gpu_id

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

        self.frame_fetcher: Optional[NvDecFrameFetcher] = None

    def __len__(self) -> int:
        return self.num_actions

    def make_stack_indexes(self, frame_index: int):
        return list(
            range(
                frame_index - self.behind_frames,
                frame_index + self.ahead_frames + 1,
                self.frame_stack_step,
            )
        )

    def clip_frame_index(self, frame_index: int, frame_count: int):
        if frame_index < self.behind_frames:
            frame_index = self.behind_frames
        elif frame_index >= frame_count - self.ahead_frames:
            frame_index = frame_count - self.ahead_frames - 1
        return frame_index

    @abc.abstractmethod
    def get_video_frame_indexes(self, index: int) -> tuple[int, int]:
        pass

    def get_frames_targets(self,
                           video_index: int,
                           frame_index: int) -> tuple[torch.Tensor, np.ndarray]:
        video_data = self.videos_data[video_index]
        frame_indexes = self.make_stack_indexes(frame_index)
        self.frame_fetcher = NvDecFrameFetcher(video_data["video_path"],
                                               gpu_id=self.gpu_id)
        self.frame_fetcher.num_frames = video_data["frame_count"]
        frames = self.frame_fetcher.fetch_frames(frame_indexes)
        targets = self.videos_target[video_index].targets(frame_indexes)
        return frames, targets

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_index, frame_index = self.get_video_frame_indexes(index)
        frames, targets = self.get_frames_targets(video_index, frame_index)
        input_tensor = normalize_tensor_frames(frames)
        target_tensor = targets_to_tensor(targets)
        return input_tensor, target_tensor


class TrainActionBallDataset(ActionBallDataset):
    def __init__(self,
                 videos_data: list[dict],
                 frame_stack_size: int,
                 frame_stack_step: int,
                 target_gauss_scale: float,
                 epoch_size: int,
                 action_prob: float,
                 action_random_shift: int,
                 gpu_id: int = 0):
        super().__init__(
            videos_data,
            frame_stack_size,
            frame_stack_step,
            target_gauss_scale,
            gpu_id=gpu_id
        )
        self.epoch_size = epoch_size
        self.action_prob = action_prob
        self.action_random_shift = action_random_shift

    def __len__(self) -> int:
        return self.epoch_size

    def get_video_frame_indexes(self, index) -> tuple[int, int]:
        set_random_seed(index)
        video_index = random.randrange(0, self.num_videos)
        video_target = self.videos_target[video_index]
        video_data = self.videos_data[video_index]
        video_frame_count = video_data["frame_count"]
        if random.random() < self.action_prob:
            action_index = random.randrange(0, video_target.num_actions())
            frame_index = video_target.get_frame_index_by_action_index(action_index)
            if self.action_random_shift:
                frame_index += random.randint(-self.action_random_shift, self.action_random_shift)
            frame_index = self.clip_frame_index(frame_index, video_frame_count)
        else:
            frame_index = random.randrange(
                self.behind_frames,
                video_frame_count - self.ahead_frames
            )
        return video_index, frame_index


class ValActionBallDataset(ActionBallDataset):
    def get_video_frame_indexes(self, index: int) -> tuple[int, int]:
        assert 0 <= index < self.__len__()
        action_index = index
        video_index = 0
        for video_index, num_video_actions in enumerate(self.num_videos_actions):
            if action_index >= num_video_actions:
                action_index -= num_video_actions
            else:
                break
        video_target = self.videos_target[video_index]
        video_data = self.videos_data[video_index]
        frame_index = video_target.get_frame_index_by_action_index(action_index)
        frame_index = self.clip_frame_index(frame_index, video_data["frame_count"])
        return video_index, frame_index
