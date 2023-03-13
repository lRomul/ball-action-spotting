import abc
import random
from typing import Callable, Type

import numpy as np

import torch

from src.ball_action.indexes import StackIndexesGenerator
from src.frame_fetchers import AbstractFrameFetcher, NvDecFrameFetcher
from src.ball_action.target import VideoTarget
from src.utils import set_random_seed


class ActionBallDataset(metaclass=abc.ABCMeta):
    def __init__(
            self,
            videos_data: list[dict],
            indexes_generator: StackIndexesGenerator,
            target_process_fn: Callable[[np.ndarray], torch.Tensor],
            frames_process_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.indexes_generator = indexes_generator
        self.frames_process_fn = frames_process_fn
        self.target_process_fn = target_process_fn

        self.videos_data = videos_data
        self.num_videos = len(self.videos_data)
        self.num_videos_actions = [len(v["frame_index2action"]) for v in self.videos_data]
        self.num_actions = sum(self.num_videos_actions)
        self.videos_target = [
            VideoTarget(data) for data in self.videos_data
        ]

    def __len__(self) -> int:
        return self.num_actions

    @abc.abstractmethod
    def get_video_frame_indexes(self, index: int) -> tuple[int, int]:
        pass

    def get_targets(self, video_index: int, frame_indexes: list[int]):
        target_indexes = list(range(min(frame_indexes), max(frame_indexes) + 1))
        targets = self.videos_target[video_index].targets(target_indexes)
        return targets

    def get_frames_targets(
            self,
            video_index: int,
            frame_index: int,
            frame_fetcher: AbstractFrameFetcher
    ) -> tuple[torch.Tensor, np.ndarray]:
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        frames = frame_fetcher.fetch_frames(frame_indexes)
        targets = self.get_targets(video_index, frame_indexes)
        return frames, targets

    def get_frame_fetcher(self,
                          video_index: int,
                          frame_fetcher_class: Type[AbstractFrameFetcher],
                          gpu_id: int = 0):
        video_data = self.videos_data[video_index]
        frame_fetcher = frame_fetcher_class(
            video_data["video_path"],
            gpu_id=gpu_id
        )
        frame_fetcher.num_frames = video_data["frame_count"]
        return frame_fetcher

    def process_frames_targets(self, frames: torch.Tensor, targets: np.ndarray):
        input_tensor = self.frames_process_fn(frames)
        target_tensor = self.target_process_fn(targets)
        return input_tensor, target_tensor

    def get(self,
            index: int,
            frame_fetcher_class: Type[AbstractFrameFetcher] = NvDecFrameFetcher,
            gpu_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        video_index, frame_index = self.get_video_frame_indexes(index)
        frame_fetcher = self.get_frame_fetcher(video_index, frame_fetcher_class, gpu_id)
        frames, targets = self.get_frames_targets(video_index, frame_index, frame_fetcher)
        return self.process_frames_targets(frames, targets)


class TrainActionBallDataset(ActionBallDataset):
    def __init__(
            self,
            videos_data: list[dict],
            indexes_generator: StackIndexesGenerator,
            epoch_size: int,
            action_prob: float,
            action_random_shift: int,
            target_process_fn: Callable[[np.ndarray], torch.Tensor],
            frames_process_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__(
            videos_data=videos_data,
            indexes_generator=indexes_generator,
            target_process_fn=target_process_fn,
            frames_process_fn=frames_process_fn,
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
        else:
            frame_index = random.randrange(
                self.indexes_generator.behind,
                video_frame_count - self.indexes_generator.ahead
            )
        frame_index = self.indexes_generator.clip_index(frame_index, video_frame_count, 1)
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
        frame_index = self.indexes_generator.clip_index(frame_index, video_data["frame_count"], 1)
        return video_index, frame_index
