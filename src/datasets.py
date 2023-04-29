import abc
import random
from typing import Callable, Type, Optional

import numpy as np

import torch

from src.indexes import StackIndexesGenerator, FrameIndexShaker
from src.frame_fetchers import AbstractFrameFetcher, NvDecFrameFetcher
from src.utils import set_random_seed
from src.target import VideoTarget


class ActionDataset(metaclass=abc.ABCMeta):
    def __init__(
            self,
            videos_data: list[dict],
            classes: list[str],
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
            VideoTarget(data, classes) for data in self.videos_data
        ]

    def __len__(self) -> int:
        return self.num_actions

    @abc.abstractmethod
    def get_video_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        pass

    def get_targets(self, video_index: int, frame_indexes: list[int]):
        target_indexes = list(range(min(frame_indexes), max(frame_indexes) + 1))
        targets = self.videos_target[video_index].targets(target_indexes)
        return targets

    def get_frames_targets(
            self,
            video_index: int,
            frame_indexes: list[int],
            frame_fetcher: AbstractFrameFetcher
    ) -> tuple[torch.Tensor, np.ndarray]:
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
        video_index, frame_indexes = self.get_video_frame_indexes(index)
        frame_fetcher = self.get_frame_fetcher(video_index, frame_fetcher_class, gpu_id)
        frames, targets = self.get_frames_targets(video_index, frame_indexes, frame_fetcher)
        return self.process_frames_targets(frames, targets)


class TrainActionDataset(ActionDataset):
    def __init__(
            self,
            videos_data: list[dict],
            classes: list[str],
            indexes_generator: StackIndexesGenerator,
            epoch_size: int,
            videos_sampling_weights: list[np.ndarray],
            target_process_fn: Callable[[np.ndarray], torch.Tensor],
            frames_process_fn: Callable[[torch.Tensor], torch.Tensor],
            frame_index_shaker: Optional[FrameIndexShaker] = None,
    ):
        super().__init__(
            videos_data=videos_data,
            classes=classes,
            indexes_generator=indexes_generator,
            target_process_fn=target_process_fn,
            frames_process_fn=frames_process_fn,
        )
        self.epoch_size = epoch_size
        self.frame_index_shaker = frame_index_shaker

        self.videos_sampling_weights = videos_sampling_weights
        self.videos_frame_indexes = [np.arange(v["frame_count"]) for v in videos_data]

    def __len__(self) -> int:
        return self.epoch_size

    def get_video_frame_indexes(self, index) -> tuple[int, list[int]]:
        set_random_seed(index)
        video_index = random.randrange(0, self.num_videos)
        frame_index = np.random.choice(self.videos_frame_indexes[video_index],
                                       p=self.videos_sampling_weights[video_index])
        save_zone = 1
        if self.frame_index_shaker is not None:
            save_zone += max(abs(sh) for sh in self.frame_index_shaker.shifts)
        frame_index = self.indexes_generator.clip_index(
            frame_index, self.videos_data[video_index]["frame_count"], save_zone
        )
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        if self.frame_index_shaker is not None:
            frame_indexes = self.frame_index_shaker(frame_indexes)
        return video_index, frame_indexes


class ValActionDataset(ActionDataset):
    def get_video_frame_indexes(self, index: int) -> tuple[int, list[int]]:
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
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        return video_index, frame_indexes
