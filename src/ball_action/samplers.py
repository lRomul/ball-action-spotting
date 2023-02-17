import abc
import random
from typing import Optional

import numpy as np

from src.abstract_sampler import AbstractSampler
from src.ball_action.target import VideoTarget
from src.frame_fetcher import FrameFetcher


class ActionBallSampler(AbstractSampler, metaclass=abc.ABCMeta):
    def __init__(self, frame_stack_size: int, frame_stack_step: int, target_gauss_scale: float):
        self.frame_stack_size = frame_stack_size
        self.frame_stack_step = frame_stack_step
        self.target_gauss_scale = target_gauss_scale

        self.behind_frames = self.frame_stack_size // 2
        self.ahead_frames = self.frame_stack_size - self.behind_frames - 1
        self.behind_frames *= self.frame_stack_step
        self.ahead_frames *= self.frame_stack_step

        self.videos_data: Optional[list[dict]] = None
        self.num_videos: Optional[int] = None
        self.num_videos_actions: Optional[list[int]] = None
        self.num_actions: Optional[int] = None
        self.videos_target: Optional[list[VideoTarget]] = None

        self.frame_fetcher = FrameFetcher()

    def init_data(self, videos_data: list[dict]):
        self.videos_data = videos_data
        self.num_videos = len(videos_data)
        self.num_videos_actions = [len(v["frame_index2action"]) for v in self.videos_data]
        self.num_actions = sum(self.num_videos_actions)
        self.videos_target = [
            VideoTarget(data, gauss_scale=self.target_gauss_scale) for data in videos_data
        ]

    def sample_frame_indexes(self, frame_index: int):
        return list(
            range(
                frame_index - self.behind_frames,
                frame_index + self.ahead_frames + 1,
                self.frame_stack_step,
            )
        )

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def sample(self, index: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        pass


class TrainActionBallSampler(ActionBallSampler):
    def __init__(self,
                 frame_stack_size: int,
                 frame_stack_step: int,
                 target_gauss_scale: float,
                 epoch_size: int):
        super().__init__(frame_stack_size, frame_stack_step, target_gauss_scale)
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return self.epoch_size

    def sample(self, index: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        assert self.videos_data is not None
        assert self.videos_target is not None

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
