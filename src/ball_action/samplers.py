import abc
import random

from src.abstract_sampler import AbstractSampler
from src.ball_action.target import VideoTarget
from src.frame_fetcher import FrameFetcher


class ActionBallSampler(AbstractSampler, metaclass=abc.ABCMeta):
    def __init__(self, num_frames: int, target_gauss_scale: float):
        self.num_frames = num_frames
        self.target_gauss_scale = target_gauss_scale
        self.behind_num_frames = self.num_frames // 2
        self.ahead_num_frames = self.num_frames - self.behind_num_frames - 1

        self.videos_data = None
        self.num_videos = None
        self.num_videos_actions = None
        self.num_actions = None
        self.videos_target = None

        self.frame_fetcher = FrameFetcher()

    def init_data(self, videos_data):
        self.videos_data = videos_data
        self.num_videos = len(videos_data)
        self.num_videos_actions = [len(v["frame_index2action"]) for v in self.videos_data]
        self.num_actions = sum(self.num_videos_actions)
        self.videos_target = [
            VideoTarget(data, gauss_scale=self.target_gauss_scale) for data in videos_data
        ]

    def sample_frame_indexes(self, frame_index):
        return list(
            range(
                frame_index - self.behind_num_frames,
                frame_index + self.ahead_num_frames + 1
            )
        )


class TrainActionBallSampler(ActionBallSampler):
    def __init__(self, num_frames, target_gauss_scale, num_samples):
        super().__init__(num_frames, target_gauss_scale)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def sample(self, index: int):
        assert self.videos_data is not None

        video_index = random.randrange(0, self.num_videos)

        video_data = self.videos_data[video_index]
        video_frame_count = video_data["frame_count"]
        frame_index = random.randrange(
            self.behind_num_frames,
            video_frame_count - self.ahead_num_frames
        )
        frame_indexes = self.sample_frame_indexes(frame_index)
        self.frame_fetcher.init_video(video_data["video_path"], video_frame_count)
        frames = self.frame_fetcher.fetch(frame_indexes)

        targets = self.videos_target[video_index].targets(frame_indexes)

        return frames, targets
