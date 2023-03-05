from queue import Queue
from typing import Optional

import torch
from torch.utils.data._utils.collate import default_collate

from src.ball_action.datasets import ActionBallDataset
from src.frame_fetchers import NvDecFrameFetcher


class SequentialDataLoader:
    def __init__(self,
                 dataset: ActionBallDataset,
                 batch_size: int,
                 frame_stack_size: int,
                 gpu_id: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.frame_stack_size = frame_stack_size
        self.gpu_id = gpu_id

        self._index_queue = Queue(maxsize=len(self.dataset))

        self._video_index = -1
        self._frame_fetcher: Optional[NvDecFrameFetcher] = None
        self._frame_index2frame: dict[int, torch.Tensor] = dict()
        self._last_frame_index = 0

        self._num_samples_left = 0

    def reset(self, video_index: int = -1):
        if video_index == -1:
            self._video_index = -1
            self._frame_fetcher = None
            self._last_frame_index = 0
        else:
            self._video_index = video_index
            video_data = self.dataset.videos_data[video_index]
            frame_fetcher = NvDecFrameFetcher(
                video_data["video_path"],
                gpu_id=self.gpu_id
            )
            frame_fetcher.num_frames = video_data["frame_count"]
            self._frame_fetcher = frame_fetcher
            self._last_frame_index = 0
        self._frame_index2frame = dict()

    def __iter__(self):
        self._num_samples_left = len(self.dataset)
        while not self._index_queue.empty():
            self._index_queue.get()
        for index in range(len(self.dataset)):
            self._index_queue.put(index)
        self.reset()
        return self

    def read_until_last(self, last_frame_index: int):
        while True:
            frame = self._frame_fetcher.fetch_frame()
            frame_index = self._frame_fetcher.current_index
            self._frame_index2frame[frame_index] = frame
            del_frame_index = frame_index - self.frame_stack_size
            if del_frame_index in self._frame_index2frame:
                del self._frame_index2frame[del_frame_index]
            if frame_index == last_frame_index:
                break

    def get_next(self):
        index = self._index_queue.get()
        video_index, frame_index = self.dataset.get_video_frame_indexes(index)
        frame_indexes = self.dataset.indexes_generator.make_stack_indexes(frame_index)
        last_frame_index = max(frame_indexes)
        if video_index != self._video_index:
            self.reset(video_index)
        elif last_frame_index < self._last_frame_index:
            self.reset(video_index)
        self.read_until_last(last_frame_index)

        frames = torch.stack([self._frame_index2frame[i] for i in frame_indexes], dim=0)
        target_indexes = list(range(min(frame_indexes), max(frame_indexes) + 1))
        targets = self.dataset.videos_target[video_index].targets(target_indexes)

        input_tensor = self.dataset.frames_process_fn(frames)
        target_tensor = self.dataset.target_process_fn(targets)
        return input_tensor, target_tensor

    def __next__(self):
        batch_list = []
        while self._num_samples_left:
            sample = self.get_next()
            batch_list.append(sample)
            self._num_samples_left -= 1
            if len(batch_list) == self.batch_size:
                return default_collate(batch_list)
        if batch_list:
            return default_collate(batch_list)
        self.reset()
        raise StopIteration
