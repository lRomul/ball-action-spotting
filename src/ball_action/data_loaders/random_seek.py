import queue
import random
from typing import Type, Optional

from rosny import ProcessStream, ComposeStream
from torch.multiprocessing import Queue

from src.frame_fetchers import AbstractFrameFetcher, NvDecFrameFetcher, OpencvFrameFetcher
from src.ball_action.datasets import ActionBallDataset
from src.base_data_loader import BaseDataLoader
from src.ball_action.mixup import Mixup


class RandomSeekWorkerStream(ProcessStream):
    def __init__(self,
                 dataset: ActionBallDataset,
                 index_queue: Queue,
                 result_queue: Queue,
                 frame_fetcher_class: Type[AbstractFrameFetcher],
                 gpu_id: int = 0,
                 timeout: float = 1.0,
                 mixup_params: Optional[dict] = None):
        super().__init__()
        self._dataset = dataset
        self._index_queue = index_queue
        self._result_queue = result_queue
        self._frame_fetcher_class = frame_fetcher_class
        self._gpu_id = gpu_id
        self._timeout = timeout
        self._mixup: Optional[Mixup] = None
        self._mixup_prob: Optional[float] = None
        if mixup_params is not None:
            self._mixup = Mixup(dist_type=mixup_params["dist_type"],
                                dist_args=mixup_params["dist_args"])
            self._mixup_prob = mixup_params["prob"]

    def work(self):
        try:
            index = self._index_queue.get(timeout=self._timeout)
        except queue.Empty:
            return
        sample = self._dataset.get(index, self._frame_fetcher_class, self._gpu_id)
        if self._mixup is not None and self._mixup_prob is not None:
            if random.random() < self._mixup_prob:
                random_index = random.randrange(0, len(self._dataset))
                random_sample = self._dataset.get(
                    random_index, self._frame_fetcher_class, self._gpu_id
                )
                sample = self._mixup(sample, random_sample)
        self._result_queue.put(sample)


class RandomSeekWorkersStream(ComposeStream):
    def __init__(self, streams: list[ProcessStream]):
        super().__init__()
        for index, stream in enumerate(streams):
            self.__setattr__(f"random_seek_{index}", stream)


class RandomSeekDataLoader(BaseDataLoader):
    def __init__(self,
                 dataset: ActionBallDataset,
                 batch_size: int,
                 num_nvenc_workers: int = 1,
                 num_opencv_workers: int = 0,
                 gpu_id: int = 0,
                 mixup_params: Optional[dict] = None):
        self.num_nvenc_workers = num_nvenc_workers
        self.num_opencv_workers = num_opencv_workers
        self.mixup_params = mixup_params
        super().__init__(dataset=dataset, batch_size=batch_size, gpu_id=gpu_id)

    def init_workers_stream(self) -> RandomSeekWorkersStream:
        nvenc_streams = [
            RandomSeekWorkerStream(self.dataset,
                                   self._index_queue,
                                   self._result_queue,
                                   NvDecFrameFetcher,
                                   gpu_id=self.gpu_id,
                                   mixup_params=self.mixup_params)
            for _ in range(self.num_nvenc_workers)
        ]
        opencv_streams = [
            RandomSeekWorkerStream(self.dataset,
                                   self._index_queue,
                                   self._result_queue,
                                   OpencvFrameFetcher,
                                   gpu_id=self.gpu_id,
                                   mixup_params=self.mixup_params)
            for _ in range(self.num_opencv_workers)
        ]
        return RandomSeekWorkersStream(nvenc_streams + opencv_streams)
