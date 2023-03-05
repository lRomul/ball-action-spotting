from typing import Type
from multiprocessing import Queue

from rosny import ProcessStream, ComposeStream

from torch.utils.data._utils.collate import default_collate

from src.frame_fetchers import AbstractFrameFetcher, NvDecFrameFetcher, OpencvFrameFetcher
from src.ball_action.datasets import ActionBallDataset


class WorkerStream(ProcessStream):
    def __init__(self,
                 dataset: ActionBallDataset,
                 index_queue: Queue,
                 result_queue: Queue,
                 frame_fetcher_class: Type[AbstractFrameFetcher],
                 gpu_id: int = 0):
        super().__init__()
        self._dataset = dataset
        self._index_queue = index_queue
        self._result_queue = result_queue
        self._frame_fetcher_class = frame_fetcher_class
        self._gpu_id = gpu_id

    def work(self):
        index = self._index_queue.get()
        sample = self._dataset.get(index, self._frame_fetcher_class, self._gpu_id)
        self._result_queue.put(sample)


class WorkersStream(ComposeStream):
    def __init__(self, streams: list[ProcessStream]):
        super().__init__()
        for index, stream in enumerate(streams):
            self.__setattr__(f"process_{index}", stream)


class DataLoader:
    def __init__(self,
                 dataset: ActionBallDataset,
                 batch_size: int,
                 num_nvenc_workers: int = 1,
                 num_opencv_workers: int = 0,
                 gpu_id: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_nvenc_workers = num_nvenc_workers
        self.num_opencv_workers = num_opencv_workers
        self.gpu_id = gpu_id
        self.num_workers = self.num_nvenc_workers + self.num_opencv_workers

        self._index_queue = Queue(maxsize=len(self.dataset))
        self._result_queue = Queue(maxsize=self.num_workers)

        nvenc_streams = [
            WorkerStream(self.dataset, self._index_queue, self._result_queue,
                         NvDecFrameFetcher, self.gpu_id)
            for _ in range(self.num_nvenc_workers)
        ]
        opencv_streams = [
            WorkerStream(self.dataset, self._index_queue, self._result_queue,
                         OpencvFrameFetcher, self.gpu_id)
            for _ in range(self.num_opencv_workers)
        ]

        self._workers_stream = WorkersStream(nvenc_streams + opencv_streams)
        self._workers_stream.start()

        self.num_samples_left = 0

    def clear_queues(self):
        while not self._index_queue.empty():
            self._index_queue.get()
        while not self._result_queue.empty():
            self._result_queue.get()

    def __iter__(self):
        self.num_samples_left = len(self.dataset)
        self.clear_queues()
        for index in range(len(self.dataset)):  # noqa
            self._index_queue.put(index)
        return self

    def __next__(self):
        batch_list = []
        while self.num_samples_left:
            sample = self._result_queue.get()
            batch_list.append(sample)
            self.num_samples_left -= 1
            if len(batch_list) == self.batch_size:
                return default_collate(batch_list)
        if batch_list:
            return default_collate(batch_list)
        self.clear_queues()
        raise StopIteration

    def __del__(self):
        self._workers_stream.stop()
        self._workers_stream.join()
