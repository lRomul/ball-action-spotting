import queue
from typing import Type
from multiprocessing import Queue

from rosny import ProcessStream, ComposeStream

from src.frame_fetchers import AbstractFrameFetcher, NvDecFrameFetcher, OpencvFrameFetcher
from src.data_loaders.base_data_loader import BaseDataLoader
from src.datasets import ActionDataset


class RandomSeekWorkerStream(ProcessStream):
    def __init__(self,
                 dataset: ActionDataset,
                 index_queue: Queue,
                 result_queue: Queue,
                 frame_fetcher_class: Type[AbstractFrameFetcher],
                 gpu_id: int = 0,
                 timeout: float = 1.0):
        super().__init__()
        self._dataset = dataset
        self._index_queue = index_queue
        self._result_queue = result_queue
        self._frame_fetcher_class = frame_fetcher_class
        self._gpu_id = gpu_id
        self._timeout = timeout

    def work(self):
        try:
            index = self._index_queue.get(timeout=self._timeout)
        except queue.Empty:
            return
        sample = self._dataset.get(index, self._frame_fetcher_class, self._gpu_id)
        self._result_queue.put(sample)


class RandomSeekWorkersStream(ComposeStream):
    def __init__(self, streams: list[ProcessStream]):
        super().__init__()
        for index, stream in enumerate(streams):
            self.__setattr__(f"random_seek_{index}", stream)


class RandomSeekDataLoader(BaseDataLoader):
    def __init__(self,
                 dataset: ActionDataset,
                 batch_size: int,
                 num_nvdec_workers: int = 1,
                 num_opencv_workers: int = 0,
                 gpu_id: int = 0):
        self.num_nvdec_workers = num_nvdec_workers
        self.num_opencv_workers = num_opencv_workers
        super().__init__(dataset=dataset, batch_size=batch_size, gpu_id=gpu_id)

    def init_workers_stream(self) -> RandomSeekWorkersStream:
        nvdec_streams = [
            RandomSeekWorkerStream(self.dataset, self._index_queue, self._result_queue,
                                   NvDecFrameFetcher, self.gpu_id)
            for _ in range(self.num_nvdec_workers)
        ]
        opencv_streams = [
            RandomSeekWorkerStream(self.dataset, self._index_queue, self._result_queue,
                                   OpencvFrameFetcher, self.gpu_id)
            for _ in range(self.num_opencv_workers)
        ]
        return RandomSeekWorkersStream(nvdec_streams + opencv_streams)
