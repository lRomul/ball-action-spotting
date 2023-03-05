import queue
from typing import Type
from multiprocessing import Queue

from rosny.abstract import AbstractStream
from rosny import ProcessStream, ComposeStream

from src.data_loaders.abstract import AbstractDataLoader
from src.frame_fetchers import AbstractFrameFetcher, NvDecFrameFetcher, OpencvFrameFetcher


class SeekWorkerStream(ProcessStream):
    def __init__(self,
                 dataset,
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


class WorkersStream(ComposeStream):
    def __init__(self, streams: list[ProcessStream]):
        super().__init__()
        for index, stream in enumerate(streams):
            self.__setattr__(f"worker_{index}", stream)


class RandomSeekDataLoader(AbstractDataLoader):
    def __init__(self,
                 dataset,
                 batch_size: int,
                 num_nvenc_workers: int = 1,
                 num_opencv_workers: int = 0,
                 gpu_id: int = 0):
        self.num_nvenc_workers = num_nvenc_workers
        self.num_opencv_workers = num_opencv_workers
        super().__init__(dataset=dataset, batch_size=batch_size, gpu_id=gpu_id)

    def init_workers_stream(self) -> AbstractStream:
        nvenc_streams = [
            SeekWorkerStream(self.dataset, self._index_queue, self._result_queue,
                             NvDecFrameFetcher, self.gpu_id)
            for _ in range(self.num_nvenc_workers)
        ]
        opencv_streams = [
            SeekWorkerStream(self.dataset, self._index_queue, self._result_queue,
                             OpencvFrameFetcher, self.gpu_id)
            for _ in range(self.num_opencv_workers)
        ]
        return WorkersStream(nvenc_streams + opencv_streams)
