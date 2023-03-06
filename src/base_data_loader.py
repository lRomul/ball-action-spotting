import abc
from torch.multiprocessing import Queue

from rosny.loop import LoopStream

from torch.utils.data._utils.collate import default_collate


class BaseDataLoader(metaclass=abc.ABCMeta):
    def __init__(self,
                 dataset,
                 batch_size: int,
                 gpu_id: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.gpu_id = gpu_id

        self._index_queue = Queue(maxsize=len(self.dataset))
        self._result_queue = Queue(maxsize=self.batch_size)

        self._num_samples_left = 0

        self._workers_stream = self.init_workers_stream()
        self.start_workers()

    @abc.abstractmethod
    def init_workers_stream(self) -> LoopStream:
        pass

    def start_workers(self):
        self._workers_stream.start()

    def stop_workers(self):
        if not self._workers_stream.stopped():
            self._workers_stream.stop()
        if not self._workers_stream.joined():
            self._workers_stream.join()

    def clear_queues(self):
        while not self._index_queue.empty():
            self._index_queue.get()
        while not self._result_queue.empty():
            self._result_queue.get()

    def __iter__(self):
        self._num_samples_left = len(self.dataset)
        self.clear_queues()
        for index in range(len(self.dataset)):
            self._index_queue.put(index)
        return self

    def __next__(self):
        batch_list = []
        while self._num_samples_left:
            sample = self._result_queue.get()
            batch_list.append(sample)
            self._num_samples_left -= 1
            if len(batch_list) == self.batch_size:
                return default_collate(batch_list)
        if batch_list:
            return default_collate(batch_list)
        self.clear_queues()
        raise StopIteration

    def __del__(self):
        self.stop_workers()
