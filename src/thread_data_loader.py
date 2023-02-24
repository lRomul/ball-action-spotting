import queue
from queue import Queue
from typing import Optional

from rosny import ThreadStream, ComposeStream

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

TIMEOUT = 3.0


class DatasetStream(ThreadStream):
    def __init__(self,
                 dataset: Dataset,
                 index_queue: Queue,
                 result_queue: Queue):
        super().__init__()
        self._dataset = dataset
        self._index_queue = index_queue
        self._result_queue = result_queue
        self.logger.disabled = True

    def on_compile_end(self):
        self.logger.disabled = True

    def work(self):
        try:
            if self._index_queue.empty():
                raise queue.Empty
            index = self._index_queue.get(timeout=TIMEOUT)
        except queue.Empty:
            self.common_state.set_exit()
            return
        sample = self._dataset[index]
        self._result_queue.put(sample)


class ThreadsStream(ComposeStream):
    def __init__(self, streams: list[ThreadStream]):
        super().__init__()
        for index, stream in enumerate(streams):
            self.__setattr__(f"thread_{index}", stream)
        self.logger.disabled = True

    def on_compile_end(self):
        self.logger.disabled = True


class ThreadDataLoader:
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 num_threads: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_threads = num_threads

        self._index_queue = Queue(maxsize=len(self.dataset))  # noqa
        self._result_queue = Queue(maxsize=self.num_threads)

        self._threads_stream: Optional[ThreadsStream] = None

    def __iter__(self):
        self._index_queue = Queue(maxsize=len(self.dataset))  # noqa
        for index in range(len(self.dataset)):  # noqa
            self._index_queue.put(index)
        self._result_queue = Queue(maxsize=self.num_threads)

        self._threads_stream = ThreadsStream([
            DatasetStream(self.dataset, self._index_queue, self._result_queue)
            for _ in range(self.num_threads)
        ])
        self._threads_stream.start()
        return self

    def __next__(self):
        batch_list = []
        for _ in range(self.batch_size):
            try:
                sample = self._result_queue.get(timeout=TIMEOUT)
            except queue.Empty:
                break
            batch_list.append(sample)
        if batch_list:
            return default_collate(batch_list)

        self._threads_stream.stop()
        self._threads_stream.join()
        raise StopIteration
