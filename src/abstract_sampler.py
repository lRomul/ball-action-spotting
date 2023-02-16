import abc
from typing import Any


class AbstractSampler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init_data(self, data: Any):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def sample(self, index: int) -> Any:
        pass
