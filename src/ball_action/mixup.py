from typing import Optional

import torch

import numpy as np


SampleType = tuple[torch.Tensor, torch.Tensor]


class Mixup:
    def __init__(self,
                 dist_type: str = "uniform",
                 dist_args: Optional[list] = None):
        assert dist_type in ["uniform", "beta"]
        self.dist_type = dist_type
        if dist_args is None:
            if self.dist_type == "uniform":
                self.dist_args = [0, 0.5]
            else:
                self.dist_args = [0.4, 0.4]
        else:
            self.dist_args = dist_args

    def sample_lam(self):
        if self.dist_type == "uniform":
            return np.random.uniform(*self.dist_args)
        elif self.dist_type == "beta":
            return np.random.beta(*self.dist_args)

    def __call__(self, sample1: SampleType, sample2: SampleType) -> SampleType:
        frames1, target1 = sample1
        frames2, target2 = sample2
        lam = self.sample_lam()
        frames = (1 - lam) * frames1 + lam * frames2
        target = (1 - lam) * target1 + lam * target2
        return frames, target
