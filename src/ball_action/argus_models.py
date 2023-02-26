from typing import Optional

import torch
from torch import nn

import timm
import argus
from argus.engine import State
from argus.utils import deep_to, deep_detach


class BallActionModel(argus.Model):
    nn_module = {
        "timm": timm.create_model,
    }
    prediction_transform = nn.Sigmoid

    def __init__(self, params: dict):
        super().__init__(params)
        self.augmentations: Optional[nn.Module] = None

    def train_step(self, batch, state: State) -> dict:
        self.train()
        self.optimizer.zero_grad()
        input, target = deep_to(batch, device=self.device, non_blocking=True)
        if self.augmentations is not None:
            with torch.no_grad():
                input = self.augmentations(input)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }
