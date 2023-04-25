from typing import Optional

import torch
from torch import nn

import timm
import argus
from argus.engine import State
from argus.loss import pytorch_losses
from argus.utils import deep_to, deep_detach, deep_chunk

from src.models.multidim_stacker import MultiDimStacker
from src.losses import FocalLoss
from src.mixup import TimmMixup


class BallActionModel(argus.Model):
    nn_module = {
        "timm": timm.create_model,
        "multidim_stacker": MultiDimStacker,
    }
    loss = {
        **pytorch_losses,
        "focal_loss": FocalLoss,
    }
    prediction_transform = nn.Sigmoid

    def __init__(self, params: dict):
        super().__init__(params)
        self.iter_size = 1 if 'iter_size' not in self.params else int(self.params['iter_size'])
        self.amp = False if 'amp' not in self.params else bool(self.params['amp'])
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.model_ema = None
        self.augmentations: Optional[nn.Module] = None
        self.mixup: Optional[TimmMixup] = None

    def train_step(self, batch, state: State) -> dict:
        self.train()
        self.optimizer.zero_grad()

        # Gradient accumulation
        for i, chunk_batch in enumerate(deep_chunk(batch, self.iter_size)):
            input, target = deep_to(chunk_batch, self.device, non_blocking=True)
            with torch.no_grad():
                if self.augmentations is not None:
                    input = self.augmentations(input)
                if self.mixup is not None:
                    input, target = self.mixup(input, target)
            with torch.cuda.amp.autocast(enabled=self.amp):
                prediction = self.nn_module(input)
                loss = self.loss(prediction, target)
                loss = loss / self.iter_size

            if self.amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.model_ema is not None:
            self.model_ema.update(self.nn_module)

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }

    def val_step(self, batch, state: State) -> dict:
        self.eval()
        with torch.no_grad():
            input, target = deep_to(batch, device=self.device, non_blocking=True)
            if self.model_ema is None:
                prediction = self.nn_module(input)
            else:
                prediction = self.model_ema.ema(input)
            loss = self.loss(prediction, target)
            prediction = self.prediction_transform(prediction)
            return {
                'prediction': prediction,
                'target': target,
                'loss': loss.item()
            }

    def predict(self, input):
        self._check_predict_ready()
        with torch.no_grad():
            self.eval()
            input = deep_to(input, self.device)
            if self.model_ema is None:
                prediction = self.nn_module(input)
            else:
                prediction = self.model_ema.ema(input)
            prediction = self.prediction_transform(prediction)
            return prediction
