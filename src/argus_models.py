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
        self.freeze_conv2d_encoder = (
            False if 'freeze_conv2d_encoder' not in params
            else bool(params['freeze_conv2d_encoder'])
        )
        super().__init__(params)
        self.iter_size = int(params.get('iter_size', 1))
        self.amp = bool(params.get('amp', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.model_ema = None
        self.augmentations: Optional[nn.Module] = None
        self.mixup: Optional[TimmMixup] = None

    def train_step(self, batch, state: State) -> dict:
        self.train()
        self.optimizer.zero_grad()

        # Gradient accumulation
        loss_value = 0
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
            self.scaler.scale(loss).backward()
            loss_value += loss.item()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.model_ema is not None:
            self.model_ema.update(self.nn_module)

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss_value
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

    def build_nn_module(self, nn_module_meta, nn_module_params):
        nn_module = super().build_nn_module(nn_module_meta, nn_module_params)
        if self.freeze_conv2d_encoder:
            self.logger.info("Freeze conv2d encoder")
            for p in nn_module.conv2d_encoder.parameters():
                p.requires_grad_(False)
        return nn_module
