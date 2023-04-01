import copy
from typing import Optional

import torch
from torch import nn

import timm

import argus
from argus.engine import State
from argus.utils import deep_to, deep_detach, deep_chunk
from argus.model.build import (
    choose_attribute_from_dict,
    cast_optimizer,
)

from src.models.multidim_stacker import MultiDimStacker


class BallActionModel(argus.Model):
    nn_module = {
        "timm": timm.create_model,
        "multidim_stacker": MultiDimStacker,
    }
    prediction_transform = nn.Sigmoid

    def __init__(self, params: dict):
        super().__init__(params)
        self.iter_size = 1 if 'iter_size' not in self.params else int(self.params['iter_size'])
        self.amp = False if 'amp' not in self.params else bool(self.params['amp'])
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.model_ema = None
        self.augmentations: Optional[nn.Module] = None

    def train_step(self, batch, state: State) -> dict:
        self.train()
        self.optimizer.zero_grad()

        # Gradient accumulation
        for i, chunk_batch in enumerate(deep_chunk(batch, self.iter_size)):
            input, target = deep_to(chunk_batch, self.device, non_blocking=True)
            if self.augmentations is not None:
                with torch.no_grad():
                    input = self.augmentations(input)
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

    def build_optimizer(self, optimizer_meta, optim_params):
        optimizer, optim_params = choose_attribute_from_dict(optimizer_meta,
                                                             optim_params)
        optimizer = cast_optimizer(optimizer)

        optim_params = copy.deepcopy(optim_params)
        weight_decay = optim_params.pop("weight_decay")
        decay_params, no_decay_params = get_weight_decay_params(self.nn_module)
        grad_params = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        optimizer = optimizer(params=grad_params, **optim_params)
        return optimizer


@torch.no_grad()
def get_weight_decay_params(model: nn.Module):
    """ No bias decay
    https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/9
    """
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        if hasattr(param, 'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay
