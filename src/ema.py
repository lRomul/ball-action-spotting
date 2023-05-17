from copy import deepcopy

import torch
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from argus.utils import deep_to
from argus.engine import State
from argus.callbacks import Checkpoint


class ModelEma(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.

    Source: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
    """
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.ema.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EmaCheckpoint(Checkpoint):
    def save_model(self, state: State, file_path):
        nn_module = state.model.model_ema.ema
        if isinstance(nn_module, (DataParallel, DistributedDataParallel)):
            nn_module = nn_module.module

        no_ema_nn_module = state.model.get_nn_module()
        if isinstance(no_ema_nn_module, (DataParallel, DistributedDataParallel)):
            no_ema_nn_module = no_ema_nn_module.module

        torch_state = {
            'model_name': state.model.__class__.__name__,
            'params': state.model.params,
            'nn_state_dict': deep_to(nn_module.state_dict(), 'cpu'),
            'no_ema_nn_state_dict': deep_to(no_ema_nn_module.state_dict(), 'cpu')
        }
        torch.save(torch_state, file_path)
        state.logger.info(f"Model saved to '{file_path}'")
