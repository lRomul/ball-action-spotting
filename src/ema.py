import torch
import logging
from copy import deepcopy
from collections import OrderedDict
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

from argus.utils import deep_to
from argus.engine import State
from argus.callbacks import MonitorCheckpoint


class ModelEma:
    """ Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).
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
    I've tested with the sequence in my own train_cls.py for torch.DataParallel, apex.DDP,
    and single-GPU.
    """
    def __init__(self, model, decay=0.9999, device='', resume=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            logging.info("Loaded state_dict_ema")
        else:
            logging.warning("Failed to find state_dict_ema, starting from loaded model weights")

    @torch.no_grad()
    def update(self, model):
        torch.cuda.synchronize()
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)


class EmaMonitorCheckpoint(MonitorCheckpoint):
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
