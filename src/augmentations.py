import random

import torch
import torch.nn as nn

from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from kornia.augmentation import random_generator
from kornia.core import as_tensor


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Source: https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


class RandomCameraMove(nn.Module):
    def __init__(self,
                 degrees: tuple[float, float],
                 translate: tuple[float, float],
                 scale: tuple[float, float],
                 p: float = 1.0):
        super().__init__()
        self.param_generator = random_generator.AffineGenerator(degrees, translate, scale)
        self.p = p

    def forward(self, frames_batch):
        output = frames_batch.clone()
        _, num_frames, height, width = frames_batch.shape
        device = frames_batch.device
        for index, frames in enumerate(frames_batch):
            if random.random() > self.p:
                continue

            frames = frames[:, None, :, :]
            params = self.param_generator((2, 1, height, width))
            translations = tensor_linspace(params["translations"][0],
                                           params["translations"][1], num_frames).T
            center = tensor_linspace(params["center"][0], params["center"][1], num_frames).T
            scale = tensor_linspace(params["scale"][0], params["scale"][1], num_frames).T
            angle = tensor_linspace(params["angle"][0], params["angle"][1], num_frames)
            transform = get_affine_matrix2d(
                as_tensor(translations, device=device, dtype=frames.dtype),
                as_tensor(center, device=device, dtype=frames.dtype),
                as_tensor(scale, device=device, dtype=frames.dtype),
                as_tensor(angle, device=device, dtype=frames.dtype),
            )
            output[index] = warp_affine(
                frames,
                transform[:, :2, :],
                (height, width),
            ).squeeze(1)
        return output
