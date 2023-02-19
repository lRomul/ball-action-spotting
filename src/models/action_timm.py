import torch
import torch.nn as nn

import timm


class ActionTimm(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 num_frames: int,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.model = timm.create_model(
            model_name=model_name,
            in_chans=num_frames,
            num_classes=num_classes * num_frames,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = torch.reshape(x, (x.shape[0], self.num_frames, self.num_classes))
        return x
