from typing import Optional

import torch
from torch import nn


@torch.jit.script
def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Source: https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha is not None:
        alpha = alpha.to(targets.device, targets.dtype)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[tuple[float]] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if self.alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32).view(1, -1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs, targets,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction
        )
