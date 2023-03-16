import torch

from timm.data.mixup import Mixup


def mixup_target(target, num_classes, lam=1., smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = (1 - target) * off_value + target * on_value
    y2 = y1.flip(0)
    return y1 * lam + y2 * (1. - lam)


class TimmMixup(Mixup):
    @torch.no_grad()
    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target
