import torch.nn as nn
import kornia.augmentation as augm

from src.augmentations import RandomCameraMove


def get_train_augmentations(size: tuple[int, int]) -> nn.Module:
    size = size[::-1]
    ratio = size[0] / size[1]
    transforms = nn.Sequential(
        RandomCameraMove((-2.5, 2.5), (0.1, 0.05), (0.95, 1.05), p=0.2),
        augm.RandomRotation(degrees=(-2.5, 2.5), p=0.3),
        augm.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(ratio - 0.1, ratio + 0.1), p=0.8),
        augm.RandomHorizontalFlip(p=0.5),
        augm.RandomSharpness(sharpness=1., p=0.2),
        augm.RandomMotionBlur(kernel_size=11, angle=7.5, direction=1.0, p=0.2),
        augm.RandomBrightness(brightness=(0.8, 1.2), p=0.3),
        augm.RandomContrast(contrast=(0.8, 1.2), p=0.3),
        augm.RandomPosterize(bits=3, p=0.2),
        augm.RandomGaussianNoise(mean=0., std=0.05, p=0.2),
    )
    return transforms
