import torch.nn as nn
import kornia.augmentation as augm


def get_train_augmentations(size: tuple[int, int]) -> nn.Module:
    size = size[::-1]
    ratio = size[0] / size[1]
    transforms = nn.Sequential(
        augm.RandomAffine(degrees=(-10, 10), p=0.5),
        augm.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(ratio - 0.2, ratio + 0.2), p=1.0),
        augm.RandomHorizontalFlip(p=0.5),
        augm.RandomBrightness(brightness=(0.8, 1.2), p=0.5),
        augm.RandomContrast(contrast=(0.8, 1.2), p=0.5),
    )
    return transforms
