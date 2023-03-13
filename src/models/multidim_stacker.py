"""
Single-stage model combining 2.5D and 3D data to properly extract temporal information
from the video data.
Original idea:
https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/359932
Implementation example:
https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/392402#2170010
"""

from typing import Optional, Type

from torch import nn

import timm
from timm.models.layers import (
    DropPath,
    create_conv2d,
    create_classifier,
    get_act_layer,
    get_norm_act_layer,
)


class Conv3dNormAct(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 act_layer: Optional[Type] = nn.ReLU,
                 act_inplace: bool = True,
                 padding="same"):
        super().__init__()
        self.num_features = in_features
        self.conv3d = nn.Conv3d(
            in_features,
            out_features,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=padding,
        )
        self.bn3d = nn.BatchNorm3d(out_features)
        if act_layer is None:
            self.act = nn.Identity()
        else:
            self.act = act_layer(inplace=act_inplace)

    def forward(self, frames):
        x = self.conv3d(frames)
        x = self.bn3d(x)
        x = self.act(x)
        return x


class Conv3dResidual(nn.Module):
    def __init__(self,
                 num_features: int,
                 hidden_features: int,
                 act_layer: Optional[Type] = nn.ReLU,
                 drop_path_rate=0.,
                 padding="same"):
        super().__init__()
        self.num_features = num_features
        self.block1 = Conv3dNormAct(num_features, hidden_features,
                                    act_layer=act_layer, padding=padding)
        self.block2 = Conv3dNormAct(hidden_features, num_features,
                                    act_layer=None, padding=padding)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, frames):
        shortcut = frames
        x = self.block1(frames)
        x = self.block2(x)
        x = self.drop_path(x) + shortcut
        return x


class MultiDimStacker(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 num_frames: int = 15,
                 stack_size: int = 3,
                 index_2d_features: int = 4,
                 num_3d_features: int = 512,
                 num_3d_hidden: int = 256,
                 num_3d_stack_proj: int = 256,
                 pretrained: bool = False,
                 drop_rate: bool = 0.,
                 drop_path_rate: float = 0.,
                 act_layer: str = "silu",
                 **kwargs):
        super().__init__()
        assert num_frames > 0 and num_frames % 3 == 0
        self.num_frames = num_frames
        self.stack_size = stack_size
        self.num_3d_features = num_3d_features
        self.num_stacks = num_frames // stack_size
        self.num_features = num_3d_stack_proj * self.num_stacks
        self.drop_rate = drop_rate

        act_layer = get_act_layer(act_layer)
        norm_act_layer = get_norm_act_layer(nn.BatchNorm2d, act_layer)
        pad_type = "same"

        self.conv2d_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=stack_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            features_only=True,
            out_indices=[index_2d_features],
            **kwargs
        )

        self.conv2d_projection = nn.Sequential(
            create_conv2d(
                self.conv2d_encoder.feature_info[index_2d_features]["num_chs"],
                num_3d_features,
                kernel_size=1, stride=1,
                padding=pad_type
            ),
            norm_act_layer(num_3d_features, inplace=True)
        )

        self.conv3d_encoder = Conv3dResidual(
            num_3d_features,
            num_3d_hidden,
            act_layer=act_layer,
            drop_path_rate=drop_path_rate,
            padding=pad_type,
        )

        self.conv3d_projection = nn.Sequential(
            create_conv2d(
                num_3d_features,
                num_3d_stack_proj,
                kernel_size=1, stride=1,
                padding=pad_type
            ),
            norm_act_layer(num_3d_stack_proj, inplace=True),
        )

        self.global_pool, self.classifier = create_classifier(
            self.num_features, num_classes, pool_type="avg"
        )

    def forward_2d(self, frames):
        b, t, h, w = frames.shape  # (2, 15, 736, 1280)
        assert t % self.stack_size == 0
        num_stacks = t // self.stack_size
        stacked_frames = frames.view(
            b * num_stacks, self.stack_size, h, w
        )  # (10, 3, 736, 1280)
        conv2d_features = self.conv2d_encoder(
            stacked_frames
        )[-1]  # (10, 1280, 23, 40)
        conv2d_features = self.conv2d_projection(conv2d_features)  # (10, 512, 23, 40)
        _, _, h, w = conv2d_features.shape
        conv2d_features = conv2d_features.contiguous().view(
            b, self.num_3d_features, num_stacks, h, w
        )  # (2, 512, 5, 23, 40)
        return conv2d_features

    def forward_3d(self, conv2d_features):
        b, c, t, h, w = conv2d_features.shape  # (2, 512, 5, 23, 40)
        assert c == self.num_3d_features and t == self.num_stacks
        conv3d_features = self.conv3d_encoder(conv2d_features)  # (2, 512, 5, 23, 40)
        conv3d_features = conv3d_features.view(b * t, c, h, w)  # (10, 512, 23, 40)
        conv3d_features = self.conv3d_projection(conv3d_features)  # (10, 256, 23, 40)
        conv3d_features = conv3d_features.view(
            b, self.num_features, h, w
        )  # (2, 1280, 23, 40)
        return conv3d_features

    def forward_head(self, x):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = nn.functional.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_2d(x)
        x = self.forward_3d(x)
        x = self.forward_head(x)
        return x
