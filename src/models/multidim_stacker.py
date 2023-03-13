"""
Single-stage model combining 2.5D and 3D data to properly extract temporal information
from the video data.
Original idea:
https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/359932
Implementation example:
https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/392402#2170010
"""

from torch import nn

import timm


class Conv3dBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, act_layer=nn.ReLU):
        super().__init__()
        self.num_features = in_features
        self.conv3d = nn.Conv3d(
            in_features,
            out_features,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=1,
        )
        self.bn3d = nn.BatchNorm3d(out_features)
        if act_layer is None:
            self.act = None
        else:
            self.act = act_layer()

    def forward(self, frames, residual_features=None):
        x = self.conv3d(frames)
        x = self.bn3d(x)
        if residual_features is not None:
            x = x + residual_features
        if self.act is not None:
            x = self.act(x)
        return x


class Conv3dEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_features: int, act_layer=nn.ReLU):
        super().__init__()
        self.num_features = num_features
        self.block1 = Conv3dBlock(num_features, hidden_features, act_layer=act_layer)
        self.block2 = Conv3dBlock(hidden_features, num_features, act_layer=act_layer)

    def forward(self, frames):
        residual_features = frames
        x = self.block1(frames)
        x = self.block2(x, residual_features)
        return x


class MultiDimStacker(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 num_frames: int,
                 conv2d_stack: int = 3,
                 num_conv3d_features: int = 512,
                 num_conv3d_hidden: int = 256,
                 pretrained: bool = False,
                 **kwargs):
        super().__init__()
        assert num_frames > 0 and num_frames % 3 == 0
        self.num_frames = num_frames
        self.conv2d_stack = conv2d_stack
        self.num_stacks = num_frames // conv2d_stack
        self.num_conv3d_features = num_conv3d_features
        self.num_conv3d_hidden = num_conv3d_hidden

        self.conv2d_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=conv2d_stack,
            **kwargs
        )

        self.conv2d_projection = nn.Sequential(
            nn.Conv2d(
                self.conv2d_encoder.num_features,
                num_conv3d_features,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(num_conv3d_features),
            nn.ReLU(),
        )

        self.conv3d_encoder = Conv3dEncoder(
            num_conv3d_features, num_conv3d_hidden, act_layer=nn.ReLU
        )

        assert self.conv2d_encoder.num_features % self.num_stacks == 0
        self.conv3d_projection = nn.Sequential(
            nn.Conv2d(
                num_conv3d_features,
                self.conv2d_encoder.num_features // self.num_stacks,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(self.conv2d_encoder.num_features // self.num_stacks),
            nn.ReLU(),
        )

    def forward(self, frames):
        b, t, h, w = frames.shape  # (2, 15, 736, 1280)
        assert t == self.num_frames
        stacked_frames = frames.view(
            b * self.num_stacks, self.conv2d_stack, h, w
        )  # (10, 3, 736, 1280)
        conv2d_features = self.conv2d_encoder.forward_features(
            stacked_frames
        )  # (10, 1280, 23, 40)
        conv2d_features = self.conv2d_projection(conv2d_features)  # (10, 512, 23, 40)
        b2d, c2d, h, w = conv2d_features.shape
        conv2d_features = conv2d_features.contiguous().view(
            b, self.num_conv3d_features, self.num_stacks, h, w
        )  # (2, 512, 5, 23, 40)
        conv3d_features = self.conv3d_encoder(conv2d_features)  # (2, 512, 5, 23, 40)
        conv3d_features = conv3d_features.view(b2d, c2d, h, w)  # (10, 512, 23, 40)
        conv3d_features = self.conv3d_projection(conv3d_features)  # (10, 256, 23, 40)
        conv3d_features = conv3d_features.view(
            b, self.conv2d_encoder.num_features, h, w
        )  # (2, 1280, 23, 40)
        output = self.conv2d_encoder.forward_head(conv3d_features)  # (2, 2)
        return output
