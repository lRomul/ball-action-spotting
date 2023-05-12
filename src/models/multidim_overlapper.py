import math

from torch import nn

import timm
from timm.models.layers import (
    create_conv2d,
    get_act_layer,
    get_norm_act_layer,
)

from .gem import GeneralizedMeanPooling
from .inverted_residual_3d import InvertedResidual3d


class MultiDimOverlapper(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 num_frames: int = 29,
                 overlap_step: int = 5,
                 encoder_2d_frames: int = 9,
                 encoder_2d_step: int = 2,
                 index_2d_features: int = 4,
                 pretrained: bool = False,
                 num_3d_blocks: int = 2,
                 num_3d_features: int = 192,
                 num_3d_stack_proj: int = 256,
                 expansion_3d_ratio: int = 6,
                 se_reduce_3d_ratio: int = 24,
                 drop_rate: bool = 0.,
                 drop_path_rate: float = 0.,
                 act_layer: str = "silu",
                 **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.overlap_step = overlap_step
        self.encoder_2d_frames = encoder_2d_frames
        self.encoder_2d_step = encoder_2d_step
        self.stack_size = math.ceil(encoder_2d_frames / encoder_2d_step)
        self.num_3d_features = num_3d_features
        self.num_stacks = (num_frames - encoder_2d_frames) // overlap_step + 1
        self.num_features = num_3d_stack_proj * self.num_stacks
        self.drop_rate = drop_rate

        act_layer = get_act_layer(act_layer)
        norm_act_layer = get_norm_act_layer(nn.BatchNorm2d, act_layer)

        self.conv2d_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=self.stack_size,
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
            ),
            norm_act_layer(num_3d_features, inplace=True)
        )

        self.conv3d_encoder = nn.Sequential(*[
            InvertedResidual3d(
                num_3d_features,
                num_3d_features,
                expansion_ratio=expansion_3d_ratio,
                se_reduce_ratio=se_reduce_3d_ratio,
                act_layer=act_layer,
                drop_path_rate=drop_path_rate,
            ) for _ in range(num_3d_blocks)
        ])

        self.conv3d_projection = nn.Sequential(
            create_conv2d(
                num_3d_features,
                num_3d_stack_proj,
                kernel_size=1, stride=1,
            ),
            norm_act_layer(num_3d_stack_proj, inplace=True),
        )

        self.global_pool = GeneralizedMeanPooling(3.0)
        self.classifier = nn.Linear(self.num_features, num_classes, bias=True)

    def overlap_frames(self, frames):
        b, t, h, w = frames.shape  # (2, 29, 736, 1280)
        if t == self.num_frames:
            num_stacks = self.num_stacks
            overlapped_frames = frames.unfold(
                1, self.encoder_2d_frames, self.overlap_step
            )  # (2, 5, 736, 1280, 9)
            overlapped_frames = overlapped_frames.permute((0, 1, 4, 2, 3))  # (2, 5, 9, 736, 1280)
            overlapped_frames = overlapped_frames[:, :, ::self.encoder_2d_step]  # (2, 5, 5, 736, 1280)
        elif t == self.stack_size:
            num_stacks = 1
            overlapped_frames = frames.view(b, 1, t, h, w)
        else:
            raise RuntimeError(
                f"Number of frames must be equal to {self.num_frames} or {self.stack_size}"
            )
        return overlapped_frames, num_stacks

    def forward_2d(self, frames):
        b, _, h, w = frames.shape  # (2, 29, 736, 1280)
        frames, num_stacks = self.overlap_frames(frames)
        frames = frames.reshape(
            b * num_stacks, self.stack_size, h, w
        )  # (10, 5, 736, 1280)
        conv2d_features = self.conv2d_encoder(frames)[-1]  # (10, 192, 23, 40)
        conv2d_features = self.conv2d_projection(conv2d_features)  # (10, 192, 23, 40)
        _, _, h, w = conv2d_features.shape
        conv2d_features = conv2d_features.contiguous().view(
            b, num_stacks, self.num_3d_features, h, w
        )  # (2, 5, 192, 23, 40)
        return conv2d_features

    def forward_3d(self, conv2d_features):
        b, t, c, h, w = conv2d_features.shape  # (2, 5, 192, 23, 40)
        assert c == self.num_3d_features and t == self.num_stacks
        conv2d_features = conv2d_features.transpose(1, 2)  # (2, 192, 5, 23, 40)
        conv3d_features = self.conv3d_encoder(conv2d_features)  # (2, 192, 5, 23, 40)
        conv3d_features = conv3d_features.transpose(1, 2)  # (2, 5, 192, 23, 40)
        conv3d_features = conv3d_features.reshape(b * t, c, h, w)  # (10, 192, 23, 40)
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
        if x.shape[1] != self.num_frames:
            raise RuntimeError(f"Number of frames must be equal to {self.num_frames}")
        x = self.forward_2d(x)
        x = self.forward_3d(x)
        x = self.forward_head(x)
        return x
