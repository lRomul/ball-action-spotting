"""
3D version of inverted residual block.
Ported from 2D implementation:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_efficientnet_blocks.py
"""

from torch import nn

from timm.models.layers import DropPath


class BatchNormAct3d(nn.Module):
    def __init__(self,
                 num_features: int,
                 act_layer=nn.ReLU,
                 apply_act: bool = True,
                 inplace_act: bool = True):
        super().__init__()
        self.bn3d = nn.BatchNorm3d(num_features)
        if apply_act:
            self.act = act_layer(inplace=inplace_act)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.bn3d(x)
        x = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self,
                 in_features: int,
                 reduce_ratio: int = 8,
                 act_layer=nn.ReLU,
                 gate_layer=nn.Sigmoid):
        super().__init__()
        rd_channels = in_features // reduce_ratio
        self.conv_reduce = nn.Conv3d(in_features, rd_channels, (1, 1, 1), bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv3d(rd_channels, in_features, (1, 1, 1), bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class InvertedResidual3d(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 expansion_ratio: int = 6,
                 se_reduce_ratio: int = 24,
                 act_layer=nn.ReLU,
                 drop_path_rate: float = 0.,
                 bias: bool = False):
        super().__init__()
        mid_features = in_features * expansion_ratio

        # Point-wise expansion
        self.conv_pw = nn.Conv3d(in_features, mid_features, (1, 1, 1), bias=bias)
        self.bn1 = BatchNormAct3d(mid_features, act_layer=act_layer)

        # Depth-wise convolution
        self.conv_dw = nn.Conv3d(mid_features, mid_features,
                                 kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                 dilation=(1, 1, 1), padding=(1, 1, 1),
                                 groups=mid_features, bias=bias)
        self.bn2 = BatchNormAct3d(mid_features, act_layer=act_layer)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_features, act_layer=act_layer, reduce_ratio=se_reduce_ratio)

        # Point-wise linear projection
        self.conv_pwl = nn.Conv3d(mid_features, out_features, (1, 1, 1), bias=bias)
        self.bn3 = BatchNormAct3d(out_features, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        x = self.drop_path(x) + shortcut
        return x
