import timm

from .multidim_stacker import MultiDimStacker


class MultiChanStacker(MultiDimStacker):
    """MultiDimStacker modification for RGB frames (or with any number of channels).
    This architecture is not working in the pipeline because the pipeline operates on grayscale frames.
    """
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 num_frames: int = 15,
                 num_chans: int = 3,
                 stack_size: int = 3,
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
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            num_frames=num_frames,
            stack_size=stack_size,
            index_2d_features=index_2d_features,
            pretrained=pretrained,
            num_3d_blocks=num_3d_blocks,
            num_3d_features=num_3d_features,
            num_3d_stack_proj=num_3d_stack_proj,
            expansion_3d_ratio=expansion_3d_ratio,
            se_reduce_3d_ratio=se_reduce_3d_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            act_layer=act_layer,
            **kwargs
        )
        self.num_chans = num_chans
        self.conv2d_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=stack_size * num_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            features_only=True,
            out_indices=[index_2d_features],
            **kwargs
        )

    def forward_2d(self, x):
        b, t, c, h, w = x.shape  # (2, 15, 3, 736, 1280)
        assert t % self.stack_size == 0 and c == self.num_chans
        num_stacks = t // self.stack_size
        x = x.view(b * num_stacks, self.stack_size * self.num_chans, h, w)  # (10, 9, 736, 1280)
        x = self.conv2d_encoder(x)[-1]  # (10, 192, 23, 40)
        x = self.conv2d_projection(x).contiguous()  # (10, 192, 23, 40)
        _, _, h, w = x.shape
        x = x.view(b, num_stacks, self.num_3d_features, h, w)  # (2, 5, 192, 23, 40)
        return x
