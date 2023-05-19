from src.ball_action import constants
from src.utils import get_lr


image_size = (1280, 736)
batch_size = 4
base_lr = 1e-3
frame_stack_size = 33

config = dict(
    image_size=image_size,
    batch_size=batch_size,
    base_lr=base_lr,
    min_base_lr=base_lr * 0.05,
    ema_decay=0.999,
    max_targets_window_size=15,
    train_epoch_size=6000,
    train_sampling_weights=dict(
        action_window_size=9,
        action_prob=0.5,
        pred_experiment="sampling_weights_001",
        clear_pred_window_size=27,
    ),
    metric_accuracy_threshold=0.5,
    num_nvdec_workers=3,
    num_opencv_workers=1,
    num_epochs=[2, 7],
    stages=["warmup", "train"],
    argus_params={
        "nn_module": ("multidim_stacker", {
            "model_name": "tf_efficientnetv2_b0.in1k",
            "num_classes": constants.num_classes,
            "num_frames": frame_stack_size,
            "stack_size": 3,
            "index_2d_features": 4,
            "pretrained": False,
            "num_3d_blocks": 4,
            "num_3d_features": 192,
            "expansion_3d_ratio": 3,
            "se_reduce_3d_ratio": 24,
            "num_3d_stack_proj": 256,
            "drop_rate": 0.2,
            "drop_path_rate": 0.2,
            "act_layer": "silu",
        }),
        "loss": ("focal_loss", {
            "alpha": 0.4,
            "gamma": 1.2,
            "reduction": "mean",
        }),
        "optimizer": ("SGD", {
            "lr": get_lr(base_lr, batch_size),
            "momentum": 0.9,
            "nesterov": True,
        }),
        "device": ["cuda:0"],
        "image_size": image_size,
        "frame_stack_size": frame_stack_size,
        "frame_stack_step": 2,
        "amp": True,
        "iter_size": 1,
        "frames_processor": ("pad_normalize", {
            "size": image_size,
            "pad_mode": "constant",
            "fill_value": 0,
        }),
        "freeze_conv2d_encoder": True,
    },
    frame_index_shaker={
        "shifts": [-1, 0, 1],
        "weights": [0.2, 0.6, 0.2],
        "prob": 0.25,
    },
    pretrain_action_experiment="",
    pretrain_ball_experiment="ball_tuning_001",
    torch_compile={
        "backend": "inductor",
        "mode": "default",
    },
)
