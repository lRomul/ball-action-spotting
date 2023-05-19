from src.action import constants
from src.utils import get_lr


image_size = (1280, 736)
batch_size = 4
base_lr = 3e-4
frame_stack_size = 15

config = dict(
    image_size=image_size,
    batch_size=batch_size,
    base_lr=base_lr,
    min_base_lr=base_lr * 0.01,
    ema_decay=0.9995,
    max_targets_window_size=15,
    train_epoch_size=36000,
    train_sampling_weights=dict(
        action_window_size=9,
        action_prob=0.5,
        action_weights={
            "Penalty": 0.244,
            "Kick-off": 0.197,
            "Goal": 0.08,
            "Substitution": 0.06,
            "Offside": 0.069,
            "Shots on target": 0.028,
            "Shots off target": 0.03,
            "Clearance": 0.041,
            "Ball out of play": 0.011,
            "Throw-in": 0.015,
            "Foul": 0.017,
            "Indirect free-kick": 0.028,
            "Direct free-kick": 0.077,
            "Corner": 0.035,
            "Card": 0.07,
        }
    ),
    metric_accuracy_threshold=0.5,
    num_nvdec_workers=3,
    num_opencv_workers=1,
    num_epochs=[4, 20],
    stages=["warmup", "train"],
    argus_params={
        "nn_module": ("multidim_stacker", {
            "model_name": "tf_efficientnetv2_b0.in1k",
            "num_classes": constants.num_classes,
            "num_frames": frame_stack_size,
            "stack_size": 3,
            "index_2d_features": 4,
            "pretrained": True,
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
            "alpha": -1.0,
            "gamma": 1.2,
            "reduction": "mean",
        }),
        "optimizer": ("AdamW", {
            "lr": get_lr(base_lr, batch_size),
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
        "freeze_conv2d_encoder": False,
    },
    frame_index_shaker={
        "shifts": [-1, 0, 1],
        "weights": [0.2, 0.6, 0.2],
        "prob": 0.25,
    },
    torch_compile={
        "backend": "inductor",
        "mode": "default",
    },
)
