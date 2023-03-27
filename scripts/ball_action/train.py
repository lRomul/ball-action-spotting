import json
import argparse
import multiprocessing
from pathlib import Path

import torch

from argus.callbacks import (
    Checkpoint,
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    LambdaLR,
)

from src.ball_action.data_loaders import RandomSeekDataLoader, SequentialDataLoader
from src.ball_action.datasets import TrainActionBallDataset, ValActionBallDataset
from src.ball_action.indexes import StackIndexesGenerator, FrameIndexShaker
from src.ball_action.augmentations import get_train_augmentations
from src.ball_action.metrics import AveragePrecision, Accuracy
from src.ball_action.target import MaxWindowTargetsProcessor
from src.ball_action.argus_models import BallActionModel
from src.ball_action.annotations import get_videos_data
from src.ema import ModelEma, EmaCheckpoint
from src.frames import get_frames_processor
from src.ball_action import constants


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=str)
args = parser.parse_args()


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 4)


IMAGE_SIZE = (1280, 736)
BATCH_SIZE = 4
BASE_LR = 3e-4
FRAME_STACK_SIZE = 15
FRAME_STACK_STEP = 2
CONFIG = dict(
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    base_lr=BASE_LR,
    min_base_lr=BASE_LR * 0.01,
    use_ema=True,
    ema_decay=0.999,
    frame_stack_size=FRAME_STACK_SIZE,
    frame_stack_step=FRAME_STACK_STEP,
    max_targets_window_size=15,
    train_epoch_size=6000,
    train_action_prob=0.5,
    train_action_random_shift=4,
    metric_accuracy_threshold=0.5,
    num_nvenc_workers=3,
    num_opencv_workers=1,
    num_epochs=[6, 30],
    stages=["warmup", "train"],
    experiments_dir=str(constants.experiments_dir / args.experiment),
    argus_params={
        "nn_module": ("multidim_stacker", {
            "model_name": "tf_efficientnetv2_b0",
            "num_classes": constants.num_classes,
            "num_frames": FRAME_STACK_SIZE,
            "stack_size": 3,
            "index_2d_features": 4,
            "pretrained": True,
            "num_3d_blocks": 4,
            "num_3d_features": 192,
            "expansion_3d_ratio": 3,
            "se_reduce_3d_ratio": 24,
            "num_3d_stack_proj": 256,
            "drop_rate": 0.3,
            "drop_path_rate": 0.1,
            "act_layer": "silu",
        }),
        "loss": "BCEWithLogitsLoss",
        "optimizer": ("AdamW", {
            "lr": get_lr(BASE_LR, BATCH_SIZE),
            "weight_decay": 0.05,
        }),
        "device": [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        "image_size": IMAGE_SIZE,
        "frame_stack_size": FRAME_STACK_SIZE,
        "frame_stack_step": FRAME_STACK_STEP,
        "amp": True,
        "iter_size": 1,
        "frames_processor": ("pad_normalize", {
            "size": IMAGE_SIZE,
            "pad_mode": "constant",
            "fill_value": 0,
        }),
    },
    frame_index_shaker={
        "shifts": [-1, 0, 1],
        "weights": [0.2, 0.6, 0.2],
        "prob": 0.25,
    },
)


def train_ball_action(config: dict, save_dir: Path):
    model = BallActionModel(config["argus_params"])
    if "pretrained" in model.params["nn_module"][1]:
        model.params["nn_module"][1]["pretrained"] = False

    augmentations = get_train_augmentations(config["image_size"])
    model.augmentations = augmentations

    targets_processor = MaxWindowTargetsProcessor(
        window_size=config["max_targets_window_size"]
    )
    frames_processor = get_frames_processor(*config["argus_params"]["frames_processor"])
    indexes_generator = StackIndexesGenerator(
        config["frame_stack_size"],
        config["frame_stack_step"],
    )
    frame_index_shaker = FrameIndexShaker(**config["frame_index_shaker"])

    if config["use_ema"]:
        ema_decay = config["ema_decay"]
        print(f"EMA decay: {ema_decay}")
        model.model_ema = ModelEma(model.nn_module, decay=ema_decay)
        checkpoint = EmaCheckpoint
    else:
        checkpoint = Checkpoint

    device = torch.device(config["argus_params"]["device"][0])
    train_data = get_videos_data(constants.train_games)
    train_dataset = TrainActionBallDataset(
        train_data,
        indexes_generator=indexes_generator,
        epoch_size=config["train_epoch_size"],
        action_prob=config["train_action_prob"],
        action_random_shift=config["train_action_random_shift"],
        target_process_fn=targets_processor,
        frames_process_fn=frames_processor,
        frame_index_shaker=frame_index_shaker,
    )
    print(f"Train dataset len {len(train_dataset)}")
    val_data = get_videos_data(constants.val_games, add_empty_actions=True)
    val_dataset = ValActionBallDataset(
        val_data,
        indexes_generator=indexes_generator,
        target_process_fn=targets_processor,
        frames_process_fn=frames_processor,
    )
    print(f"Val dataset len {len(val_dataset)}")
    train_loader = RandomSeekDataLoader(train_dataset,
                                        batch_size=config["batch_size"],
                                        num_nvenc_workers=config["num_nvenc_workers"],
                                        num_opencv_workers=config["num_opencv_workers"],
                                        gpu_id=device.index)
    val_loader = SequentialDataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        frame_buffer_size=config["frame_stack_size"] * config["frame_stack_step"],
        gpu_id=device.index
    )

    for num_epochs, stage in zip(config["num_epochs"], config["stages"]):
        callbacks = [
            LoggingToFile(save_dir / "log.txt", append=True),
            LoggingToCSV(save_dir / "log.csv", append=True),
        ]

        num_iterations = (len(train_dataset) // config["batch_size"]) * num_epochs
        if stage == "train":
            checkpoint_format = "model-{epoch:03d}-{val_average_precision:.6f}.pth"
            callbacks += [
                checkpoint(save_dir, file_format=checkpoint_format, max_saves=1),
                CosineAnnealingLR(
                    T_max=num_iterations,
                    eta_min=get_lr(config["min_base_lr"], config["batch_size"]),
                    step_on_iteration=True
                ),
            ]
        elif stage == "warmup":
            callbacks += [
                LambdaLR(lambda x: x / num_iterations,
                         step_on_iteration=True),
            ]

        metrics = [
            AveragePrecision(),
            Accuracy(threshold=config["metric_accuracy_threshold"]),
        ]

        model.fit(train_loader,
                  val_loader=val_loader,
                  num_epochs=num_epochs,
                  callbacks=callbacks,
                  metrics=metrics)

    train_loader.stop_workers()
    val_loader.stop_workers()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    experiments_dir = Path(CONFIG["experiments_dir"])
    print("Experiment dir", experiments_dir)
    if not experiments_dir.exists():
        experiments_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {experiments_dir} already exists.")

    with open(experiments_dir / "source.py", "w") as outfile:
        outfile.write(open(__file__).read())

    print("Experiment config", CONFIG)
    with open(experiments_dir / "config.json", "w") as outfile:
        json.dump(CONFIG, outfile, indent=4)

    train_ball_action(CONFIG, experiments_dir)
