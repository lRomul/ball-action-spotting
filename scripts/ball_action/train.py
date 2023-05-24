import time
import json
import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint
from importlib.machinery import SourceFileLoader

import torch
import torch._dynamo

from argus import load_model
from argus.callbacks import (
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    LambdaLR,
)

from src.ball_action.annotations import get_videos_data, get_videos_sampling_weights
from src.utils import load_weights_from_pretrain, get_best_model_path, get_lr
from src.data_loaders import RandomSeekDataLoader, SequentialDataLoader
from src.ball_action.augmentations import get_train_augmentations
from src.indexes import StackIndexesGenerator, FrameIndexShaker
from src.datasets import TrainActionDataset, ValActionDataset
from src.metrics import AveragePrecision, Accuracy
from src.target import MaxWindowTargetsProcessor
from src.argus_models import BallActionModel
from src.ema import ModelEma, EmaCheckpoint
from src.frames import get_frames_processor
from src.ball_action import constants
from src.mixup import TimmMixup

from src.action.constants import experiments_dir as action_experiments_dir


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    return parser.parse_args()


def train_ball_action(config: dict, save_dir: Path,
                      train_games: list[str], val_games: list[str]):
    argus_params = config["argus_params"]
    model = BallActionModel(argus_params)
    if "pretrained" in model.params["nn_module"][1]:
        model.params["nn_module"][1]["pretrained"] = False

    pretrain_dir = ""
    if config["pretrain_action_experiment"]:
        pretrain_dir = action_experiments_dir / config["pretrain_action_experiment"]
    elif config["pretrain_ball_experiment"]:
        pretrain_dir = constants.experiments_dir / config["pretrain_ball_experiment"] / f"fold_{fold}"

    if pretrain_dir:
        pretrain_model_path = get_best_model_path(pretrain_dir)
        print(f"Load pretrain model: {pretrain_model_path}")
        pretrain_model = load_model(pretrain_model_path, device=argus_params["device"])
        load_weights_from_pretrain(model.nn_module, pretrain_model.nn_module)
        del pretrain_model

    augmentations = get_train_augmentations(config["image_size"])
    model.augmentations = augmentations

    if "mixup_params" in config:
        model.mixup = TimmMixup(**config["mixup_params"])

    targets_processor = MaxWindowTargetsProcessor(
        window_size=config["max_targets_window_size"]
    )
    frames_processor = get_frames_processor(*argus_params["frames_processor"])
    indexes_generator = StackIndexesGenerator(
        argus_params["frame_stack_size"],
        argus_params["frame_stack_step"],
    )
    frame_index_shaker = FrameIndexShaker(**config["frame_index_shaker"])

    print("EMA decay:", config["ema_decay"])
    model.model_ema = ModelEma(model.nn_module, decay=config["ema_decay"])

    if "torch_compile" in config:
        print("torch.compile:", config["torch_compile"])
        torch._dynamo.reset()
        model.nn_module = torch.compile(model.nn_module, **config["torch_compile"])

    device = torch.device(argus_params["device"][0])
    train_data = get_videos_data(train_games)
    videos_sampling_weights = get_videos_sampling_weights(
        train_data, **config["train_sampling_weights"],
    )
    train_dataset = TrainActionDataset(
        train_data,
        constants.classes,
        indexes_generator=indexes_generator,
        epoch_size=config["train_epoch_size"],
        videos_sampling_weights=videos_sampling_weights,
        target_process_fn=targets_processor,
        frames_process_fn=frames_processor,
        frame_index_shaker=frame_index_shaker,
    )
    print(f"Train dataset len {len(train_dataset)}")
    val_data = get_videos_data(val_games, add_empty_actions=True)
    val_dataset = ValActionDataset(
        val_data,
        constants.classes,
        indexes_generator=indexes_generator,
        target_process_fn=targets_processor,
        frames_process_fn=frames_processor,
    )
    print(f"Val dataset len {len(val_dataset)}")
    train_loader = RandomSeekDataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_nvdec_workers=config["num_nvdec_workers"],
        num_opencv_workers=config["num_opencv_workers"],
        gpu_id=device.index,
    )
    val_loader = SequentialDataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        frame_buffer_size=argus_params["frame_stack_size"] * argus_params["frame_stack_step"],
        gpu_id=device.index,
    )

    for num_epochs, stage in zip(config["num_epochs"], config["stages"]):
        callbacks = [
            LoggingToFile(save_dir / "log.txt", append=True),
            LoggingToCSV(save_dir / "log.csv", append=True),
        ]

        num_iterations = (len(train_dataset) // config["batch_size"]) * num_epochs
        if stage == "warmup":
            callbacks += [
                LambdaLR(lambda x: x / num_iterations,
                         step_on_iteration=True),
            ]

            model.fit(train_loader,
                      num_epochs=num_epochs,
                      callbacks=callbacks)
        elif stage == "train":
            checkpoint_format = "model-{epoch:03d}-{val_average_precision:.6f}.pth"
            callbacks += [
                EmaCheckpoint(save_dir, file_format=checkpoint_format, max_saves=1),
                CosineAnnealingLR(
                    T_max=num_iterations,
                    eta_min=get_lr(config["min_base_lr"], config["batch_size"]),
                    step_on_iteration=True,
                ),
            ]

            metrics = [
                AveragePrecision(constants.classes),
                Accuracy(constants.classes, threshold=config["metric_accuracy_threshold"]),
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
    args = parse_arguments()
    print("Experiment:", args.experiment)

    config_path = constants.configs_dir / f"{args.experiment}.py"
    if not config_path.exists():
        raise RuntimeError(f"Config '{config_path}' is not exists")

    config = SourceFileLoader(args.experiment, str(config_path)).load_module().config
    print("Experiment config:")
    pprint(config, sort_dicts=False)

    experiments_dir = constants.experiments_dir / args.experiment
    print("Experiment dir:", experiments_dir)
    if not experiments_dir.exists():
        experiments_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder '{experiments_dir}' already exists.")

    with open(experiments_dir / "train.py", "w") as outfile:
        outfile.write(open(__file__).read())

    with open(experiments_dir / "config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)

    if args.folds == "all":
        folds = constants.folds
    else:
        folds = [int(fold) for fold in args.folds.split(",")]

    for fold in folds:
        train_folds = list(set(constants.folds) - {fold})
        val_games = constants.fold2games[fold]
        train_games = []
        for train_fold in train_folds:
            train_games += constants.fold2games[train_fold]
        fold_experiment_dir = experiments_dir / f"fold_{fold}"
        print(f"Val fold: {fold}, train folds: {train_folds}")
        print(f"Val games: {val_games}, train games: {train_games}")
        print(f"Fold experiment dir: {fold_experiment_dir}")
        train_ball_action(config, fold_experiment_dir, train_games, val_games)

        torch._dynamo.reset()
        torch.cuda.empty_cache()
        time.sleep(12)
