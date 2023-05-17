import json
import argparse

import numpy as np

from src.evaluate import evaluate
from src.ball_action import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    return parser.parse_args()


def evaluate_predictions(experiment: str, fold: int):
    print(f"Evaluate predictions: {experiment=}, {fold=}")
    predictions_path = constants.predictions_dir / experiment / "cv" / f"fold_{fold}"
    print("Predictions path", predictions_path)
    games = constants.fold2games[fold]
    print("Evaluate games", games)

    results = evaluate(
        SoccerNet_path=constants.soccernet_dir,
        Predictions_path=str(predictions_path),
        list_games=games,
        prediction_file="results_spotting.json",
        version=2,
        metric="at1",
        num_classes=constants.num_classes,
        label_files='Labels-ball.json',
        dataset="Ball",
        framerate=25,
    )

    print("Average mAP@1: ", results["a_mAP"])
    print("Average mAP@1 per class: ", results["a_mAP_per_class"])

    evaluate_results_path = predictions_path / "evaluate_results.json"
    results = {key: (float(value) if np.isscalar(value) else list(value))
               for key, value in results.items()}
    with open(evaluate_results_path, "w") as outfile:
        json.dump(results, outfile, indent=4)
    print("Evaluate results saved to", evaluate_results_path)
    print("Results:", results)


if __name__ == "__main__":
    args = parse_arguments()

    if args.folds == "all":
        folds = constants.folds
    else:
        folds = [int(fold) for fold in args.folds.split(",")]

    for fold in folds:
        evaluate_predictions(args.experiment, fold)
