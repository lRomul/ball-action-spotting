import json
import argparse

import numpy as np

from src.evaluate import evaluate
from src.action import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--split", default="test", type=str)
    return parser.parse_args()


def evaluate_predictions(experiment: str, split: str):
    assert split in {"train", "val", "test"}
    print(f"Evaluate predictions: {experiment=}, {split=}")
    predictions_path = constants.predictions_dir / experiment / split
    print("Predictions path", predictions_path)
    games = constants.split2games[split]
    print("Evaluate games", games)

    results = evaluate(
        SoccerNet_path=constants.soccernet_dir,
        Predictions_path=str(predictions_path),
        list_games=games,
        prediction_file="results_spotting.json",
        version=2,
        metric="tight",
        num_classes=17,
        label_files='Labels-v2.json',
        dataset="SoccerNet",
        framerate=25,
    )

    print("Average mAP (tight): ", results["a_mAP"])
    print("Average mAP (tight) per class: ", results["a_mAP_per_class"])

    evaluate_results_path = predictions_path / "evaluate_results.json"
    results = {key: (float(value) if np.isscalar(value) else list(value))
               for key, value in results.items()}
    with open(evaluate_results_path, "w") as outfile:
        json.dump(results, outfile, indent=4)
    print("Evaluate results saved to", evaluate_results_path)
    print("Results:", results)


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_predictions(args.experiment, args.split)
