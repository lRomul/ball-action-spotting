import json
import argparse

from SoccerNet.Evaluation.ActionSpotting import evaluate

from src.ball_action import constants


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

    results = evaluate(
        SoccerNet_path=constants.ball_action_soccernet_dir,
        Predictions_path=str(predictions_path),
        split="valid" if split == "val" else split,
        version=2,
        prediction_file="results_spotting.json",
        metric="tight",
        num_classes=constants.num_classes,
        label_files='Labels-ball.json',
        dataset="Ball",
        framerate=25,
    )

    print("tight Average mAP: ", results["a_mAP"])
    print("tight Average mAP per class: ", results["a_mAP_per_class"])

    evaluate_results_path = predictions_path / "evaluate_results.json"
    print("Evaluate results saved to", evaluate_results_path)
    with open(evaluate_results_path, "w") as outfile:
        json.dump(results, outfile, indent=4)
    print("Results:", results)


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_predictions(args.experiment, args.split)
