import argparse
from pathlib import Path
from pprint import pprint

import numpy as np

from src.ball_action.annotations import raw_predictions_to_actions, prepare_game_spotting_results
from src.ball_action import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", required=True, type=str)
    parser.add_argument("--challenge", action="store_true")
    return parser.parse_args()


def load_and_blend_predictions(prediction_paths: list[Path]):
    frame_indexes_lst = []
    prediction_lst = []
    for prediction_path in prediction_paths:
        with np.load(str(prediction_path)) as npz_predictions:
            frame_indexes = npz_predictions["frame_indexes"]
            predictions = npz_predictions["raw_predictions"]
        frame_indexes_lst.append(frame_indexes)
        prediction_lst.append(predictions)

    frame_indexes = np.unique(np.concatenate(frame_indexes_lst))
    blend_prediction = np.zeros((np.max(frame_indexes) + 1, constants.num_classes))
    for frame_indexes, prediction in zip(frame_indexes_lst, prediction_lst):
        blend_prediction[frame_indexes] += prediction
    blend_prediction /= len(prediction_lst)
    blend_prediction = blend_prediction[np.min(frame_indexes):]
    assert blend_prediction.shape[0] == frame_indexes.shape[0]
    return blend_prediction, frame_indexes


def ensemble_challenge_video(game: str, half: int, game_ensemble_path: Path):
    prediction_paths = []
    for experiment in experiments:
        for fold in constants.folds:
            prediction_path = (
                    constants.predictions_dir
                    / experiment
                    / "challenge"
                    / f"fold_{fold}"
                    / game
                    / f"{half}_raw_predictions.npz"
            )
            prediction_paths.append(prediction_path)
    print("Blend raw predictions:")
    pprint(prediction_paths)
    blend_prediction, frame_indexes = load_and_blend_predictions(prediction_paths)
    blend_prediction_path = game_ensemble_path / f"{half}_raw_predictions.npz"
    np.savez(
        blend_prediction_path,
        frame_indexes=frame_indexes,
        raw_predictions=blend_prediction,
    )
    print(f"Blend predictions saved to {blend_prediction_path}")

    class2actions = raw_predictions_to_actions(frame_indexes, blend_prediction)
    return class2actions


def ensemble_cv_video(fold: int, game: str, half: int, game_ensemble_path: Path):
    prediction_paths = []
    for experiment in experiments:
        prediction_path = (
                constants.predictions_dir
                / experiment
                / "cv"
                / f"fold_{fold}"
                / game
                / f"{half}_raw_predictions.npz"
        )
        prediction_paths.append(prediction_path)
    print("Blend raw predictions:")
    pprint(prediction_paths)
    blend_prediction, frame_indexes = load_and_blend_predictions(prediction_paths)
    blend_prediction_path = game_ensemble_path / f"{half}_raw_predictions.npz"
    np.savez(
        blend_prediction_path,
        frame_indexes=frame_indexes,
        raw_predictions=blend_prediction,
    )
    print(f"Blend predictions saved to {blend_prediction_path}")

    class2actions = raw_predictions_to_actions(frame_indexes, blend_prediction)
    return class2actions


def ensemble_challenge(experiments: list[str]):
    print("Ensemble challenge predictions:", experiments)
    ensemble_path = constants.predictions_dir / ",".join(experiments) / "challenge" / "ensemble"
    for game in constants.challenge_games:
        game_ensemble_path = ensemble_path / game
        game_ensemble_path.mkdir(parents=True, exist_ok=True)
        half2class_actions = dict()
        for half in constants.halves:
            class_actions = ensemble_challenge_video(game, half, game_ensemble_path)
            half2class_actions[half] = class_actions

        prepare_game_spotting_results(half2class_actions, game, ensemble_path)


def ensemble_cv(experiments: list[str]):
    assert len(experiments) > 1
    print("Ensemble cross-validation predictions:", experiments)
    ensemble_path = constants.predictions_dir / ",".join(experiments) / "cv"
    for fold in constants.folds:
        for game in constants.fold2games[fold]:
            game_ensemble_path = ensemble_path / f"fold_{fold}" / game
            game_ensemble_path.mkdir(parents=True, exist_ok=True)
            half2class_actions = dict()
            for half in constants.halves:
                class_actions = ensemble_cv_video(fold, game, half, game_ensemble_path)
                half2class_actions[half] = class_actions

            prepare_game_spotting_results(half2class_actions, game,
                                          ensemble_path / f"fold_{fold}")


if __name__ == "__main__":
    args = parse_arguments()
    experiments = sorted(args.experiments.split(','))
    if args.challenge:
        ensemble_challenge(experiments)
    else:
        ensemble_cv(experiments)
