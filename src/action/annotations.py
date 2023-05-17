import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import maximum_filter

from src.utils import get_video_info, post_processing
from src.action import constants


def get_game_videos_data(game: str,
                         resolution="720p",
                         only_visible=True,
                         add_empty_actions: bool = False) -> list[dict]:
    assert resolution in {"224p", "720p"}

    game_dir = constants.soccernet_dir / game
    labels_json_path = game_dir / "Labels-v2.json"
    with open(labels_json_path) as file:
        labels = json.load(file)

    annotations = labels["annotations"]

    halves_set = set()
    for annotation in annotations:
        half = int(annotation["gameTime"].split(" - ")[0])
        halves_set.add(half)
        annotation["half"] = half
    halves = sorted(halves_set)

    half2video_data = dict()
    for half in halves:
        half_video_path = str(game_dir / f"{half}_{resolution}.mkv")
        half2video_data[half] = dict(
            video_path=half_video_path,
            game=game,
            half=half,
            **get_video_info(half_video_path),
            frame_index2action=dict(),
        )

    for annotation in annotations:
        if only_visible and annotation["visibility"] != "visible":
            continue
        video_data = half2video_data[annotation["half"]]
        frame_index = round(float(annotation["position"]) * video_data["fps"] * 0.001)
        label = annotation["label"]
        if label in constants.card_classes:
            video_data["frame_index2action"][frame_index] = "Card"
        else:
            video_data["frame_index2action"][frame_index] = label

    if add_empty_actions:
        for half in halves:
            video_data = half2video_data[half]
            prev_frame_index = -1
            for frame_index in sorted(video_data["frame_index2action"].keys()):
                if prev_frame_index != -1:
                    empty_frame_index = (prev_frame_index + frame_index) // 2
                    if empty_frame_index not in video_data["frame_index2action"]:
                        video_data["frame_index2action"][empty_frame_index] = "EMPTY"
                prev_frame_index = frame_index

    return list(half2video_data.values())


def get_videos_data(games: list[str],
                    resolution="720p",
                    only_visible=True,
                    add_empty_actions: bool = False) -> list[dict]:
    games_data = list()
    for game in games:
        games_data += get_game_videos_data(
            game,
            resolution=resolution,
            only_visible=only_visible,
            add_empty_actions=add_empty_actions
        )
    return games_data


def raw_predictions_to_actions(frame_indexes: list[int], raw_predictions: np.ndarray):
    class2actions = dict()
    for cls, cls_index in constants.class2target.items():
        class2actions[cls] = post_processing(
            frame_indexes, raw_predictions[:, cls_index], **constants.postprocess_params
        )
        print(f"Predicted {len(class2actions[cls][0])} {cls} actions")
    return class2actions


def prepare_game_spotting_results(half2class_actions: dict, game: str, prediction_dir: Path):
    game_prediction_dir = prediction_dir / game
    game_prediction_dir.mkdir(parents=True, exist_ok=True)

    results_spotting = {
        "UrlLocal": game,
        "predictions": list(),
    }

    for half in half2class_actions.keys():
        for cls, (frame_indexes, confidences) in half2class_actions[half].items():
            cls = "Yellow card" if cls == "Card" else cls
            for frame_index, confidence in zip(frame_indexes, confidences):
                position = round(frame_index / constants.video_fps * 1000)
                seconds = int(frame_index / constants.video_fps)
                prediction = {
                    "gameTime": f"{half} - {seconds // 60:02}:{seconds % 60:02}",
                    "label": cls,
                    "position": str(position),
                    "half": str(half),
                    "confidence": str(confidence),
                }
                results_spotting["predictions"].append(prediction)
    results_spotting["predictions"] = sorted(
        results_spotting["predictions"],
        key=lambda pred: (int(pred["half"]), int(pred["position"]))
    )

    results_spotting_path = game_prediction_dir / "results_spotting.json"
    with open(results_spotting_path, "w") as outfile:
        json.dump(results_spotting, outfile, indent=4)
    print("Spotting results saved to", results_spotting_path)
    with open(game_prediction_dir / "postprocess_params.json", "w") as outfile:
        json.dump(constants.postprocess_params, outfile, indent=4)


def get_video_sampling_weights(video_data: dict,
                               action_window_size: int,
                               action_prob: float,
                               action_weights: Optional[dict] = None) -> np.ndarray:
    frame_count = video_data["frame_count"]
    weights = np.zeros(frame_count)

    for frame_index, action in video_data["frame_index2action"].items():
        if frame_index >= frame_count:
            print(f"Clip action {action} on {frame_index} frame. "
                  f"Video: {video_data['video_path']}, {frame_count=}")
            frame_index = frame_count - 1
        value = action_weights[action] if action_weights is not None else 1.0
        weights[frame_index] = max(value, weights[frame_index])

    weights = maximum_filter(weights, size=action_window_size)
    no_action_mask = weights == 0.0
    no_action_count = no_action_mask.sum()

    no_action_weights_sum = (1 - action_prob) / action_prob * weights.sum()
    weights[no_action_mask] = no_action_weights_sum / no_action_count

    weights /= weights.sum()
    return weights


def get_videos_sampling_weights(videos_data: list[dict],
                                action_window_size: int,
                                action_prob: float,
                                action_weights: Optional[dict] = None) -> list[np.ndarray]:
    videos_sampling_weights = []
    for video_data in videos_data:
        video_sampling_weights = get_video_sampling_weights(
            video_data, action_window_size, action_prob, action_weights=action_weights
        )
        videos_sampling_weights.append(video_sampling_weights)
    return videos_sampling_weights
