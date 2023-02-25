import json

from src.utils import get_video_info
from src.ball_action import constants


def get_game_videos_data(game: str,
                         resolution="720p",
                         add_empty_actions: bool = False) -> list[dict]:
    assert resolution in {"224p", "720p"}

    game_dir = constants.ball_action_soccernet_dir / game
    labels_json_path = game_dir / "Labels-ball.json"
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
            half=half,
            **get_video_info(half_video_path),
            frame_index2action=dict(),
        )

    for annotation in annotations:
        video_data = half2video_data[annotation["half"]]
        frame_index = round(float(annotation["position"]) * video_data["fps"] * 0.001)
        video_data["frame_index2action"][frame_index] = annotation["label"]

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
                    add_empty_actions: bool = False) -> list[dict]:
    games_data = list()
    for game in games:
        games_data += get_game_videos_data(
            game,
            resolution=resolution,
            add_empty_actions=add_empty_actions
        )
    return games_data
