import json

from src.utils import get_video_info
from src.ball_action import constants


def get_game_data(game, resolution="720p"):
    assert resolution in {"224p", "720p"}

    game_dir = constants.ball_action_soccernet_dir / game
    labels_json_path = game_dir / "Labels-ball.json"
    with open(labels_json_path) as file:
        labels = json.load(file)

    game_data = labels["annotations"]

    halves = set()
    for annotation in game_data:
        half = int(annotation["gameTime"].split(" - ")[0])
        halves.add(half)
        annotation["half"] = half
    halves = sorted(halves)

    half2video_info = dict()
    for half in halves:
        half_video_path = str(game_dir / f"{half}_{resolution}.mkv")
        half2video_info[half] = dict(
            path=half_video_path,
            **get_video_info(half_video_path)
        )

    for annotation in game_data:
        video_info = half2video_info[annotation["half"]]
        frame_number = float(annotation["position"]) * video_info["fps"] * 0.001
        annotation["video_info"] = video_info
        annotation["frame_number"] = round(frame_number)

    return game_data


def get_games_data(games, resolution="720p"):
    games_data = list()
    for game in games:
        games_data += get_game_data(game, resolution=resolution)
    return games_data
