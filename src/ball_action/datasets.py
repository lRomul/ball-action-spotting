import json

import torch
from torch.utils.data import Dataset

from src.utils import get_video_info, set_random_seed
from src.ball_action.samplers import ActionBallSampler
from src.ball_action import constants


def get_game_videos_data(game: str, resolution="720p") -> list[dict]:
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
        assert isinstance(video_data["fps"], float | int)
        frame_index = round(float(annotation["position"]) * video_data["fps"] * 0.001)
        assert isinstance(video_data["frame_index2action"], dict)
        video_data["frame_index2action"][frame_index] = annotation["label"]

    return list(half2video_data.values())


def get_videos_data(games: list[str], resolution="720p") -> list[dict]:
    games_data = list()
    for game in games:
        games_data += get_game_videos_data(game, resolution=resolution)
    return games_data


class BallActionDataset(Dataset):
    def __init__(self,
                 games: list[str],
                 sampler: ActionBallSampler,
                 resolution: str = "720p"):
        self.videos_data = get_videos_data(games, resolution=resolution)
        self.sampler = sampler
        self.sampler.init_data(self.videos_data)

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        set_random_seed(index)
        frames, targets = self.sampler.sample(index)
        return frames, targets
