from SoccerNet.utils import getListGames

from src.constants import data_dir, soccernet_dir

action_dir = data_dir / "action"
experiments_dir = action_dir / "experiments"
predictions_dir = action_dir / "predictions"
visualizations_dir = action_dir / "visualizations"

action_soccernet_dir = soccernet_dir / "action-spotting-2023"

fold2games = {
    0: getListGames(split="test", task="spotting", dataset="SoccerNet"),
    1: getListGames(split="valid", task="spotting", dataset="SoccerNet"),
    2: getListGames(split="train", task="spotting", dataset="SoccerNet"),
}
game2fold = {game: fold for fold, games in fold2games.items() for game in games}
folds = sorted(fold2games.keys())
challenge_games = getListGames(split="challenge", task="spotting", dataset="SoccerNet")

classes = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card"
]

num_classes = len(classes)
target2class: dict[int, str] = {trg: cls for trg, cls in enumerate(classes)}
class2target: dict[str, int] = {cls: trg for trg, cls in enumerate(classes)}

num_halves = 2
halves = list(range(1, num_halves + 1))
postprocess_params = {
    "gauss_sigma": 3.0,
    "height": 0.2,
    "distance": 15,
}

video_fps = 25.0
