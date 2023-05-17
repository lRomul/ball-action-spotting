import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import cv2

from src.ball_action.annotations import get_game_videos_data
from src.utils import get_video_info, post_processing
from src.frame_fetchers import NvDecFrameFetcher
from src.target import VideoTarget
from src.ball_action import constants


RESOLUTION = "720p"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--challenge", action="store_true")
    return parser.parse_args()


def pad_to_length(lst, length):
    lst = lst[-length:]
    lst = [0.] * (length - len(lst)) + lst
    return lst


def draw_graph(targets, predictions, pred_actions, length=96, height=33, upscale=4):
    targets = pad_to_length(targets, length)
    predictions = pad_to_length(predictions, length)
    pred_actions = pad_to_length(pred_actions, length)

    graph = np.zeros((height, length, 3), dtype=np.uint8)
    for i in range(length):
        if targets[i] and pred_actions[i]:
            color = (255, 0, 255)
        elif pred_actions[i]:
            color = (255, 0, 0)
        elif targets[i]:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        pred_height = round(predictions[i] * (height - 1))
        graph[height - pred_height:-1, i] = color
        graph[-1, i] = color

    graph = cv2.resize(graph,
                       (length * upscale, height * upscale),
                       interpolation=cv2.INTER_NEAREST)
    return graph


def load_video_predictions(game_prediction_dir: Path, half: int):
    raw_predictions_path = game_prediction_dir / f"{half}_raw_predictions.npz"
    raw_predictions_npz = np.load(str(raw_predictions_path))
    frame_indexes = raw_predictions_npz["frame_indexes"]
    raw_predictions = raw_predictions_npz["raw_predictions"]
    video_prediction = defaultdict(lambda: np.zeros(2, dtype=np.float32))
    for frame_index, prediction in zip(frame_indexes, raw_predictions):
        video_prediction[frame_index] = prediction
    video_pred_actions = defaultdict(lambda: np.zeros(2, dtype=np.float32))
    for cls, cls_index in constants.class2target.items():
        action_frame_indexes, _ = post_processing(
            frame_indexes, raw_predictions[:, cls_index], **constants.postprocess_params
        )
        for frame_index in action_frame_indexes:
            video_pred_actions[frame_index][cls_index] = 1.0
    return video_prediction, video_pred_actions


def visualize_video(half: int,
                    game_dir: Path,
                    game_prediction_dir: Path,
                    game_visualization_dir: Path,
                    game_video_data: dict,
                    gpu_id: int):
    video_path = game_dir / f"{half}_{RESOLUTION}.mkv"
    visualize_video_path = game_visualization_dir / f"{half}_{RESOLUTION}.avi"
    video_target = VideoTarget(game_video_data, constants.classes)
    video_prediction, video_pred_actions = load_video_predictions(game_prediction_dir, half)

    video_info = get_video_info(video_path)
    frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=gpu_id)
    frame_fetcher.num_frames = video_info["frame_count"]
    video_writer = cv2.VideoWriter(str(visualize_video_path),
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   video_info["fps"],
                                   (video_info["width"], video_info["height"]))

    targets = {cls: [] for cls in constants.classes}
    predictions = {cls: [] for cls in constants.classes}
    pred_actions = {cls: [] for cls in constants.classes}
    for _ in tqdm(range(frame_fetcher.num_frames - 1)):
        frame = frame_fetcher.fetch_frame().cpu().numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame_index = frame_fetcher.current_index

        target = video_target.target(frame_index)
        prediction = video_prediction[frame_index]
        pred_action = video_pred_actions[frame_index]

        for cls, cls_index in constants.class2target.items():
            predictions[cls].append(prediction[cls_index])
            targets[cls].append(target[cls_index])
            pred_actions[cls].append(pred_action[cls_index])

        x = 50
        for cls, y in zip(constants.classes, (300, 500)):
            pass_graph = draw_graph(targets[cls], predictions[cls], pred_actions[cls])
            crop = frame[y: y + pass_graph.shape[0], x: x + pass_graph.shape[1]]
            cv2.addWeighted(pass_graph, 1., crop, 1., 0.0, crop)

        cv2.putText(frame, str(frame_index), (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2, cv2.LINE_AA)

        video_writer.write(frame)
    video_writer.release()


def visualize_game(game: str,
                   prediction_dir: Path,
                   visualization_dir: Path,
                   gpu_id: int,
                   challenge: bool):
    game_dir = constants.soccernet_dir / game
    game_prediction_dir = prediction_dir / game
    game_visualization_dir = visualization_dir / game
    game_visualization_dir.mkdir(parents=True, exist_ok=True)
    print("Visualize game:", game)
    if challenge:
        game_videos_data = [{"half": h, "frame_index2action": dict()} for h in constants.halves]
    else:
        game_videos_data = get_game_videos_data(game, resolution=RESOLUTION)

    for half, game_video_data in zip(constants.halves, game_videos_data):
        assert half == game_video_data["half"]
        visualize_video(half, game_dir, game_prediction_dir,
                        game_visualization_dir, game_video_data, gpu_id)


def visualize_fold(experiment: str, fold: int | str, gpu_id: int, challenge: bool):
    print(f"Visualize games: {experiment=}, {fold=}, {gpu_id=} {challenge=}")
    if challenge:
        data_split = "challenge"
        games = constants.challenge_games
        if fold == "ensemble":
            fold_dir = "ensemble"
        else:
            fold_dir = f"fold_{fold}"
    else:
        data_split = "cv"
        games = constants.fold2games[fold]
        if fold == "ensemble":
            raise ValueError("Ensemble visualization possible only with challenge")
        fold_dir = f"fold_{fold}"
    prediction_dir = constants.predictions_dir / experiment / data_split / fold_dir
    visualization_dir = constants.visualizations_dir / experiment / data_split / fold_dir

    for game in games:
        visualize_game(game, prediction_dir, visualization_dir, gpu_id, challenge)


if __name__ == "__main__":
    args = parse_arguments()

    if args.folds == "all":
        folds = constants.folds
    elif args.folds == "ensemble":
        folds = ["ensemble"]
    else:
        folds = [int(fold) for fold in args.folds.split(",")]

    for fold in folds:
        visualize_fold(args.experiment, fold, args.gpu_id, args.challenge)
