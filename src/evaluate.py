import os
import json
import glob
import zipfile

from tqdm import tqdm
import numpy as np

from SoccerNet.Evaluation.ActionSpotting import label2vector, predictions2vector, average_mAP
from SoccerNet.Evaluation.utils import LoadJsonFromZip, EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, EVENT_DICTIONARY_BALL


def evaluate(SoccerNet_path, Predictions_path, list_games, prediction_file="results_spotting.json", version=2,
             framerate=2, metric="loose", label_files="Labels-v2.json", num_classes=17, dataset="SoccerNet"):
    # evaluate the prediction with respect to some ground truth
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - list_games: games to evaluate
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()
    if dataset == "SoccerNet" and version == 1:
        EVENT_DICTIONARY = EVENT_DICTIONARY_V1
    elif dataset == "SoccerNet" and version == 2:
        EVENT_DICTIONARY = EVENT_DICTIONARY_V2
    elif dataset == "Headers":
        EVENT_DICTIONARY = {"Header": 0}
    elif dataset == "Headers-headimpacttype":
        EVENT_DICTIONARY = {"1. Purposeful header": 0, "2. Header Duel": 1,
                            "3. Attempted header": 2, "4. Unintentional header": 3, "5. Other head impacts": 4}
    elif dataset == "Ball":
        EVENT_DICTIONARY = EVENT_DICTIONARY_BALL

    for game in tqdm(list_games):

        # # Load labels
        # if version==2:
        #     label_files = "Labels-v2.json"
        #     num_classes = 17
        # elif version==1:
        #     label_files = "Labels.json"
        #     num_classes = 3
        # if dataset == "Headers":
        #     label_files = "Labels-Header.json"
        #     num_classes = 3

        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, label_files))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        # convert labels to vector
        label_half_1, label_half_2 = label2vector(
            labels, num_classes=num_classes, version=version, EVENT_DICTIONARY=EVENT_DICTIONARY, framerate=framerate)
        # print(version)
        # print(label_half_1)
        # print(label_half_2)

        # infer name of the prediction_file
        if prediction_file is None:
            if zipfile.is_zipfile(Predictions_path):
                with zipfile.ZipFile(Predictions_path, "r") as z:
                    for filename in z.namelist():
                        #       print(filename)
                        if filename.endswith(".json"):
                            prediction_file = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(os.path.join(Predictions_path, "*/*/*/*.json")):
                    prediction_file = os.path.basename(filename)
                    # print(prediction_file)
                    break

        # Load predictions
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        # convert predictions to vector
        predictions_half_1, predictions_half_2 = predictions2vector(
            predictions, num_classes=num_classes, version=version, EVENT_DICTIONARY=EVENT_DICTIONARY,
            framerate=framerate)

        targets_numpy.append(label_half_1)
        targets_numpy.append(label_half_2)
        detections_numpy.append(predictions_half_1)
        detections_numpy.append(predictions_half_2)

        closest_numpy = np.zeros(label_half_1.shape) - 1
        # Get the closest action index
        for c in np.arange(label_half_1.shape[-1]):
            indexes = np.where(label_half_1[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = label_half_1[indexes[i], c]
        closests_numpy.append(closest_numpy)

        closest_numpy = np.zeros(label_half_2.shape) - 1
        for c in np.arange(label_half_2.shape[-1]):
            indexes = np.where(label_half_2[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = label_half_2[indexes[i], c]
        closests_numpy.append(closest_numpy)

    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])  # np.arange(1)*1 + 1
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])
        # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version == 2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version == 2 else None,
        "a_mAP_unshown": a_mAP_unshown if version == 2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version == 2 else None,
    }
    return results
