import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score

import torch
from argus.metrics import Metric


class PerClassMetric(Metric):
    name: str = ''
    better: str = 'min'

    def __init__(self, classes: list[str]):
        self.target2class = {trg: cls for trg, cls in enumerate(classes)}
        self.predictions = []
        self.targets = []

    def reset(self):
        self.predictions = []
        self.targets = []

    @torch.no_grad()
    def update(self, step_output: dict):
        pred = step_output['prediction'].cpu().numpy()
        target = step_output['target'].cpu().numpy()

        self.predictions.append(pred)
        self.targets.append(target)

    def compute(self) -> list[float]:
        raise NotImplementedError

    def epoch_complete(self, state):
        scores = self.compute()
        name_prefix = f"{state.phase}_" if state.phase else ''
        state.metrics[f"{name_prefix}{self.name}"] = np.mean(scores)
        for trg, cls in self.target2class.items():
            state.metrics[f"{name_prefix}{self.name}_{cls.lower()}"] = scores[trg]


class AveragePrecision(PerClassMetric):
    name = 'average_precision'
    better = 'max'

    def compute(self) -> list[float]:
        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.predictions, axis=0)
        scores: list[float] = average_precision_score(y_true, y_pred, average=None)
        return scores


class Accuracy(PerClassMetric):
    name = 'binary_accuracy'
    better = 'max'

    def __init__(self, classes: list[str], threshold: float = 0.5):
        super().__init__(classes)
        self.threshold = threshold

    def compute(self) -> list[float]:
        targets = np.concatenate(self.targets, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)

        y_true = targets > self.threshold
        y_pred = predictions > self.threshold

        scores: list[float] = []
        for trg in range(y_true.shape[1]):
            scores.append(accuracy_score(y_true[:, trg], y_pred[:, trg]))
        return scores
