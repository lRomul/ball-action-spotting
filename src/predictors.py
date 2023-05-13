from pathlib import Path
from itertools import islice
from typing import Optional, Iterable

import torch
from kornia.geometry.transform import hflip

import argus

from src.indexes import StackIndexesGenerator
from src.frames import get_frames_processor


def batched(iterable: Iterable, size: int):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, size)):
        yield batch


def overlapped(frame_indexes: list[int], size: int, overlap: int, step: int) -> list[tuple[int]]:
    indexes_tensor = torch.tensor(list(frame_indexes), dtype=torch.int)
    overlapped_indexes = indexes_tensor.unfold(0, size, overlap)[:, ::step].tolist()
    return [tuple(ind) for ind in overlapped_indexes]


class MultiDimPredictor:
    def __init__(self, model_path: Path, device: str = "cuda:0", tta: bool = False):
        self.model = argus.load_model(model_path, device=device, optimizer=None, loss=None)
        self.model.eval()
        self.device = self.model.device
        self.tta = tta
        self.model_name = self.model.params["nn_module"][0]
        self.model_params = self.model.params["nn_module"][1]
        assert self.model_name in {"multidim_stacker", "multidim_overlapper"}
        self.frames_processor = get_frames_processor(*self.model.params["frames_processor"])
        self.frame_stack_size = self.model.params["frame_stack_size"]
        self.frame_stack_step = self.model.params["frame_stack_step"]
        self.indexes_generator = StackIndexesGenerator(self.frame_stack_size,
                                                       self.frame_stack_step)

        self._frame_index2frame: dict[int, torch.Tensor] = dict()
        self._stack_indexes2features: dict[tuple[int], torch.Tensor] = dict()
        self._predict_offset: int = self.indexes_generator.make_stack_indexes(0)[-1]

    def reset_buffers(self):
        self._frame_index2frame = dict()
        self._stack_indexes2features = dict()

    def _clear_old(self, minimum_index: int):
        for index in list(self._frame_index2frame.keys()):
            if index < minimum_index:
                del self._frame_index2frame[index]
        for stack_indexes in list(self._stack_indexes2features.keys()):
            if any([i < minimum_index for i in stack_indexes]):
                del self._stack_indexes2features[stack_indexes]

    def get_stacks_indexes(self, predict_indexes) -> list[tuple[int]]:
        if self.model_name == "multidim_stacker":
            stacks_indexes = list(batched(predict_indexes, self.model_params["stack_size"]))
        elif self.model_name == "multidim_overlapper":
            stacks_indexes = overlapped(
                predict_indexes,
                self.model_params["encoder_2d_frames"],
                self.model_params["overlap_step"],
                self.model_params["encoder_2d_step"],
            )
        else:
            raise RuntimeError(f" Model '{self.model_name}' is not supported")
        return stacks_indexes

    @torch.no_grad()
    def predict(self, frame: torch.Tensor, index: int) -> tuple[Optional[torch.Tensor], int]:
        frame = frame.to(device=self.model.device)
        self._frame_index2frame[index] = self.frames_processor(frame[None, None, ...])[0, 0]
        predict_index = index - self._predict_offset
        predict_indexes = self.indexes_generator.make_stack_indexes(predict_index)
        self._clear_old(predict_indexes[0])
        if set(predict_indexes) <= set(self._frame_index2frame.keys()):
            stacks_indexes = self.get_stacks_indexes(predict_indexes)
            for stack_indexes in stacks_indexes:
                if stack_indexes not in self._stack_indexes2features:
                    frames = torch.stack([self._frame_index2frame[i] for i in stack_indexes], dim=0)
                    if self.tta:
                        frames = torch.stack([frames, hflip(frames)], dim=0)
                    else:
                        frames = frames.unsqueeze(0)
                    features = self.model.nn_module.forward_2d(frames)
                    self._stack_indexes2features[stack_indexes] = features
            features = torch.cat([self._stack_indexes2features[s] for s in stacks_indexes], dim=1)
            features = self.model.nn_module.forward_3d(features)
            prediction = self.model.nn_module.forward_head(features)
            prediction = self.model.prediction_transform(prediction)
            prediction = torch.mean(prediction, dim=0)
            return prediction, predict_index
        else:
            return None, predict_index
