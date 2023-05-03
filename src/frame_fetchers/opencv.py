from typing import Any
from pathlib import Path

import cv2
import torch

from src.frame_fetchers.abstract import AbstractFrameFetcher


class OpencvFrameFetcher(AbstractFrameFetcher):
    def __init__(self, video_path: str | Path, gpu_id: int):
        super().__init__(video_path=video_path, gpu_id=gpu_id)
        self.video = cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG)
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _next_decode(self) -> Any:
        _, frame = self.video.read()
        return frame

    def _seek_and_decode(self, index: int) -> Any:
        self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        _, frame = self.video.read()
        return frame

    def _convert(self, frame: Any) -> torch.Tensor:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_tensor = torch.from_numpy(grayscale_frame)
        frame_tensor = frame_tensor.to(device=f"cuda:{self.gpu_id}")
        return frame_tensor
