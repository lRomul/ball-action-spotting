from typing import Any
from pathlib import Path

import torch

import PyNvCodec as nvc
import PytorchNvCodec as pnvc


from src.frame_fetchers.abstract import AbstractFrameFetcher


class NvDecFrameFetcher(AbstractFrameFetcher):
    def __init__(self, video_path: str | Path, gpu_id: int):
        super().__init__(video_path=video_path, gpu_id=gpu_id)
        self._nv_dec = nvc.PyNvDecoder(str(self.video_path), self.gpu_id)
        self.num_frames = self._nv_dec.Numframes()
        self.width = self._nv_dec.Width()
        self.height = self._nv_dec.Height()

        self._current_index = 0  # It seems VPF skips first frame at start

        self._to_grayscale = nvc.PySurfaceConverter(
            self.width,
            self.height,
            nvc.PixelFormat.NV12,
            nvc.PixelFormat.Y,
            self.gpu_id,
        )
        self._cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG
        )

    def _next_decode(self) -> Any:
        nv12_surface = self._nv_dec.DecodeSingleSurface()
        return nv12_surface

    def _seek_and_decode(self, index: int) -> Any:
        # Apparently frame indexing starts from 1 in VPF unlike OpenCV
        seek_ctx = nvc.SeekContext(index - 1)
        nv12_surface = self._nv_dec.DecodeSingleSurface(seek_context=seek_ctx)
        return nv12_surface

    def _convert(self, frame: Any) -> torch.Tensor:
        grayscale_surface = self._to_grayscale.Execute(frame, self._cc_ctx)
        surf_plane = grayscale_surface.PlanePtr()
        frame_tensor = pnvc.makefromDevicePtrUint8(
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )
        frame_tensor.resize_(self.height, self.width)
        return frame_tensor
