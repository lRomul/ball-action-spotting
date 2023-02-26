from pathlib import Path
from typing import Optional

import torch

import PyNvCodec as nvc
import PytorchNvCodec as pnvc


class NvDecFrameFetcher:
    def __init__(self, video_path: str | Path, gpu_id: int):
        self.video_path = Path(video_path)
        self.gpu_id = gpu_id
        self._nv_dec = nvc.PyNvDecoder(video_path, self.gpu_id)
        self.num_frames = self._nv_dec.Numframes()
        self.width = self._nv_dec.Width()
        self.height = self._nv_dec.Height()

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
        self._current_index = 0  # It seems VPF skips first frame at start

    @property
    def current_index(self) -> int:
        return self._current_index

    def fetch_frames(self, indexes: list[int]) -> torch.Tensor:
        min_frame_index = min(indexes)
        max_frame_index = max(indexes)

        index2frame = dict()
        frame_indexes_set = set(indexes)
        for index in range(min_frame_index, max_frame_index + 1):
            if index not in frame_indexes_set:
                self._nv_dec.DecodeSingleSurface()
                self._current_index += 1
                continue
            if index == min_frame_index:
                frame_tensor = self.fetch_frame(index)
            else:
                frame_tensor = self.fetch_frame()
            index2frame[index] = frame_tensor

        frames = [index2frame[index] for index in indexes]
        return torch.stack(frames, dim=0)

    def _next_decode(self):
        if self._current_index < self.num_frames - 1:
            # Fetch next frame
            nv12_surface = self._nv_dec.DecodeSingleSurface()
            self._current_index += 1
            return nv12_surface
        else:
            raise RuntimeError(f"End of frames")

    def _seek_and_decode(self, index: int):
        if index < 0 or index >= self.num_frames:
            raise RuntimeError(f"Frame index {index} out of range")
        # Apparently frame indexing starts from 1 in VPF unlike OpenCV
        seek_ctx = nvc.SeekContext(index - 1)
        nv12_surface = self._nv_dec.DecodeSingleSurface(seek_context=seek_ctx)
        self._current_index = index
        return nv12_surface

    def fetch_frame(self, index: Optional[int] = None) -> torch.Tensor:
        if index is None:
            nv12_surface = self._next_decode()
        else:
            nv12_surface = self._seek_and_decode(index)

        grayscale_surface = self._to_grayscale.Execute(nv12_surface, self._cc_ctx)
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
