import torch

import PyNvCodec as nvc
import PytorchNvCodec as pnvc

from pathlib import Path


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

    def fetch(self, frame_indexes: list[int]) -> torch.Tensor:
        min_frame_index = min(frame_indexes)
        max_frame_index = max(frame_indexes)

        if min_frame_index < 0 or max_frame_index >= self.num_frames:
            raise RuntimeError("Frame index out of range")

        seek_ctx = nvc.SeekContext(min_frame_index)

        index2frame = dict()
        frame_indexes_set = set(frame_indexes)
        for index in range(min_frame_index, max_frame_index + 1):
            if index == min_frame_index:
                nv12_surface = self._nv_dec.DecodeSingleSurface(seek_context=seek_ctx)
            else:
                nv12_surface = self._nv_dec.DecodeSingleSurface()

            if index not in frame_indexes_set:
                continue

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
            index2frame[index] = frame_tensor

        frames = [index2frame[index] for index in frame_indexes]
        return torch.stack(frames, dim=0)
