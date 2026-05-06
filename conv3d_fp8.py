from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


def _normalize_3tuple(value: Iterable[int], name: str) -> Tuple[int, int, int]:
    value = tuple(int(v) for v in value)
    if len(value) != 3:
        raise ValueError(f"{name} must be a 3-tuple/list, got {value}")
    return value


def _normalize_5tuple(value: Iterable[int], name: str) -> Tuple[int, int, int, int, int]:
    value = tuple(int(v) for v in value)
    if len(value) != 5:
        raise ValueError(f"{name} must be a 5-tuple/list, got {value}")
    return value


try:
    import conv3d_fp8_ext  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "conv3d_fp8_ext is not built yet. Please run:\n"
        "  cd /home/scratch.xueweil_sw/tmp/Conv3D_perf\n"
        "  python setup_conv3d_fp8.py build_ext --inplace"
    ) from exc


@dataclass
class Conv3dFp8Op:
    _handle_id: int
    _padding: Tuple[int, int, int]
    _stride: Tuple[int, int, int]
    _dilation: Tuple[int, int, int]

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        descale_x: torch.Tensor,
        descale_w: torch.Tensor,
        scale_y: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_cuda or not w.is_cuda:
            raise ValueError("x and w must be CUDA tensors.")
        if x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
            raise ValueError("x and w must be torch.float8_e4m3fn tensors.")

        x_cl = x.to(memory_format=torch.channels_last_3d)
        w_cl = w.to(memory_format=torch.channels_last_3d)
        return torch.ops.conv3d_fp8.forward(self._handle_id, x_cl, w_cl, descale_x, descale_w, scale_y)

    def __del__(self) -> None:
        try:
            torch.ops.conv3d_fp8.destroy(self._handle_id)
        except Exception:
            # Best effort cleanup in destructor.
            pass


def init(
    x_shape: Iterable[int],
    w_shape: Iterable[int],
    device_index: int | None = None,
    padding: Iterable[int] = (0, 0, 0),
    stride: Iterable[int] = (1, 1, 1),
    dilation: Iterable[int] = (1, 1, 1),
) -> Conv3dFp8Op:
    x_shape = _normalize_5tuple(x_shape, "x_shape")
    w_shape = _normalize_5tuple(w_shape, "w_shape")
    padding = _normalize_3tuple(padding, "padding")
    stride = _normalize_3tuple(stride, "stride")
    dilation = _normalize_3tuple(dilation, "dilation")

    if device_index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        device_index = torch.cuda.current_device()
    device_index = int(device_index)

    handle_id = conv3d_fp8_ext.init(
        list(x_shape),
        list(w_shape),
        device_index,
        list(padding),
        list(stride),
        list(dilation),
    )
    return Conv3dFp8Op(handle_id, padding, stride, dilation)
