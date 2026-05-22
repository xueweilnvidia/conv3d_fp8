from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch

import nvtx


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
    import conv3d_bf16_ext  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "Failed to import conv3d_bf16_ext. This is usually caused by either:\n"
        "1) extension not built, or\n"
        "2) ABI mismatch between the built extension and current torch/cuda runtime.\n\n"
        "Try rebuilding in the current environment:\n"
        "  cd <project_root>\n"
        "  rm -rf build *.egg-info *.so\n"
        "  pip install -e . --no-build-isolation\n\n"
        f"Original import error: {exc}"
    ) from exc


@dataclass
class Conv3dBf16Op:
    _handle_id: int
    _padding: Tuple[int, int, int]
    _stride: Tuple[int, int, int]
    _dilation: Tuple[int, int, int]
    _with_bias: bool

    @nvtx.annotate(message="Conv3dBf16Op.forward")
    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not x.is_cuda or not w.is_cuda:
            raise ValueError("x and w must be CUDA tensors.")
        if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            raise ValueError("x and w must be torch.bfloat16 tensors.")
        if self._with_bias:
            if bias is None:
                raise ValueError("bias is required when op is initialized with with_bias=True.")
            if not bias.is_cuda:
                raise ValueError("bias must be a CUDA tensor.")
            if bias.dtype != torch.bfloat16:
                raise ValueError("bias must be torch.bfloat16.")
            if bias.dim() != 1 or int(bias.shape[0]) != int(w.shape[0]):
                raise ValueError("bias must have shape (out_channels,).")
        elif bias is not None:
            raise ValueError("bias must be None when op is initialized with with_bias=False.")

        x_cl = x.to(memory_format=torch.channels_last_3d)
        w_cl = w.to(memory_format=torch.channels_last_3d)
        return torch.ops.conv3d_bf16.forward(self._handle_id, x_cl, w_cl, bias)

    def __del__(self) -> None:
        try:
            torch.ops.conv3d_bf16.destroy(self._handle_id)
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
    with_bias: bool = False,
) -> Conv3dBf16Op:
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

    handle_id = conv3d_bf16_ext.init(
        list(x_shape),
        list(w_shape),
        device_index,
        list(padding),
        list(stride),
        list(dilation),
        bool(with_bias),
    )
    return Conv3dBf16Op(handle_id, padding, stride, dilation, bool(with_bias))
