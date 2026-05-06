from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

import conv3d_fp8_op


class Conv3dFp8(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        if not isinstance(module, nn.Conv3d):
            raise TypeError(f"module must be nn.Conv3d, got {type(module).__name__}")
        if module.bias is not None:
            raise ValueError("Conv3dFp8 currently supports only Conv3d without bias.")
        if module.groups != 1:
            raise ValueError("Conv3dFp8 currently supports only groups=1 Conv3d.")

        self.module = module
        self.padding: Tuple[int, int, int] = tuple(int(v) for v in module.padding)
        self.stride: Tuple[int, int, int] = tuple(int(v) for v in module.stride)
        self.dilation: Tuple[int, int, int] = tuple(int(v) for v in module.dilation)

        self._op: Optional[conv3d_fp8_op.Conv3dFp8Op] = None
        self._cached_x_shape: Optional[Tuple[int, int, int, int, int]] = None
        self._cached_weight_shape: Tuple[int, int, int, int, int] = tuple(int(v) for v in module.weight.shape)
        self._cached_weight_fp8: Optional[torch.Tensor] = None
        self._cached_descale_w: Optional[torch.Tensor] = None

    def _get_weight_fp8_cached(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_weight_fp8 is None:
            weight = self.module.weight
            if weight.device != device:
                weight = weight.to(device)

            weight_absmax = weight.abs().max()
            fp8_max = torch.tensor(torch.finfo(torch.float8_e4m3fn).max, dtype=weight.dtype, device=device)
            safe_absmax = torch.clamp(weight_absmax, min=torch.finfo(weight.dtype).eps)

            # Scale weight to occupy the FP8 representable range, then keep inverse scale in descale_w.
            weight_scale = fp8_max / safe_absmax
            weight_scaled = weight * weight_scale

            self._cached_weight_fp8 = weight_scaled.to(torch.float8_e4m3fn).to(memory_format=torch.channels_last_3d)
            descale_w_scalar = (safe_absmax / fp8_max).to(torch.float32)
            self._cached_descale_w = descale_w_scalar.reshape(1, 1, 1, 1, 1)
            self._cached_weight_shape = tuple(int(v) for v in self._cached_weight_fp8.shape)

        assert self._cached_descale_w is not None
        return self._cached_weight_fp8, self._cached_descale_w

    def _quantize_input_to_fp8(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_absmax = x.abs().max()
        fp8_max = torch.tensor(torch.finfo(torch.float8_e4m3fn).max, dtype=x.dtype, device=x.device)
        safe_absmax = torch.clamp(x_absmax, min=torch.finfo(x.dtype).eps)

        x_scale = fp8_max / safe_absmax
        x_scaled = x * x_scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn).to(memory_format=torch.channels_last_3d)
        descale_x = (safe_absmax / fp8_max).to(torch.float32).reshape(1, 1, 1, 1, 1)
        return x_fp8, descale_x

    def _ensure_op(
        self,
        x_shape: Tuple[int, int, int, int, int],
        w_shape: Tuple[int, int, int, int, int],
        device_index: int,
    ) -> None:
        if self._op is not None and self._cached_x_shape == x_shape and self._cached_weight_shape == w_shape:
            return
        self._op = conv3d_fp8_op.init(
            x_shape=x_shape,
            w_shape=w_shape,
            device_index=device_index,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        self._cached_x_shape = x_shape
        self._cached_weight_shape = w_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor.")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"x must be torch.bfloat16, got {x.dtype}.")
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("Current PyTorch does not expose float8_e4m3fn.")

        x_fp8, descale_x = self._quantize_input_to_fp8(x)
        w_fp8, descale_w = self._get_weight_fp8_cached(x_fp8.device)

        x_shape = tuple(int(v) for v in x_fp8.shape)
        w_shape = tuple(int(v) for v in w_fp8.shape)
        if len(x_shape) != 5:
            raise ValueError(f"x must be a 5D tensor, got shape={x_shape}")
        if len(w_shape) != 5:
            raise ValueError(f"weight must be a 5D tensor, got shape={w_shape}")

        self._ensure_op(x_shape, w_shape, x_fp8.device.index)
        assert self._op is not None

        return self._op.forward(x_fp8, w_fp8, descale_x, descale_w)
