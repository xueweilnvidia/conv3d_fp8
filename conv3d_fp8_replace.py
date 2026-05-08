# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route ``nn.Conv3d`` (or its subclasses) through the FP8-accelerated ``Conv3dFp8``.

For pure ``nn.Conv3d`` instances the whole module is swapped with a
``Conv3dFp8``. For subclasses such as ``WanCausalConv3d`` whose ``forward``
adds custom logic (causal padding, ``cache_x``), we keep the subclass instance
in place and only override its ``_conv_forward`` so that the final convolution
call goes through ``Conv3dFp8`` while the surrounding logic is preserved.

``Conv3dFp8`` requires bfloat16 CUDA inputs; callers must run the host module
in bf16 (e.g. ``vae.to(torch.bfloat16)``).
"""

import types
from typing import Iterable, List, Tuple

import torch.nn as nn

DEFAULT_EXCLUDE_KEYWORDS: Tuple[str, ...] = ("upsamplers", "conv_in", "conv_out", "conv_shortcut")


def _make_patched_conv_forward(fp8: nn.Module):
    """Build a ``_conv_forward`` that routes the inner conv through ``fp8``.

    ``nn.Conv3d.forward`` calls ``self._conv_forward(input, self.weight, self.bias)``.
    The weight/bias args are ignored here because ``Conv3dFp8`` already owns its
    own cached FP8 weight (and reads bias via ``module.bias``).
    """

    def _patched(self, input, weight, bias):
        return fp8(input)

    return _patched


def replace_conv3d_with_fp8(
    model: nn.Module,
    exclude_keywords: Iterable[str] = DEFAULT_EXCLUDE_KEYWORDS,
) -> List[str]:
    """In-place modify the model so eligible Conv3d ops dispatch to ``Conv3dFp8``.

    - Pure ``nn.Conv3d`` instances are swapped with ``Conv3dFp8``.
    - Subclasses (e.g. ``WanCausalConv3d``) keep their ``forward`` (causal
      padding, ``cache_x`` etc.) and only have ``_conv_forward`` overridden to
      call ``Conv3dFp8`` for the underlying convolution.

    Modules whose dotted name contains any keyword in ``exclude_keywords`` are
    skipped. Returns the list of modified module names.
    """
    from conv3d_fp8 import Conv3dFp8

    exclude_keywords = tuple(exclude_keywords)

    matched: List[Tuple[str, nn.Conv3d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            if any(kw in name for kw in exclude_keywords):
                continue
            matched.append((name, module))

    annotated: List[Tuple[str, str]] = []
    swap_count = 0
    patch_count = 0

    for name, module in matched:
        fp8 = Conv3dFp8(module)
        if type(module) is nn.Conv3d:
            parent = model
            attrs = name.split(".")
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, attrs[-1], fp8)
            swap_count += 1
            annotated.append((name, "swap (nn.Conv3d -> Conv3dFp8)"))
        else:
            # Register Conv3dFp8 as a sub-module of the original Conv3d
            # subclass so device/dtype moves carry it along, then redirect
            # `_conv_forward` to it. The subclass's `forward` (causal padding,
            # cache_x, etc.) keeps running unchanged.
            module._conv3d_fp8 = fp8
            module._conv_forward = types.MethodType(_make_patched_conv_forward(fp8), module)
            patch_count += 1
            annotated.append((name, f"patch _conv_forward ({type(module).__name__})"))

    print(
        f"[Conv3dFp8] Modified {len(matched)} Conv3d module(s) "
        f"(swap={swap_count}, patch={patch_count}; "
        f"excluded keywords: {list(exclude_keywords)}):"
    )
    for name, kind in annotated:
        print(f"  {name}  [{kind}]")

    return [name for name, _ in matched]
