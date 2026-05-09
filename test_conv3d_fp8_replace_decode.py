#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compare ``vae.decode`` output before and after ``Conv3dFp8`` replacement.

Steps:
    1. Load an ``AutoencoderKLWan`` VAE.
    2. Run ``vae.decode`` with the original ``nn.Conv3d`` modules (baseline).
    3. Call ``replace_conv3d_with_fp8(vae.decoder)`` to swap eligible
       ``nn.Conv3d`` modules with the FP8 wrapper.
    4. Run ``vae.decode`` again and compare output against the baseline.

Requires:
    - CUDA + PyTorch with ``float8_e4m3fn``
    - The ``conv3d_fp8`` package importable (the C++ extension built).

Usage:
    python test_conv3d_fp8_replace_decode.py \
        --model-id ../Wan2.2-T2V-A14B-Diffusers --subfolder vae
"""

import argparse
import os
import sys
from typing import Tuple

import torch
from diffusers import AutoencoderKLWan

import nvtx

# Allow importing the conv3d_fp8 package without a system install.
# _DEFAULT_CONV3D_FP8_PATH = os.environ.get("CONV3D_FP8_PATH", "/workdir/tmp/conv3d_fp8")
# if os.path.isdir(_DEFAULT_CONV3D_FP8_PATH) and _DEFAULT_CONV3D_FP8_PATH not in sys.path:
#     sys.path.insert(0, _DEFAULT_CONV3D_FP8_PATH)

from conv3d_fp8_replace import replace_conv3d_with_fp8  # noqa: E402

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in shape_str.split(","))


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[key]


def _require_cuda_fp8() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Current PyTorch does not expose float8_e4m3fn.")


def decode_once(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        with nvtx.annotate("decode_once"):
            out = vae.decode(latents, return_dict=False)[0]
    return out


def report_diff(label: str, y_ref: torch.Tensor, y: torch.Tensor) -> dict:
    # print("y_ref: ", y_ref)
    # print("y: ", y)
    y_ref_f = y_ref.float()
    y_f = y.float()
    diff = (y_f - y_ref_f).abs()
    ref_abs = y_ref_f.abs()
    rel = diff / torch.clamp(ref_abs, min=1e-6)
    cos_sim = torch.nn.functional.cosine_similarity(
        y_f.flatten().unsqueeze(0), y_ref_f.flatten().unsqueeze(0), dim=1, eps=1e-12
    )

    stats = {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
        "y_ref_max_abs": ref_abs.max().item(),
        "y_ref_mean_abs": ref_abs.mean().item(),
        "cos_sim": cos_sim.item(),
    }
    print(
        f"{label}: "
        f"max_abs={stats['max_abs']:.6f}, mean_abs={stats['mean_abs']:.6f}, "
        f"max_rel={stats['max_rel']:.6f}, mean_rel={stats['mean_rel']:.6f}, "
        f"cos_sim={stats['cos_sim']:.6f}, "
        f"y_ref_max_abs={stats['y_ref_max_abs']:.6f}, y_ref_mean_abs={stats['y_ref_mean_abs']:.6f}"
    )
    return stats


def run_compare(
    model_id: str,
    subfolder: str,
    latent_shape: Tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    _require_cuda_fp8()
    torch.manual_seed(seed)

    print(f"Loading VAE: model_id={model_id}, subfolder={subfolder}, dtype={dtype}, device={device}")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch.float32).to(dtype).to(device)
    vae.eval()

    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    print(f"latents shape: {tuple(latents.shape)}, dtype: {latents.dtype}")

    print("\n== Baseline decode (original nn.Conv3d) ==")
    y_ref = decode_once(vae, latents)


    # torch.cuda.cudart().cudaProfilerStart()
    # for i in range(2):
    #     y_ref = decode_once(vae, latents)
    #     print("decode bf16")
    # torch.cuda.cudart().cudaProfilerStop()
    # Move baseline to CPU to free GPU memory before the FP8 path runs.
    y_ref_cpu = y_ref.detach().to("cpu")
    del y_ref
    torch.cuda.empty_cache()
    print(f"output shape: {tuple(y_ref_cpu.shape)}, dtype: {y_ref_cpu.dtype}")

    print("\n== Replacing nn.Conv3d in vae.decoder with Conv3dFp8 ==")
    replaced = replace_conv3d_with_fp8(vae.decoder)
    if not replaced:
        print(
            "WARNING: no nn.Conv3d was replaced — the diff below will be ~0 and not "
            "exercise Conv3dFp8. Check the decoder structure / exclude_keywords."
        )

    print("\n== FP8 decode (after replacement) ==")
    y_fp8 = decode_once(vae, latents)
    print(f"output shape: {tuple(y_fp8.shape)}, dtype: {y_fp8.dtype}")

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(2):
        y_fp8 = decode_once(vae, latents)
        print("decode")

    torch.cuda.cudart().cudaProfilerStop()

    print("\n== Accuracy diff (fp8 vs baseline) ==")
    y_ref_dev = y_ref_cpu.to(y_fp8.device)
    stats = report_diff("vae.decode fp8 vs ref", y_ref_dev, y_fp8)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vae.decode precision before/after Conv3dFp8 replacement."
    )
    parser.add_argument("--model-id", type=str, default="../Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--subfolder", type=str, default="vae")
    parser.add_argument(
        "--latent-shape",
        type=str,
        default="1,16,17,104,80",
        help="Latent tensor shape as B,C,T,H,W. Smaller default to keep memory modest.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_compare(
        model_id=args.model_id,
        subfolder=args.subfolder,
        latent_shape=parse_shape(args.latent_shape),
        device=args.device,
        dtype=parse_dtype(args.dtype),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
