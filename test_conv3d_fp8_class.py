import torch
import torch.nn as nn

from conv3d_fp8 import Conv3dFp8


def _require_cuda_fp8() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Current PyTorch does not expose float8_e4m3fn.")


def _build_modules_and_input(with_bias: bool):
    _require_cuda_fp8()
    torch.manual_seed(0)

    device = "cuda"
    in_channels = 96
    out_channels = 96
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)

    # Keep the same shape family as existing tests for apples-to-apples comparison.
    x = torch.randn(1, in_channels, 6, 834, 642, dtype=torch.bfloat16, device=device)

    conv_ref = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=with_bias,
    ).to(dtype=torch.bfloat16, device=device)
    conv_ref.eval()

    conv_fp8 = Conv3dFp8(conv_ref)
    return x, conv_ref, conv_fp8


def run_conv3d_fp8_class_accuracy_check(with_bias: bool):
    x, conv_ref, conv_fp8 = _build_modules_and_input(with_bias=with_bias)

    with torch.no_grad():
        y_fp8 = conv_fp8(x)
        y_ref = conv_ref(x)

    diff = (y_fp8 - y_ref).abs().float()
    ref_abs = y_ref.abs().float()
    rel = diff / torch.clamp(ref_abs, min=1e-6)

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()
    y_ref_max_abs = ref_abs.max().item()
    y_ref_mean_abs = ref_abs.mean().item()

    print(
        "Conv3dFp8 accuracy vs nn.Conv3d: "
        f"max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, "
        f"max_rel={max_rel:.6f}, mean_rel={mean_rel:.6f}, "
        f"y_ref_max_abs={y_ref_max_abs:.6f}, y_ref_mean_abs={y_ref_mean_abs:.6f}"
    )

    # print("y_fp8:")
    # print(y_fp8)
    # print("y_ref:")
    # print(y_ref)

    return max_abs, mean_abs, max_rel, mean_rel


def _benchmark_ms(fn, iters: int = 30, warmup: int = 10) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
    torch.cuda.synchronize()

    durations_ms = []
    with torch.no_grad():
        for _ in range(iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = fn()
            end_event.record()
            torch.cuda.synchronize()
            durations_ms.append(start_event.elapsed_time(end_event))
    return sum(durations_ms) / len(durations_ms)


def run_conv3d_fp8_class_perf_check(with_bias: bool):
    x, conv_ref, conv_fp8 = _build_modules_and_input(with_bias=with_bias)

    fp8_ms = _benchmark_ms(lambda: conv_fp8(x))
    ref_ms = _benchmark_ms(lambda: conv_ref(x))
    speedup = ref_ms / fp8_ms if fp8_ms > 0 else float("inf")

    print(
        "Conv3dFp8 performance: "
        f"fp8_avg_ms={fp8_ms:.3f}, torch_avg_ms={ref_ms:.3f}, speedup={speedup:.3f}x"
    )
    return fp8_ms, ref_ms, speedup


def test_conv3d_fp8_class_accuracy():
    max_abs, mean_abs, _, mean_rel = run_conv3d_fp8_class_accuracy_check(with_bias=False)

    # FP8 quantization introduces visible numeric differences; keep thresholds practical.
    assert max_abs < 2.0, f"max_abs too large: {max_abs}"
    assert mean_abs < 0.2, f"mean_abs too large: {mean_abs}"
    assert mean_rel < 0.2, f"mean_rel too large: {mean_rel}"


def test_conv3d_fp8_class_accuracy_with_bias():
    max_abs, mean_abs, _, mean_rel = run_conv3d_fp8_class_accuracy_check(with_bias=True)

    # FP8 quantization introduces visible numeric differences; keep thresholds practical.
    assert max_abs < 2.0, f"max_abs too large: {max_abs}"
    assert mean_abs < 0.2, f"mean_abs too large: {mean_abs}"
    assert mean_rel < 0.2, f"mean_rel too large: {mean_rel}"


def test_conv3d_fp8_class_performance():
    # Keep one perf path to control test runtime; accuracy tests cover both bias modes.
    fp8_ms, ref_ms, _ = run_conv3d_fp8_class_perf_check(with_bias=True)

    assert fp8_ms > 0.0
    assert ref_ms > 0.0


def main():
    run_conv3d_fp8_class_accuracy_check(with_bias=False)
    run_conv3d_fp8_class_accuracy_check(with_bias=True)
    run_conv3d_fp8_class_perf_check(with_bias=True)


if __name__ == "__main__":
    main()
