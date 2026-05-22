import torch
import torch.nn.functional as F

import conv3d_bf16_op


def _summarize(y_ours: torch.Tensor, y_ref: torch.Tensor, label: str):
    diff = (y_ours.float() - y_ref.float()).abs()
    ref_abs = y_ref.float().abs()
    rel = diff / torch.clamp(ref_abs, min=1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel.max().item()
    y_ref_max_abs = ref_abs.max().item()
    y_ref_mean_abs = ref_abs.mean().item()
    print(
        f"[{label}] max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, max_rel={max_rel:.6f}, "
        f"y_ref_max_abs={y_ref_max_abs:.6f}, y_ref_mean_abs={y_ref_mean_abs:.6f}"
    )
    return max_abs, mean_abs, max_rel


def run_check(with_bias: bool):
    torch.manual_seed(0)
    device = "cuda"

    n, c, d, h, w = 1, 96, 6, 834, 642
    k, kt, kh, kw = 96, 3, 3, 3
    padding = (0, 0, 0)
    stride = (1, 1, 1)
    dilation = (1, 1, 1)

    input_bf16 = torch.randn(n, c, d, h, w, dtype=torch.bfloat16, device=device).to(
        memory_format=torch.channels_last_3d
    )
    weight_bf16 = torch.randn(k, c, kt, kh, kw, dtype=torch.bfloat16, device=device).to(
        memory_format=torch.channels_last_3d
    )
    bias_bf16 = torch.randn(k, dtype=torch.bfloat16, device=device) if with_bias else None

    op = conv3d_bf16_op.init(
        x_shape=input_bf16.shape,
        w_shape=weight_bf16.shape,
        device_index=torch.cuda.current_device(),
        padding=padding,
        stride=stride,
        dilation=dilation,
        with_bias=with_bias,
    )

    with torch.no_grad():
        y_ours = op.forward(input_bf16, weight_bf16, bias=bias_bf16)
        y_ref = F.conv3d(
            input_bf16,
            weight_bf16,
            bias=bias_bf16,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    label = "with_bias" if with_bias else "no_bias"
    return _summarize(y_ours, y_ref, label)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("Accuracy check against torch.nn.functional.conv3d (bf16)")

    max_abs_nb, mean_abs_nb, max_rel_nb = run_check(with_bias=False)
    max_abs_b, mean_abs_b, max_rel_b = run_check(with_bias=True)

    # bf16 conv shares the same input precision as the reference. Differences are
    # dominated by reduction-order and kernel-selection differences between the
    # autotuned cuDNN plan and torch's default kernel. Thresholds are scaled by
    # the magnitude of the reference output (which is ~sqrt(c*kt*kh*kw) ~= O(30)).
    tol_max_abs = 5.0
    tol_mean_abs = 0.5

    failed = False
    if max_abs_nb >= tol_max_abs or mean_abs_nb >= tol_mean_abs:
        print(f"FAIL no_bias: max_abs={max_abs_nb}, mean_abs={mean_abs_nb}")
        failed = True
    if max_abs_b >= tol_max_abs or mean_abs_b >= tol_mean_abs:
        print(f"FAIL with_bias: max_abs={max_abs_b}, mean_abs={mean_abs_b}")
        failed = True

    if failed:
        raise RuntimeError("conv3d_bf16 accuracy check failed.")
    print("conv3d_bf16 accuracy check passed.")


if __name__ == "__main__":
    main()
