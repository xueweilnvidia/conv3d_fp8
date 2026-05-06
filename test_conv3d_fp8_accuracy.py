import torch
import torch.nn.functional as F

import conv3d_fp8_op


def run_conv3d_fp8_accuracy_check():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Current PyTorch does not expose float8_e4m3fn.")

    torch.manual_seed(0)
    device = "cuda"

    n, c, d, h, w = 1, 96, 6, 834, 642
    k, kt, kh, kw = 96, 3, 3, 3
    padding = (0, 0, 0)
    stride = (1, 1, 1)
    dilation = (1, 1, 1)

    input_bf16 = torch.randn(n, c, d, h, w, dtype=torch.bfloat16, device=device)
    weight_bf16 = torch.randn(k, c, kt, kh, kw, dtype=torch.bfloat16, device=device)

    input_fp8 = input_bf16.to(torch.float8_e4m3fn).to(memory_format=torch.channels_last_3d)
    weight_fp8 = weight_bf16.to(torch.float8_e4m3fn).to(memory_format=torch.channels_last_3d)

    descale_x = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32, device=device)
    descale_w = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32, device=device)

    op = conv3d_fp8_op.init(
        x_shape=input_fp8.shape,
        w_shape=weight_fp8.shape,
        device_index=torch.cuda.current_device(),
        padding=padding,
        stride=stride,
        dilation=dilation,
    )

    with torch.no_grad():
        y_fp8 = op.forward(input_fp8, weight_fp8, descale_x, descale_w)
        y_ref = F.conv3d(
            input_fp8.to(torch.bfloat16),
            weight_fp8.to(torch.bfloat16),
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    # print("y_fp8:")
    # print(y_fp8)
    # print("y_ref:")
    # print(y_ref)

    diff = (y_fp8 - y_ref).abs().float()
    ref_abs = y_ref.abs().float()
    rel = diff / torch.clamp(ref_abs, min=1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel.max().item()
    y_ref_max_abs = ref_abs.max().item()
    y_ref_mean_abs = ref_abs.mean().item()


    print(
        "Accuracy check against torch.conv3d: "
        f"max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, max_rel={max_rel:.6f}, "
        f"y_ref_max_abs={y_ref_max_abs:.6f}, y_ref_mean_abs={y_ref_mean_abs:.6f}"
    )

    return max_abs, mean_abs, max_rel, y_ref_max_abs, y_ref_mean_abs


def test_conv3d_fp8_accuracy_against_torch_conv3d():
    max_abs, mean_abs, _, _, _ = run_conv3d_fp8_accuracy_check()

    # FP8 accumulation and kernel choice can introduce visible differences.
    assert max_abs < 1.0, f"max_abs too large: {max_abs}"
    assert mean_abs < 0.1, f"mean_abs too large: {mean_abs}"


def main():
    max_abs, mean_abs, _, y_ref_max_abs, y_ref_mean_abs = run_conv3d_fp8_accuracy_check()
    if max_abs >= 1.0 or mean_abs >= 0.1:
        raise RuntimeError(
            "Accuracy check failed: "
            f"max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, "
            f"y_ref_max_abs={y_ref_max_abs:.6f}, y_ref_mean_abs={y_ref_mean_abs:.6f}"
        )
    print("conv3d_fp8 accuracy check passed.")


if __name__ == "__main__":
    main()
