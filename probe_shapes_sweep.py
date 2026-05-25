"""Sweep conv3d perf across many shapes for torch / bf16_op / fp8_op.

Convention: kernel=(3,3,3), padding=(1,1,1), stride=1, dilation=1, k=c.
Output spatial = input spatial.
"""
import torch
import torch.nn.functional as F

import conv3d_bf16_op
import conv3d_fp8_op

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


SHAPES = [
    (1,  48, 8, 136, 1920),
    (1,  96, 5, 135, 1920),
    (1,  48, 8,  88, 1280),
    (1,  96, 8,  68,  960),
    (1,  96, 4,  90, 1280),
    (1, 192, 5,  68,  960),
    (1,  96, 8,  44,  640),
    (1, 192, 4,  45,  640),
    (1, 192, 4,  34,  480),
    (1, 384, 3,  34,  480),
    (1, 192, 4,  22,  320),
    (1, 384, 2,  22,  320),
    (1, 192, 3,  17,  240),
    (1, 384, 2,  17,  240),
    (1, 192, 2,  17,  240),
    (1, 192, 2,  11,  160),
    (1, 384, 1,  11,  160),
    (1, 192, 1,  11,  160),
]

PADDING = (1, 1, 1)
STRIDE = (1, 1, 1)
DILATION = (1, 1, 1)
KERNEL = (3, 3, 3)
WITH_BIAS = True  # toggle for the bias-vs-no-bias sweep


def conv_out_size(in_, pad, stride, dilation, kernel):
    return (in_ + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in events:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in events)
    return sum(times[2:-2]) / max(1, len(times) - 4)  # trimmed mean


def run_one(shape):
    n, c, d, h, w = shape
    k = c
    kt, kh, kw = KERNEL

    out_d = conv_out_size(d, PADDING[0], STRIDE[0], DILATION[0], kt)
    out_h = conv_out_size(h, PADDING[1], STRIDE[1], DILATION[1], kh)
    out_w = conv_out_size(w, PADDING[2], STRIDE[2], DILATION[2], kw)

    x_bf16 = torch.randn(n, c, d, h, w, dtype=torch.bfloat16, device="cuda").to(
        memory_format=torch.channels_last_3d
    )
    w_bf16 = torch.randn(k, c, kt, kh, kw, dtype=torch.bfloat16, device="cuda").to(
        memory_format=torch.channels_last_3d
    )
    bias_bf16 = torch.randn(k, dtype=torch.bfloat16, device="cuda")

    x_fp8 = x_bf16.to(torch.float8_e4m3fn).to(memory_format=torch.channels_last_3d)
    w_fp8 = w_bf16.to(torch.float8_e4m3fn).to(memory_format=torch.channels_last_3d)
    descale_x = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32, device="cuda")
    descale_w = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32, device="cuda")

    op_bf16 = conv3d_bf16_op.init(
        x_shape=x_bf16.shape, w_shape=w_bf16.shape,
        device_index=torch.cuda.current_device(),
        padding=PADDING, stride=STRIDE, dilation=DILATION, with_bias=WITH_BIAS,
    )
    op_fp8 = conv3d_fp8_op.init(
        x_shape=x_fp8.shape, w_shape=w_fp8.shape,
        device_index=torch.cuda.current_device(),
        padding=PADDING, stride=STRIDE, dilation=DILATION, with_bias=WITH_BIAS,
    )

    fwd_bias = bias_bf16 if WITH_BIAS else None
    with torch.no_grad():
        t_torch = bench(lambda: F.conv3d(x_bf16, w_bf16, bias=fwd_bias, stride=STRIDE, padding=PADDING))
        t_bf16  = bench(lambda: op_bf16.forward(x_bf16, w_bf16, bias=fwd_bias))
        t_fp8   = bench(lambda: op_fp8.forward(x_fp8, w_fp8, descale_x, descale_w, bias=fwd_bias))

    return (out_d, out_h, out_w), t_torch, t_bf16, t_fp8


def main():
    print(f"kernel={KERNEL}, padding={PADDING}, stride={STRIDE}, k=c, with_bias={WITH_BIAS}\n")
    header = (f"{'shape (N,C,D,H,W)':28s} | "
              f"{'torch ms':>10s} {'bf16 ms':>10s} {'fp8 ms':>10s} | "
              f"{'bf16/t':>7s} {'fp8/t':>7s}")
    print(header)
    print("-" * len(header))

    totals = {"torch": 0.0, "bf16": 0.0, "fp8": 0.0}
    for shape in SHAPES:
        _, _, d, h, w = shape
        out_d = conv_out_size(d, PADDING[0], STRIDE[0], DILATION[0], KERNEL[0])
        out_h = conv_out_size(h, PADDING[1], STRIDE[1], DILATION[1], KERNEL[1])
        out_w = conv_out_size(w, PADDING[2], STRIDE[2], DILATION[2], KERNEL[2])
        if out_d <= 0 or out_h <= 0 or out_w <= 0:
            print(f"{str(shape):28s} | SKIP: out spatial <= 0 (out=({out_d},{out_h},{out_w}))")
            continue
        try:
            out_shape, t_torch, t_bf16, t_fp8 = run_one(shape)
        except Exception as exc:
            print(f"{str(shape):28s} | FAILED: {exc}")
            continue
        totals["torch"] += t_torch / 1000
        totals["bf16"] += t_bf16 / 1000
        totals["fp8"] += t_fp8 / 1000
        print(f"{str(shape):28s} | "
              f"{t_torch:10.3f} {t_bf16:10.3f} {t_fp8:10.3f} | "
              f"{t_torch/t_bf16:7.2f} {t_torch/t_fp8:7.2f}")

    print("-" * len(header))
    print(f"{'TOTAL (sum)':28s} | "
          f"{totals['torch']*1000:10.3f} {totals['bf16']*1000:10.3f} {totals['fp8']*1000:10.3f} | "
          f"{totals['torch']/totals['bf16']:7.2f} {totals['torch']/totals['fp8']:7.2f}")


if __name__ == "__main__":
    main()
