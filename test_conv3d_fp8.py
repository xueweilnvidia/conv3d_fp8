import torch
from contextlib import contextmanager

import conv3d_fp8_op


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


def conv3d_flops(c_in, c_out, k_t, k_h, k_w, t, h, w):
    return 2 * k_t * k_h * k_w * c_in * c_out * t * h * w


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Current PyTorch does not expose float8_e4m3fn.")

    c = 96
    k = 96
    n, d, h, w = 1, 6, 834, 642
    kt, kh, kw = 3, 3, 3
    padding = (0, 0, 0)
    stride = (1, 1, 1)
    dilation = (1, 1, 1)

    input_bf16 = torch.randn(n, c, d, h, w, dtype=torch.bfloat16, device="cuda")
    weight_bf16 = torch.randn(k, c, kt, kh, kw, dtype=torch.bfloat16, device="cuda")

    input_fp8 = input_bf16.to(torch.float8_e4m3fn).to(memory_format=torch.channels_last_3d)
    weight_fp8 = weight_bf16.to(torch.float8_e4m3fn).to(memory_format=torch.channels_last_3d)

    descale_x = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32, device="cuda")
    descale_w = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32, device="cuda")

    op = conv3d_fp8_op.init(
        x_shape=input_fp8.shape,
        w_shape=weight_fp8.shape,
        device_index=torch.cuda.current_device(),
        padding=padding,
        stride=stride,
        dilation=dilation,
    )

    warmup_iters = 5
    with torch.no_grad():
        for _ in range(warmup_iters):
            with nvtx_range("warmup/conv3d_fp8_forward"):
                output = op.forward(input_fp8, weight_fp8, descale_x, descale_w)
    torch.cuda.synchronize()

    test_iters = 20
    durations_ms = []
    with torch.no_grad():
        for _ in range(test_iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with nvtx_range("benchmark/conv3d_fp8_forward"):
                output = op.forward(input_fp8, weight_fp8, descale_x, descale_w)
            end_event.record()
            torch.cuda.synchronize()
            durations_ms.append(start_event.elapsed_time(end_event))

    avg_ms = sum(durations_ms) / len(durations_ms)
    min_ms = min(durations_ms)
    max_ms = max(durations_ms)

    total_flops = conv3d_flops(c, k, kt, kh, kw, d, h, w)
    avg_tflops = total_flops / (avg_ms / 1000) / 1e12

    print(f"Output shape: {tuple(output.shape)}")
    print(f"conv3d_fp8.forward avg: {avg_ms:.3f} ms, min: {min_ms:.3f} ms, max: {max_ms:.3f} ms")
    print(f"Estimated throughput: {avg_tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
