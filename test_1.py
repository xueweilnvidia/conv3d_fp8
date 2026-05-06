import torch
import torch.nn as nn
from contextlib import contextmanager

# Explicitly enable cuDNN for Conv3d benchmarking.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

c = 96
k = 96


# 1. Prepare input tensor
input_tensor = torch.randn(1, c, 6, 834, 642, dtype=torch.bfloat16, device='cuda')

# 2. Prepare weight and bias parameters
weight = torch.randn(k, c, 3, 3, 3, dtype=torch.bfloat16, device='cuda')
bias = torch.randn(k, dtype=torch.bfloat16, device='cuda')

# 3. Instantiate nn.Conv3d
conv3d = nn.Conv3d(in_channels=c, out_channels=k, kernel_size=3, padding=0, stride=1)
conv3d.weight = nn.Parameter(weight)
conv3d.bias = nn.Parameter(bias)
conv3d = conv3d.to(torch.bfloat16).cuda()
conv3d.eval()  # Set to evaluation mode

def conv3d_flops(c_in, c_out, k_t, k_h, k_w, t, h, w):
    return 2 * k_t * k_h * k_w * c_in * c_out * t * h * w


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


# Warmup
warmup_iters = 5
with torch.no_grad():
    for _ in range(warmup_iters):
        with nvtx_range("warmup/conv3d_forward"):
            output = conv3d(input_tensor)
torch.cuda.synchronize()

# Benchmark
test_iters = 20
durations_ms = []

with torch.no_grad():
    for _ in range(test_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with nvtx_range("benchmark/conv3d_forward"):
            output = conv3d(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        durations_ms.append(start_event.elapsed_time(end_event))

# Stats
avg_ms = sum(durations_ms) / len(durations_ms)
min_ms = min(durations_ms)
max_ms = max(durations_ms)

b, c, t, h, w = input_tensor.shape
total_flops = conv3d_flops(c, k, 3, 3, 3, t, h, w)
avg_tflops = total_flops / (avg_ms / 1000) / 1e12

print(f"Output shape: {tuple(output.shape)}")
print(f"Conv3d(input_tensor) avg: {avg_ms:.3f} ms, min: {min_ms:.3f} ms, max: {max_ms:.3f} ms")
print(f"Estimated throughput: {avg_tflops:.2f} TFLOPS")
