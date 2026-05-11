# Conv3D FP8

A PyTorch CUDA extension that implements 3D convolution in FP8 (E4M3) precision on top of [cuDNN](https://developer.nvidia.com/cudnn) via the [cuDNN frontend API](https://github.com/NVIDIA/cudnn-frontend). The kernel exposes a drop-in `conv3d` operator that consumes `float8_e4m3fn` inputs/weights with per-tensor descale factors and produces a high-precision output, targeting video / volumetric workloads such as 3D VAE encoders and decoders.

## Dependencies

This extension links against two NVIDIA libraries that must be installed before building:

1. **cuDNN** &mdash; <https://developer.nvidia.com/cudnn>
   The runtime library that provides the FP8 convolution engines. Version 9.x or newer is required for FP8 Conv3D support.
2. **cuDNN frontend** &mdash; <https://github.com/NVIDIA/cudnn-frontend>
   A header-only C++ library that builds and caches cuDNN execution plans. Clone the repository locally; only the `include/` directory is needed at build time.

In addition, the standard CUDA toolkit and a PyTorch build that exposes `torch.float8_e4m3fn` are required.

## Installation

1. Install or unpack cuDNN and clone the cuDNN frontend repository.
2. Open `setup.py` and update the include/library paths to point at your local installations. The relevant entries are:
   - `CUDNN_FRONTEND_INCLUDE` &mdash; path to `cudnn-frontend/include`
   - `CUDNN_LIBRARY_DIR` &mdash; directory containing `libcudnn.so*`
   - `CUDNN_LIBRARY` &mdash; (optional) absolute path to a specific `libcudnn.so.X` to link against
   These values can also be supplied via the corresponding environment variables.
3. Build and install the extension in editable mode:

   ```bash
   pip install -e . --no-build-isolation
   ```

   `--no-build-isolation` ensures the build picks up the active PyTorch installation rather than pulling a fresh copy into an isolated environment.

## Tests

| File | Purpose |
| --- | --- |
| `test_conv3d_fp8.py` | End-to-end functional / performance test case for the FP8 Conv3D kernel. |
| `test_conv3d_fp8_accuracy.py` | Numerical accuracy comparison between the FP8 kernel and a reference BF16 `torch.nn.functional.conv3d`. |

Run them with:

```bash
python test_conv3d_fp8.py
python test_conv3d_fp8_accuracy.py
```
