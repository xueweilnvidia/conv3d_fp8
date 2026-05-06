from pathlib import Path
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent
CUDNN_FRONTEND_INCLUDE = Path(os.getenv("CUDNN_FRONTEND_INCLUDE", "/workdir/tmp/cudnn-frontend/include"))
CUDNN_LIBRARY = os.getenv("CUDNN_LIBRARY")
CUDNN_LIBRARY_NAME = os.getenv("CUDNN_LIBRARY_NAME", "cudnn")
CUDNN_LIBRARY_DIR = os.getenv("CUDNN_LIBRARY_DIR")

include_dirs = [str(CUDNN_FRONTEND_INCLUDE)]
library_dirs = []
libraries = []
extra_link_args = []

if CUDA_HOME:
    include_dirs.append(str(Path(CUDA_HOME) / "include"))
    library_dirs.append(str(Path(CUDA_HOME) / "lib64"))
    if not CUDNN_LIBRARY_DIR:
        CUDNN_LIBRARY_DIR = str(Path(CUDA_HOME) / "lib64")

if CUDNN_LIBRARY_DIR:
    library_dirs.append(CUDNN_LIBRARY_DIR)

if CUDNN_LIBRARY:
    # Explicitly link a specific cuDNN shared library (e.g., /path/libcudnn.so.9).
    extra_link_args.append(CUDNN_LIBRARY)
else:
    libraries.append(CUDNN_LIBRARY_NAME)

# cuDNN frontend plan building may require NVRTC symbols at load time.
if "nvrtc" not in libraries:
    libraries.append("nvrtc")


setup(
    name="conv3d-fp8",
    version="0.1.0",
    description="Conv3d FP8 wrapper and CUDA extension",
    py_modules=["conv3d_fp8", "conv3d_fp8_op"],
    ext_modules=[
        CUDAExtension(
            name="conv3d_fp8_ext",
            sources=[str(ROOT / "conv3d_fp8_ext.cpp")],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args,
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
