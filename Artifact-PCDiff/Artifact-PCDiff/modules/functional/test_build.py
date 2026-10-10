import os
import sys
import torch
from torch.utils.cpp_extension import load
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
# Xử lý đường dẫn thư viện Torch trên Windows
if sys.platform == 'win32':
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib_path) and torch_lib_path not in os.environ['PATH']:
        print(f"Appending torch lib path: {torch_lib_path}")
        os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ['PATH']
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    cuda_bin_path = None
    if cuda_home:
        cuda_bin_path = os.path.join(cuda_home, 'bin')
    else:
        # Fallback: try a common default path if CUDA_HOME/CUDA_PATH is not set.
        # The build log indicates CUDA v12.5 was used.
        default_cuda_path_version = "12.5" # From build log
        default_cuda_base = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{default_cuda_path_version}"
        if os.path.exists(default_cuda_base):
            cuda_bin_path = os.path.join(default_cuda_base, 'bin')
            print(f"Warning: CUDA_HOME or CUDA_PATH environment variable not found. Using fallback path: {cuda_bin_path}")
        else:
            print(f"Warning: CUDA_HOME or CUDA_PATH environment variable not found, and fallback path {default_cuda_base} does not exist.")

    if cuda_bin_path and os.path.exists(cuda_bin_path) and cuda_bin_path not in os.environ['PATH']:
        print(f"Appending CUDA bin path to os.environ['PATH']: {cuda_bin_path}")
        os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']
    elif cuda_bin_path and not os.path.exists(cuda_bin_path):
        print(f"Warning: Deduced CUDA bin path does not exist: {cuda_bin_path}")

    extra_cflags = ['/O2', '/std:c++17']
else:
    extra_cflags = ['-O3', '-std=c++17']

# Lấy đường dẫn gốc của file
_src_path = os.path.dirname(os.path.abspath(__file__))

# Biên dịch extension
_backend = load(
    name='_pvcnn_backend',
    extra_cflags=extra_cflags,
    extra_cuda_cflags=['-O3'],  # bạn có thể thêm '--use_fast_math' nếu muốn
    verbose=True,
    sources=[os.path.join(_src_path, 'src', f) for f in [
        'ball_query/ball_query.cpp',
        'ball_query/ball_query.cu',
        'grouping/grouping.cpp',
        'grouping/grouping.cu',
        'interpolate/neighbor_interpolate.cpp',
        'interpolate/neighbor_interpolate.cu',
        'interpolate/trilinear_devox.cpp',
        'interpolate/trilinear_devox.cu',
        'sampling/sampling.cpp',
        'sampling/sampling.cu',
        'voxelization/vox.cpp',
        'voxelization/vox.cu',
        'bindings.cpp',
    ]]
)

__all__ = ['_backend']
