#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/../third_party/mega_sam/base"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: Please activate the conda environment first, for example: conda activate pdi-bench"
    exit 1
fi

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")

CUDA_INCLUDE_DIRS=(
    "$CUDA_HOME/include"
    "$CUDA_HOME/targets/x86_64-linux/include"
)
CUDA_LIBRARY_DIRS=(
    "$CUDA_HOME/lib"
    "$CUDA_HOME/targets/x86_64-linux/lib"
    "$CUDA_HOME/targets/x86_64-linux/lib/stubs"
    "$SITE_PKG/torch/lib"
)

while IFS= read -r dir; do
    CUDA_INCLUDE_DIRS+=("$dir")
done < <(find "$SITE_PKG/nvidia" -maxdepth 2 -type d -name include 2>/dev/null | sort)

while IFS= read -r dir; do
    CUDA_LIBRARY_DIRS+=("$dir")
done < <(find "$SITE_PKG/nvidia" -maxdepth 2 -type d \( -name lib -o -name lib64 \) 2>/dev/null | sort)

append_unique_path() {
    local var_name="$1"
    local dir="$2"
    local current="${!var_name:-}"
    case ":$current:" in
        *":$dir:"*) ;;
        "")
            printf -v "$var_name" "%s" "$dir"
            ;;
        *)
            printf -v "$var_name" "%s:%s" "$dir" "$current"
            ;;
    esac
}

for dir in "${CUDA_INCLUDE_DIRS[@]}"; do
    [ -d "$dir" ] || continue
    append_unique_path CPATH "$dir"
    append_unique_path CPLUS_INCLUDE_PATH "$dir"
done

for dir in "${CUDA_LIBRARY_DIRS[@]}"; do
    [ -d "$dir" ] || continue
    append_unique_path LD_LIBRARY_PATH "$dir"
    append_unique_path LIBRARY_PATH "$dir"
done

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Install CUDA 11.8 toolkit with conda before building."
    exit 1
fi

CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
if [ "$CUDA_VERSION" != "11.8" ]; then
    echo "ERROR: Expected CUDA 11.8 from the conda environment, but nvcc reports CUDA $CUDA_VERSION"
    echo "CUDA_HOME=$CUDA_HOME"
    echo "nvcc=$(command -v nvcc)"
    exit 1
fi

TORCH_CUDA_VERSION="$(python -c "import torch; print(torch.version.cuda)")"
if [ "$TORCH_CUDA_VERSION" != "11.8" ]; then
    echo "ERROR: Expected PyTorch cu118, but torch.version.cuda is $TORCH_CUDA_VERSION"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: $BASE_DIR not found. Did you run: git submodule update --init --recursive ?"
    exit 1
fi

cd "$BASE_DIR"

CUDA_COMPAT_DIR="$BASE_DIR/.cuda_compat"
mkdir -p "$CUDA_COMPAT_DIR"

link_cuda_compat() {
    local link_name="$1"
    shift
    local target=""
    for candidate in "$@"; do
        if [ -f "$candidate" ]; then
            target="$candidate"
            break
        fi
    done
    if [ -n "$target" ]; then
        ln -sfn "$target" "$CUDA_COMPAT_DIR/$link_name"
    fi
}

link_cuda_compat "libcudart.so" \
    "$SITE_PKG/nvidia/cuda_runtime/lib/libcudart.so.11.0" \
    "$SITE_PKG/torch/lib/libcudart-d0da41ae.so.11.0" \
    "$CUDA_HOME/targets/x86_64-linux/lib/libcudart.so.12"
link_cuda_compat "libcublas.so" \
    "$SITE_PKG/nvidia/cublas/lib/libcublas.so.11" \
    "$SITE_PKG/torch/lib/libcublas.so.11" \
    "$CUDA_HOME/targets/x86_64-linux/lib/libcublas.so.12"
link_cuda_compat "libcublasLt.so" \
    "$SITE_PKG/nvidia/cublas/lib/libcublasLt.so.11" \
    "$SITE_PKG/torch/lib/libcublasLt.so.11" \
    "$CUDA_HOME/targets/x86_64-linux/lib/libcublasLt.so.12"

append_unique_path LD_LIBRARY_PATH "$CUDA_COMPAT_DIR"
append_unique_path LIBRARY_PATH "$CUDA_COMPAT_DIR"
export CUDA_EXTENSION_LIBRARY_DIRS="$CUDA_COMPAT_DIR"
for dir in "${CUDA_LIBRARY_DIRS[@]}"; do
    [ -d "$dir" ] || continue
    CUDA_EXTENSION_LIBRARY_DIRS="$CUDA_EXTENSION_LIBRARY_DIRS${CUDA_EXTENSION_LIBRARY_DIRS:+:}$dir"
done

SETUP_BACKUP="$(mktemp)"
cp setup.py "$SETUP_BACKUP"
restore_setup() {
    cp "$SETUP_BACKUP" setup.py
    rm -f "$SETUP_BACKUP"
}
trap restore_setup EXIT

echo "=== Step 1/4: Building droid_backends ==="
cat > setup.py << 'SETUP_EOF'
import os.path as osp
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))
LIB_DIRS = [p for p in os.environ.get('CUDA_EXTENSION_LIBRARY_DIRS', '').split(os.pathsep) if p]

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension(
            'droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            library_dirs=LIB_DIRS,
            runtime_library_dirs=LIB_DIRS,
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
SETUP_EOF

pip install -e . --no-build-isolation

echo "=== Step 2/4: Copying droid_backends.so to site-packages ==="
cp droid_backends*.so "$SITE_PKG/"

echo "=== Step 3/4: Building lietorch ==="
cat > setup.py << 'SETUP_EOF'
import os.path as osp
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))
LIB_DIRS = [p for p in os.environ.get('CUDA_EXTENSION_LIBRARY_DIRS', '').split(os.pathsep) if p]

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension(
            'lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen'),
            ],
            library_dirs=LIB_DIRS,
            runtime_library_dirs=LIB_DIRS,
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp',
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
SETUP_EOF

pip install -e . --no-build-isolation

echo "=== Step 4/4: Copying lietorch_backends.so to site-packages ==="
cp thirdparty/lietorch/lietorch_backends*.so "$SITE_PKG/"

echo ""
echo "=== Verifying ==="
python -c "import droid_backends; print('droid_backends OK')"
python -c "from lietorch import SE3; p = SE3.Identity(1, device='cuda'); p.inv(); print('lietorch OK')"
echo "All done."
