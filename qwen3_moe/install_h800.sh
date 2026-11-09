#!/usr/bin/env bash
# Install only the optional expert kernel; preserve the pinned torch environment.
set -euo pipefail
python_bin=${PYTHON:-python}
if (( $# > 1 )); then
    echo 'Usage: bash qwen3_moe/install_h800.sh [/path/to/grouped_gemm-0.2.0.whl]' >&2
    exit 2
fi
"$python_bin" - <<'PY'
import sys
import torch
if sys.version_info[:2] != (3, 11) or torch.__version__.split('+')[0] != '2.6.0' or torch.version.cuda != '12.4':
    raise SystemExit('Activate the pinned Python 3.11 / torch 2.6.0+cu124 environment first')
print(dict(torch=torch.__version__, cuda=torch.version.cuda, cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI))
PY
if (( $# == 1 )); then
    [[ -f "$1" && "$1" == *grouped_gemm-0.2.0-*.whl ]] || { echo 'Expected a grouped_gemm 0.2.0 wheel' >&2; exit 2; }
    "$python_bin" -m pip install --no-deps --force-reinstall "$1"
else
    # CUDA-resident routing counts require the CUTLASS build, not cuBLAS fallback.
    GROUPED_GEMM_CUTLASS=1 TORCH_CUDA_ARCH_LIST=9.0 \
        "$python_bin" -m pip install --no-deps --force-reinstall --no-cache-dir \
        --no-build-isolation --no-binary=grouped_gemm grouped_gemm==0.2.0
fi
"$python_bin" -c 'import torch; from grouped_gemm.ops import gmm; print("grouped_gemm loaded; run qwen3_moe/verify_h800.py on the H800")'
