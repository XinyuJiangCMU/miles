#!/bin/bash
# Upgrade Triton 3.4 -> 3.7 (AMD ROCm 7.0.0 native) inside a DSv4-Flash rollout container,
# to unlock the DSA paged-MQA "preshuffle" optimized decode path (page_size=64) instead of
# the slow legacy page_size=1 fallback.
#
# Root cause of the fallback (before this): container ships Triton 3.4.0; the aiter gluon
# paged-MQA kernels require Triton>=3.6. The env AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1 alone
# does NOT help on 3.4 (it switches to an AOT-load path whose prebuilt zip is not in the image
# -> crashes). So we upgrade Triton.
#
# Findings (validated on gfx950 / MI355X, ROCm 7.0.0, torch 2.9.0a0):
#   - The aiter install_triton.sh guards on torch>=2.9.1 and skips otherwise. That guard is
#     overly conservative: pip-installing Triton 3.7 does NOT touch torch and works with 2.9.0a0.
#   - Triton 3.6 alone breaks: the generic triton-kernels 1.0.0 uses tl.constexpr_function
#     (a 3.4 API that moved to triton.* top-level in 3.6). Fix = install the AMD ROCm-native
#     MATCHED PAIR (same git hash) below.
#   - After upgrade, torch._inductor (built for older Triton) crashes on inductor-generated
#     elementwise kernels: triton_heuristics.py accesses binary.metadata.cluster_dims which
#     Triton>=3.6 removed (cluster_dims/num_ctas are CUDA thread-block-cluster concepts, N/A on
#     ROCm). We patch that one line to default gracefully. Semantically correct on gfx950.
#
# Run INSIDE each rollout container (all nodes). Idempotent.
set -euo pipefail

INDEX="https://pypi.amd.com/triton/release_/rocm-7.0.0/simple/"
TRITON_VER="3.7.0+amd.rocm7.0.0.gitd0d77a509"
TK_VER="1.0.0+amd.rocm7.0.0.gitd0d77a509"

echo "[triton37] installing AMD ROCm-native matched pair triton==${TRITON_VER} + triton-kernels==${TK_VER}"
python3 -m pip install --extra-index-url "$INDEX" "triton==${TRITON_VER}" "triton-kernels==${TK_VER}"

echo "[triton37] patching torch._inductor cluster_dims for Triton>=3.6 compat"
python3 - <<'PY'
import torch, os
f = os.path.join(os.path.dirname(torch.__file__),
                 "_inductor", "runtime", "triton_heuristics.py")
s = open(f).read()
old = '(binary.metadata.num_ctas, *binary.metadata.cluster_dims)'
new = '(getattr(binary.metadata, "num_ctas", 1), *getattr(binary.metadata, "cluster_dims", ()))'
if new in s:
    print("  already patched")
elif old in s:
    open(f, "w").write(s.replace(old, new)); print("  patched")
else:
    raise SystemExit("  WARN: cluster_dims pattern not found (torch version drift?)")
PY

echo "[triton37] verify"
python3 - <<'PY'
import torch, triton
print("  torch", torch.__version__, "triton", triton.__version__)
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa as f
print("  aiter_can_use_preshuffle_paged_mqa() =", f(), "(True = optimized paged-MQA path unlocked)")
PY
echo "[triton37] done. Restart Ray so worker processes pick up the new Triton."
