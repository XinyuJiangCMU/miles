"""Bit-exactness test for the in-tree Triton per_token_cast_back (amd/models/deepseek_v4/cast_back.py).

Imports the kernel DIRECTLY (bypassing the platform gate in ops/qat.py) so it runs on both
NV and ROCm — it only needs triton. Reference is the op's spec: float(fp8) * scale.
"""

import pytest
import torch

try:
    import triton  # noqa: F401
except ImportError:
    triton = None

if triton is not None:
    from miles_plugins.amd.models.deepseek_v4.cast_back import per_token_cast_back
else:
    per_token_cast_back = None


def _reference(y, scale, block_size, out_dtype):
    scale_b = scale.repeat_interleave(block_size, dim=1)  # (M, N)
    out = y.to(torch.float32) * scale_b
    return out.to(torch.bfloat16) if out_dtype == "bf16" else out


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="needs triton + a GPU (NV or ROCm)",
)
@pytest.mark.parametrize("out_dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("M", [512, 500])  # multiple / non-multiple of BLOCK_M (64) -> mask path
@pytest.mark.parametrize("N", [512, 384])  # multiple column-blocks of block_size
def test_per_token_cast_back_bit_exact(out_dtype, M, N):
    torch.manual_seed(0)
    block_size = 128
    y = torch.randn(M, N, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(M, N // block_size, device="cuda", dtype=torch.float32) + 0.5

    ref = _reference(y, scale, block_size, out_dtype)
    out = per_token_cast_back((y, scale), out_dtype, block_size)

    assert out.dtype == ref.dtype
    assert out.shape == (M, N)
    assert torch.equal(out, ref)
