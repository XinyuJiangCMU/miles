"""In-tree Triton block-wise FP8 (E4M3) dequant, cross-platform (NV + ROCm/gfx950).

Drop-in for ``tile_kernels.quant.per_token_cast_back`` (deepseek-ai/TileKernels, CUDA-only)
used by DSv4 QAT (``ops/qat.py``). Same fp8-load idiom as the in-tree
``tools/fp8_cast_bf16.py`` weight-dequant kernel: ``tl.load(fp8).to(tl.float32) * scale``.

``scale`` carries one fp32 value per (row, column-block of width ``block_size``):
    out[m, n] = float(y[m, n]) * scale[m, n // block_size].

TODO (see docs/DSV4_FLASH_ROCM_JOURNEY.md, AMD-kernel-isolation TODO 2): fold this into a
fork of deepseek-ai/TileKernels (replace the CUDA-only kernel with this triton one, pip
install the fork on ROCm). Then qat.py drops its import fork and this file goes away.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_cast_back_kernel(
    y_ptr,
    s_ptr,
    o_ptr,
    M,
    N,
    s_stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # row block
    pid_b = tl.program_id(axis=1)  # column block; one scale per row within it
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m_mask = rows < M
    offs = rows[:, None] * N + cols[None, :]
    x = tl.load(y_ptr + offs, mask=m_mask[:, None]).to(tl.float32)  # fp8 e4m3 -> fp32
    s = tl.load(s_ptr + rows * s_stride_m + pid_b, mask=m_mask)  # scale[rows, pid_b], fp32
    o = x * s[:, None]
    tl.store(o_ptr + offs, o.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])


def per_token_cast_back(x_fp8_and_scale, out_dtype: str, block_size: int) -> torch.Tensor:
    """Block-wise FP8 dequant: out[m, n] = float(y[m, n]) * scale[m, n // block_size].

    ``x_fp8_and_scale`` = (y, scale) with y (M, N) float8_e4m3 and scale (M, N // block_size)
    float32. ``out_dtype`` is "bf16" or "fp32".
    """
    y, scale = x_fp8_and_scale
    assert out_dtype in ("bf16", "fp32"), out_dtype
    assert y.dim() == 2 and scale.dim() == 2
    M, N = y.shape
    assert N % block_size == 0
    assert block_size & (block_size - 1) == 0, "block_size must be a power of 2 (Triton tile width)"
    assert scale.shape == (M, N // block_size)
    y = y.contiguous()
    scale = scale.contiguous()  # kernel gathers scale[:, pid_b] assuming column stride 1
    out = torch.empty(
        (M, N),
        dtype=torch.bfloat16 if out_dtype == "bf16" else torch.float32,
        device=y.device,
    )
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), N // block_size)
    _per_token_cast_back_kernel[grid](
        y, scale, out, M, N, scale.stride(0), BLOCK_M=BLOCK_M, BLOCK_SIZE=block_size
    )
    return out
