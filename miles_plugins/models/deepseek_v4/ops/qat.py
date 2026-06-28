import torch

from .kernel.act_quant import act_quant


def per_token_cast_back(x_fp8_and_scale, out_dtype: str, block_size: int) -> torch.Tensor:
    """Block-wise FP8 dequantization (portable torch replacement).

    Originally ``tile_kernels.quant.per_token_cast_back`` (CUDA-only). Given the
    FP8 values ``y`` (M, N) and the per-block scales ``scale`` (M, N//block_size),
    reconstruct ``out[m, n] = float(y[m, n]) * scale[m, n // block_size]``.

    ``out_dtype`` is "bf16" or "fp32".
    """
    y, scale = x_fp8_and_scale
    M, N = y.shape
    n_blocks = N // block_size
    out = y.to(torch.float32).view(M, n_blocks, block_size) * scale.to(torch.float32).view(
        M, n_blocks, 1
    )
    out = out.view(M, N)
    return out.to(torch.bfloat16 if out_dtype == "bf16" else torch.float32)


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 (E4M3) cast + dequant with UE8M0 scaling.

    The cast (via :func:`act_quant`) runs through the in-tree TileLang kernel;
    the cast-back step is a portable torch block-wise dequant.
    """
    x_c = x.contiguous()
    y, scale = act_quant(x_c, block_size, "ue8m0")

    N = x_c.size(-1)
    y_flat = y.view(-1, N)
    scale_flat = scale.reshape(y_flat.size(0), N // block_size).contiguous()

    out_flat = per_token_cast_back((y_flat, scale_flat), "bf16" if x.dtype == torch.bfloat16 else "fp32", block_size)
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
