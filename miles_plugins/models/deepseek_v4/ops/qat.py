import torch

# per_token_cast_back comes from tile_kernels (deepseek-ai/TileKernels) on both platforms. On
# ROCm/gfx950 it JIT-compiles as a real TileLang kernel once tilelang's hip_fp8.h fp8->float path
# is patched to convert on-device (see docker/Dockerfile.rocm); NV stays byte-for-byte upstream.
from tile_kernels.quant import per_token_cast_back

from .kernel.act_quant import act_quant


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 (E4M3) cast + dequant with UE8M0 scaling.

    The cast runs through the in-tree TileLang :func:`act_quant`; the cast-back uses
    ``deepseek-ai/TileKernels`` :func:`per_token_cast_back` on both platforms.
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
