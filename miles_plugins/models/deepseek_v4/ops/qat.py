import torch

# per_token_cast_back: tile_kernels (deepseek-ai/TileKernels) is CUDA-only, so on ROCm use the
# in-tree Triton dequant instead (cross-platform, same float(fp8) * scale math as
# tools/fp8_cast_bf16.py). NV (torch.version.hip is None) keeps tile_kernels byte-for-byte.
# Same platform-gate idiom as hyper_connection.py.
if torch.version.hip is not None:
    from miles_plugins.amd.models.deepseek_v4.cast_back import per_token_cast_back
else:
    from tile_kernels.quant import per_token_cast_back

from .kernel.act_quant import act_quant


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 (E4M3) cast + dequant with UE8M0 scaling.

    The cast runs through the in-tree TileLang :func:`act_quant`. The cast-back uses
    ``deepseek-ai/TileKernels`` on CUDA and the in-tree Triton ``per_token_cast_back``
    on ROCm (selected by the platform gate above).
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
