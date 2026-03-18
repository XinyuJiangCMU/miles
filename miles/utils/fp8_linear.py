"""FP8 Linear layer for training on AMD MI300X.

Uses FP8 GEMM in forward pass with BF16 backward (STE).
FP8 GEMM is ~1.5-1.7x faster than BF16 on MI300X.

Usage:
    from miles.utils.fp8_linear import convert_model_to_fp8_training
    convert_model_to_fp8_training(model)
"""

import torch
import torch.nn as nn
from miles.utils.fp8_kernel import fp8_dtype, fp8_max


class _FP8MatmulFn(torch.autograd.Function):
    """FP8 forward, BF16 backward (Straight-Through Estimator)."""

    @staticmethod
    def forward(ctx, input_2d, weight, bias):
        # Per-tensor dynamic scaling with fused amax+cast
        # Use torch.amax (single kernel) instead of abs().max() (two kernels)
        i_amax = torch.amax(input_2d.abs()).float().clamp(min=1e-12)
        i_scale = (i_amax / fp8_max)
        input_fp8 = (input_2d.float() * (fp8_max / i_amax)).to(fp8_dtype)

        w_amax = torch.amax(weight.abs()).float().clamp(min=1e-12)
        w_scale = (w_amax / fp8_max)
        weight_fp8 = (weight.float() * (fp8_max / w_amax)).to(fp8_dtype)

        # FP8 GEMM: output = (input/i_scale) @ (weight/w_scale).T * i_scale * w_scale
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=i_scale,
            scale_b=w_scale,
            out_dtype=input_2d.dtype,
        )

        if bias is not None:
            output = output + bias

        # Save for backward (BF16 tensors for BF16 backward)
        ctx.save_for_backward(input_2d, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_2d, weight, bias = ctx.saved_tensors

        # Standard BF16 backward
        grad_input = grad_output @ weight  # (M, N) @ (N, K) = (M, K)
        grad_weight = grad_output.t() @ input_2d  # (N, M) @ (M, K) = (N, K)
        grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear that uses FP8 GEMM in forward pass."""

    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        if input.dim() > 2:
            input_2d = input.reshape(-1, input.shape[-1])
        else:
            input_2d = input

        output = _FP8MatmulFn.apply(input_2d, self.weight, self.bias)

        if len(orig_shape) > 2:
            output = output.reshape(*orig_shape[:-1], self.out_features)
        return output


def convert_model_to_fp8_training(
    model: nn.Module,
    min_size: int = 512,
    skip_lm_head: bool = True,
) -> nn.Module:
    """Convert nn.Linear layers to FP8Linear for FP8 training.

    Args:
        model: The model to convert (in-place).
        min_size: Minimum weight dimension to convert (skip small layers).
        skip_lm_head: Skip the language model head (output projection).
    """
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if skip_lm_head and "lm_head" in name:
                continue
            if module.in_features < min_size or module.out_features < min_size:
                continue
            modules_to_replace[name] = module

    for name, module in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], FP8Linear(module))

    return model
