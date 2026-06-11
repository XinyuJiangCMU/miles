# SPDX-License-Identifier: Apache-2.0
#
# AMD MXFP4 (OCP microscaling) weight quantizer for the megatron->HF export path.
#
# Sibling of quantizer_mxfp8.py. Both target the open OCP microscaling format
# (block size 32, per-block E8M0 power-of-two scale, no per-tensor global scale),
# which is what AMD Instinct MI350 (gfx950) runs -- NOT NVIDIA's proprietary
# NVFP4 (block 16, FP8-E4M3 block scale + per-tensor FP32 global scale).
#
# The only structural difference from MXFP8 is the element format and packing:
#   MXFP8: element = FP8 E4M3, stored 1 byte/element, qweight shape (M, K)
#   MXFP4: element = FP4 E2M1, packed 2/byte,         qweight shape (M, K//2)
# Both emit an E8M0 block scale of shape (M, K // 32).
#
# The block-quantization kernel is a vendored pure-torch reference (no
# TransformerEngine dependency), bitwise-matched to the SGLang MXFP4Tensor
# reference (sglang/srt/layers/quantization/mxfp4_tensor.py, Apache-2.0,
# derived from NVIDIA TensorRT-Model-Optimizer). It runs identically on ROCm
# and CUDA.

import re

import torch

FP4_E2M1_MAX = 6.0
MXFP4_GROUP_SIZE = 32  # OCP microscaling block size

# E2M1 magnitude thresholds between adjacent fp4 codes.
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def quantize_params_mxfp4(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "mxfp4"

    if getattr(args, "extra_high_precision_layers_megatron", False):
        for layer_name in getattr(args, "extra_high_precision_layers_megatron", ()):
            if layer_name in megatron_name:
                return converted_named_params

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.search(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"mtp\.layers\.(\d+)\.(.+)"
        match = re.search(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # Skip quantization for BF16 head/tail of main decoder layers.
    if getattr(args, "first_last_layers_bf16", False):
        num_layers = int(args.num_layers)
        num_layers_at_start_in_bf16 = int(getattr(args, "num_layers_at_start_in_bf16", 0))
        num_layers_at_end_in_bf16 = int(getattr(args, "num_layers_at_end_in_bf16", 0))
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_layers - num_layers_at_end_in_bf16
        if int(layer_idx) < head_end_idx or int(layer_idx) >= tail_start_idx:
            return converted_named_params

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, _expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip already-emitted bf16 scale params
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(_quantize_param(converted_name, param))
            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(_quantize_param(converted_name, param))
            return quantize_named_params

    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        "self_attention.wq_b.weight",
        "self_attention.wk.weight",
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(_quantize_param(converted_name, param))
        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_param(name, weight):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {MXFP4_GROUP_SIZE} for MXFP4.")
    weight_flat = weight.view(-1, k).contiguous()
    qweight, scale = _quantize_mxfp4_2d(weight_flat)
    qweight = qweight.view(*weight.shape[:-1], k // 2)
    scale = scale.view(*weight.shape[:-1], k // MXFP4_GROUP_SIZE).contiguous()
    scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]


def _cast_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Quantize block-scaled values to E2M1 fp4 codes (uint8, one code per element).

    Sign bit is 0b1000; magnitude is the count of bounds exceeded (0..7).
    """
    sign = torch.sign(x)
    sign_bit = (2 - sign) // 2  # +1 -> 0, -1/0 -> 1
    ord_ = torch.sum((x.abs().unsqueeze(-1) - _E2M1_BOUNDS.to(x.device)) > 0, dim=-1)
    return (sign_bit * 0b1000 + ord_).to(torch.uint8)


def _fuse_uint4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Pack pairs of fp4 codes into bytes: even index -> low nibble, odd -> high nibble."""
    left = x[..., 0::2]
    right = x[..., 1::2]
    packed = right.clone() << 4
    packed[..., : left.shape[-1]] += left
    return packed


def _quantize_mxfp4_2d(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 (OCP) quantization for a 2D weight.

    Returns:
      qweight: uint8 packed fp4, shape (M, K // 2)
      block_scale: E8M0 (uint8, biased by 127), shape (M, K // MXFP4_GROUP_SIZE)
    """
    weight = weight.contiguous()
    m, n = weight.shape
    if n % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"MXFP4 requires K divisible by {MXFP4_GROUP_SIZE}, got {n}.")

    blocks = weight.to(torch.float32).view(m, n // MXFP4_GROUP_SIZE, MXFP4_GROUP_SIZE)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)  # (m, n//32, 1)

    descale = block_amax / FP4_E2M1_MAX
    # E8M0 exponent: ceil(log2(descale)), floored at -127 (E8M0 min). amax==0 ->
    # log2(0) = -inf -> clamped to -127 (smallest scale); the all-zero block
    # quantizes to the fp4 zero code anyway.
    min_exp = torch.tensor(-127.0, device=weight.device)
    e8m0_exp = torch.ceil(torch.maximum(torch.log2(descale), min_exp))

    scaled = (blocks / torch.exp2(e8m0_exp)).view(m, n)
    codes = _cast_to_fp4_e2m1(scaled)
    qweight = _fuse_uint4_to_uint8(codes)  # (m, n//2)

    block_scale = (e8m0_exp + 127).to(torch.uint8).view(m, n // MXFP4_GROUP_SIZE)
    return qweight, block_scale


def quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 quantize a 2D weight or a stack of 2D weights (rank-3 grouped experts)."""
    if weight.dim() == 2:
        return _quantize_mxfp4_2d(weight)
    if weight.dim() == 3:
        qweights = []
        block_scales = []
        for idx in range(weight.shape[0]):
            qweight, block_scale = _quantize_mxfp4_2d(weight[idx])
            qweights.append(qweight)
            block_scales.append(block_scale)
        return torch.stack(qweights, dim=0), torch.stack(block_scales, dim=0)
    raise ValueError(f"Unsupported weight rank {weight.dim()} for MXFP4 quantization.")
