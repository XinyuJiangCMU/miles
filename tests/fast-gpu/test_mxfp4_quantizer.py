from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=[],
)


import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp4 import (
    _quantize_param as processor_quantize_mxfp4_param,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp4 import quantize_params_mxfp4

MXFP4_GROUP_SIZE = 32
MXFP4_SHAPES = [
    (1, 64),
    (1, 1024),
    (3, 128),
    (16, 64),
    (64, 128),
    (128, 64),
    (256, 128),
    (512, 256),
    (128, 1024),
    (1024, 2048),
    (7168, 2048),
    (2048, 7168),
    (128, 16384),
]


def _make_weight(init_data: str, dtype: torch.dtype, shape: tuple[int, int], device: str) -> torch.Tensor:
    m, n = shape
    if init_data == "random":
        return 16 * torch.randn((m, n), dtype=dtype, device=device)
    if init_data == "boundary":
        # Straddle the E2M1 decode bounds so rounding ties are exercised.
        base = torch.linspace(-6.0, 6.0, steps=n // 2, dtype=torch.float32, device=device)
        eps = torch.full_like(base, 1e-3)
        row = torch.empty(n, dtype=torch.float32, device=device)
        row[0::2] = base - eps
        row[1::2] = base + eps
        return row.unsqueeze(0).repeat(m, 1).to(dtype=dtype)
    if init_data == "zeros":
        return torch.zeros((m, n), dtype=dtype, device=device)
    if init_data == "maxes":
        return torch.full((m, n), torch.finfo(dtype).max, dtype=dtype, device=device)
    raise ValueError(f"Unknown init_data: {init_data}")


def _processor_quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized = dict(processor_quantize_mxfp4_param("model.layers.0.mlp.experts.0.down_proj.weight", weight))
    return (
        quantized["model.layers.0.mlp.experts.0.down_proj.weight"],
        quantized["model.layers.0.mlp.experts.0.down_proj.weight_scale"],
    )


# Independent OCP MXFP4 reference, matching the SGLang MXFP4Tensor reference
# (sglang/srt/layers/quantization/mxfp4_tensor.py, derived from NVIDIA
# TensorRT-Model-Optimizer). Written separately from the processor under test.
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_MAX = 6.0


def _mxfp4_reference(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def cast_fp4(x):
        sign = torch.sign(x)
        sign_bit = (2 - sign) // 2
        ord_ = torch.sum((x.abs().unsqueeze(-1) - _E2M1_BOUNDS.to(x.device)) > 0, dim=-1)
        return (sign_bit * 0b1000 + ord_).to(torch.uint8)

    def fuse(x):
        left = x[..., 0::2]
        right = x[..., 1::2]
        packed = right.clone() << 4
        packed[..., : left.shape[-1]] += left
        return packed

    m, n = weight.shape
    blocks = weight.to(torch.float32).view(-1, MXFP4_GROUP_SIZE)
    amax = blocks.abs().max(dim=-1, keepdim=True).values
    descale = amax / _E2M1_MAX
    e8 = torch.ceil(torch.maximum(torch.log2(descale), torch.tensor(-127.0, device=weight.device)))
    scaled = (blocks / torch.exp2(e8)).view(m, n)
    qweight = fuse(cast_fp4(scaled))
    scale = (e8 + 127).to(torch.uint8).view(m, n // MXFP4_GROUP_SIZE)
    return qweight, scale


def test_mxfp4_quantize_params_respects_extra_high_precision_layers_megatron():
    weight = torch.randn((4, MXFP4_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", weight),
    ]
    args = type("Args", (), {"extra_high_precision_layers_megatron": ("linear_fc2",)})()

    out = quantize_params_mxfp4(
        args=args,
        megatron_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "mxfp4"},
    )

    assert out is converted_named_params


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_mxfp4_quantize_params_respects_first_last_layers_bf16(layer_idx):
    weight = torch.randn((4, MXFP4_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", weight),
    ]
    args = type(
        "Args",
        (),
        {
            "first_last_layers_bf16": True,
            "num_layers": 4,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        },
    )()

    out = quantize_params_mxfp4(
        args=args,
        megatron_name=f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "mxfp4"},
    )

    assert out is converted_named_params


def test_mxfp4_quantize_params_quantizes_dense_attention():
    weight = torch.randn((MXFP4_GROUP_SIZE, MXFP4_GROUP_SIZE), dtype=torch.bfloat16)
    name = "model.layers.1.self_attn.qkv_proj.weight"
    out = quantize_params_mxfp4(
        args=type("Args", (), {})(),
        megatron_name="decoder.layers.1.self_attention.linear_qkv.weight",
        converted_named_params=[(name, weight)],
        quantization_config={"quant_method": "mxfp4"},
    )
    emitted = dict(out)
    assert name in emitted
    assert name.replace(".weight", ".weight_scale") in emitted
    assert emitted[name].dtype == torch.uint8


@pytest.mark.parametrize(
    "quantize_fn",
    [_processor_quantize_mxfp4],
    ids=["processor"],
)
@pytest.mark.parametrize("shape", MXFP4_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
def test_mxfp4_quantize_matches_reference(quantize_fn, shape, dtype, init_data):
    device = "cuda"
    torch.manual_seed(42)

    weight = _make_weight(init_data, dtype, shape, device)
    qweight, scale = quantize_fn(weight)
    qweight_ref, scale_ref = _mxfp4_reference(weight)

    assert qweight.shape == (*weight.shape[:-1], weight.shape[-1] // 2)
    assert qweight.dtype == torch.uint8
    assert scale.shape == (*weight.shape[:-1], weight.shape[-1] // MXFP4_GROUP_SIZE)
    assert scale.dtype == torch.uint8
    torch.testing.assert_close(qweight, qweight_ref, rtol=0, atol=0)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(128, 256), (512, 1024)])
def test_mxfp4_roundtrip_dequant_matches_sglang(shape):
    """Cross-check the emitted checkpoint dequantizes correctly via SGLang's
    independent loader-side dequantize (validates packing + E8M0 scale semantics)."""
    from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

    torch.manual_seed(0)
    weight = (16 * torch.randn(*shape, dtype=torch.bfloat16, device="cuda")).float()
    qweight, scale = _processor_quantize_mxfp4(weight)
    deq = MXFP4QuantizeUtil.dequantize(
        qweight, torch.float32, scale.reshape(-1, 1), [1, MXFP4_GROUP_SIZE]
    ).view(weight.shape)
    rel = (deq - weight).abs().max() / weight.abs().max()
    # FP4 has ~6-12% worst-case per-element error; just assert it dequantizes
    # in-range with no NaN/Inf (semantics correct), not bitwise.
    assert torch.isfinite(deq).all()
    assert rel < 0.2


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
