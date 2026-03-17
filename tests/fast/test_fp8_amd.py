"""Fast FP8 tests for AMD MI300X compatibility.

Run with: HIP_VISIBLE_DEVICES=7 python tests/fast/test_fp8_amd.py
"""
import os
import sys

import torch


def test_fp8_dtype():
    """Test that the correct FP8 dtype is selected for the platform."""
    from miles.utils.fp8_kernel import fp8_dtype

    if torch.version.hip:
        assert fp8_dtype == torch.float8_e4m3fnuz, f"Expected e4m3fnuz on AMD, got {fp8_dtype}"
    else:
        assert fp8_dtype == torch.float8_e4m3fn, f"Expected e4m3fn on NVIDIA, got {fp8_dtype}"
    print("PASS: FP8 dtype selection")


def test_blockwise_cast():
    """Test blockwise FP8 casting on GPU."""
    from miles.utils.fp8_kernel import blockwise_cast_to_fp8_triton, fp8_dtype

    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    y, s = blockwise_cast_to_fp8_triton(x)
    assert y.dtype == fp8_dtype
    assert y.shape == x.shape
    assert s.shape == (2, 4)  # 256/128, 512/128

    # Test with custom block size
    y2, s2 = blockwise_cast_to_fp8_triton(x, block_size=[64, 64])
    assert s2.shape == (4, 8)
    print("PASS: Blockwise FP8 cast")


def test_scaled_mm():
    """Test that scaled_mm works with the platform FP8 dtype."""
    from miles.utils.fp8_kernel import fp8_dtype

    a = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16).to(fp8_dtype)
    b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16).to(fp8_dtype)
    scale_a = torch.tensor(1.0, device="cuda")
    scale_b = torch.tensor(1.0, device="cuda")
    out = torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    assert out.shape == (8, 64)
    print("PASS: scaled_mm")


def test_normalize_e4m3fn_to_e4m3fnuz():
    """Test FP8 format conversion."""
    from miles.utils.fp8_kernel import normalize_e4m3fn_to_e4m3fnuz

    w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s = torch.tensor([0.01], device="cuda")
    w_out, s_out = normalize_e4m3fn_to_e4m3fnuz(w, s)
    assert w_out.dtype == torch.float8_e4m3fnuz
    assert s_out is not None
    # Scale should be adjusted by max ratio (448/240 ≈ 1.867)
    ratio = s_out.item() / s.item()
    assert 1.8 < ratio < 1.95, f"Unexpected scale ratio: {ratio}"
    print("PASS: normalize_e4m3fn_to_e4m3fnuz")


def test_per_tensor_quant():
    """Test per-tensor FP8 quantization."""
    from miles.utils.fp8_kernel import fp8_dtype

    w = torch.randn(2560, 9728, device="cuda", dtype=torch.bfloat16)
    FP8_MAX = torch.finfo(fp8_dtype).max
    scale = w.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qw = (w / scale).clamp(min=-FP8_MAX, max=FP8_MAX).to(fp8_dtype)
    assert qw.dtype == fp8_dtype
    assert qw.shape == w.shape

    # Verify reconstruction quality
    w_recon = qw.to(torch.float32) * scale
    cosine = torch.nn.functional.cosine_similarity(w.float().flatten(), w_recon.flatten(), dim=0)
    assert cosine > 0.99, f"Poor reconstruction: cosine={cosine.item():.4f}"
    print("PASS: Per-tensor FP8 quantization")


def test_transformer_engine():
    """Test TransformerEngine FP8 on AMD."""
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch import fp8_autocast

        linear = te.Linear(128, 256, bias=True).cuda()
        x = torch.randn(16, 128, device="cuda")
        with fp8_autocast():
            y = linear(x)
        loss = y.sum()
        loss.backward()
        assert y.shape == (16, 256)
        print("PASS: TransformerEngine FP8")
    except ImportError:
        print("SKIP: TransformerEngine not installed")


def test_gpu_id_mapping():
    """Test GPU ID mapping for AMD."""
    from miles.backends.sglang_utils.sglang_engine import _to_local_gpu_id

    # When HIP_VISIBLE_DEVICES is set
    old = os.environ.get("HIP_VISIBLE_DEVICES")
    old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["HIP_VISIBLE_DEVICES"] = "4,5,6,7"
        assert _to_local_gpu_id(4) == 0
        assert _to_local_gpu_id(7) == 3
        assert _to_local_gpu_id(0) == 0  # local index
        print("PASS: GPU ID mapping")
    finally:
        if old:
            os.environ["HIP_VISIBLE_DEVICES"] = old
        else:
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        if old_cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd


def test_moe_sum_reduce():
    """Test that MoE sum reduce uses AITER on AMD."""
    from miles.backends.fsdp_utils.kernels.fused_experts import moe_sum_reduce

    x = torch.randn(32, 2, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(32, 256, device="cuda", dtype=torch.bfloat16)
    moe_sum_reduce(x, out, 1.0)
    ref = x.sum(dim=1)
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 0.01, f"moe_sum_reduce diff too large: {diff}"
    print("PASS: MoE sum reduce")


def test_moe_config_amd():
    """Test that AMD-optimized MoE config is used."""
    from miles.backends.fsdp_utils.kernels.fused_experts import _DEFAULT_MOE_CONFIG, _IS_AMD

    if _IS_AMD:
        assert _DEFAULT_MOE_CONFIG["BLOCK_SIZE_K"] == 128, f"Expected K=128, got {_DEFAULT_MOE_CONFIG['BLOCK_SIZE_K']}"
        assert _DEFAULT_MOE_CONFIG["BLOCK_SIZE_N"] == 128, f"Expected N=128, got {_DEFAULT_MOE_CONFIG['BLOCK_SIZE_N']}"
        print("PASS: AMD-optimized MoE config")
    else:
        print("SKIP: Not on AMD")


def test_check_has_nvlink():
    """Test that check_has_nvlink returns False on AMD."""
    from miles.utils.external_utils.command_utils import check_has_nvlink

    if torch.version.hip:
        assert check_has_nvlink() is False
        print("PASS: check_has_nvlink returns False on AMD")
    else:
        print("SKIP: Not on AMD")


def test_fused_moe_backward():
    """Test FSDP MoE forward+backward on AMD."""
    from miles.backends.fsdp_utils.kernels.fused_experts import (
        DownProjFunction,
        GateUpProjFunction,
        MoeSumReduceFunction,
        SiluAndMulFunction,
    )

    E, N, K = 4, 1024 * 2, 512
    num_tokens, topk = 16, 2
    hidden = torch.randn(num_tokens, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(E, K, N // 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    topk_w = torch.randn(num_tokens, topk, device="cuda", dtype=torch.bfloat16).softmax(-1).requires_grad_(True)
    topk_ids = torch.randint(0, E, (num_tokens, topk), device="cuda", dtype=torch.int32)

    y1 = GateUpProjFunction.apply(hidden, w1, topk_w, topk_ids)
    y2 = SiluAndMulFunction.apply(y1)
    y3 = DownProjFunction.apply(y2, w2, topk_w, topk_ids)
    out = MoeSumReduceFunction.apply(y3, (num_tokens, K))
    out.sum().backward()
    assert hidden.grad is not None and hidden.grad.norm() > 0
    print("PASS: Fused MoE backward")


def test_flash_attention():
    """Test flash attention on AMD."""
    from flash_attn import flash_attn_func

    q = torch.randn(1, 8, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=True)
    out.sum().backward()
    assert q.grad is not None
    print("PASS: Flash attention fwd+bwd")


if __name__ == "__main__":
    print(f"Platform: {'ROCm' if torch.version.hip else 'CUDA'}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("---")

    tests = [
        test_fp8_dtype,
        test_blockwise_cast,
        test_scaled_mm,
        test_normalize_e4m3fn_to_e4m3fnuz,
        test_per_tensor_quant,
        test_transformer_engine,
        test_gpu_id_mapping,
        test_moe_sum_reduce,
        test_moe_config_amd,
        test_check_has_nvlink,
        test_fused_moe_backward,
        test_flash_attention,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n--- Results: {passed} passed, {failed} failed ---")
    sys.exit(1 if failed > 0 else 0)
