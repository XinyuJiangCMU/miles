#!/usr/bin/env python3
"""Quick diagnostic check for AMD MI300X environment compatibility with Miles.

Usage: HIP_VISIBLE_DEVICES=0 python tools/check_amd_env.py
"""
import sys


def check(name, fn):
    try:
        result = fn()
        if result is True:
            print(f"  [PASS] {name}")
            return True
        elif result is False:
            print(f"  [FAIL] {name}")
            return False
        else:
            print(f"  [INFO] {name}: {result}")
            return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main():
    import torch

    print("=== AMD MI300X Environment Check ===\n")
    print(f"PyTorch: {torch.__version__}")
    print(f"ROCm: {torch.version.hip}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"VRAM: {total / 1024**3:.0f} GB total, {free / 1024**3:.0f} GB free")
    print()

    passed = 0
    failed = 0

    print("--- Core ---")
    r = check("ROCm detected", lambda: bool(torch.version.hip))
    passed += r; failed += not r

    r = check("GPU available", lambda: torch.cuda.is_available())
    passed += r; failed += not r

    r = check("BF16 support", lambda: bool(torch.zeros(1, device="cuda", dtype=torch.bfloat16).sum() == 0))
    passed += r; failed += not r

    print("\n--- FP8 ---")
    r = check("FP8 e4m3fnuz dtype", lambda: bool(torch.zeros(1, dtype=torch.float8_e4m3fnuz).dtype == torch.float8_e4m3fnuz))
    passed += r; failed += not r

    r = check("scaled_mm FP8", lambda: bool(
        torch._scaled_mm(
            torch.zeros(8, 128, device="cuda", dtype=torch.float8_e4m3fnuz),
            torch.zeros(64, 128, device="cuda", dtype=torch.float8_e4m3fnuz).t(),
            scale_a=torch.tensor(1.0, device="cuda"),
            scale_b=torch.tensor(1.0, device="cuda"),
            out_dtype=torch.bfloat16,
        ).shape == (8, 64)
    ))
    passed += r; failed += not r

    r = check("Miles FP8 kernel", lambda: bool(__import__("miles.utils.fp8_kernel", fromlist=["fp8_dtype"]).fp8_dtype == torch.float8_e4m3fnuz))
    passed += r; failed += not r

    print("\n--- Training ---")
    r = check("Flash attention", lambda: bool(__import__("flash_attn")))
    passed += r; failed += not r

    r = check("TransformerEngine", lambda: bool(__import__("transformer_engine")))
    passed += r; failed += not r

    r = check("FSDP MoE backward", lambda: bool(__import__("miles.backends.fsdp_utils.kernels.fused_experts", fromlist=["GateUpProjFunction"])))
    passed += r; failed += not r

    r = check("AITER moe_sum", lambda: bool(__import__("aiter", fromlist=["moe_sum"])))
    passed += r; failed += not r

    print("\n--- Inference ---")
    r = check("SGLang installed", lambda: bool(__import__("sglang")))
    passed += r; failed += not r

    import os
    print("\n--- Environment ---")
    check("HIP_VISIBLE_DEVICES", lambda: os.environ.get("HIP_VISIBLE_DEVICES", "not set"))
    check("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", lambda: os.environ.get("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "not set"))
    check("SGLANG_MEMORY_SAVER_CUDA_GRAPH", lambda: os.environ.get("SGLANG_MEMORY_SAVER_CUDA_GRAPH", "not set"))

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
