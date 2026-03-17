#!/usr/bin/env python3
"""Benchmark key operations on AMD MI300X for Miles FP8 pipeline.

Usage: HIP_VISIBLE_DEVICES=0 python tools/benchmark_amd.py
"""
import time

import torch


def benchmark(name, fn, n_warmup=5, n_iters=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_iters * 1000
    return elapsed


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"ROCm: {torch.version.hip}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM: {total / 1024**3:.0f} GB total, {free / 1024**3:.0f} GB free")
    print()

    # ===== FP8 Quantization =====
    print("=== FP8 Quantization ===")
    from miles.utils.fp8_kernel import blockwise_cast_to_fp8_triton, fp8_dtype

    for M, N in [(4096, 2560), (4096, 9728), (2048, 14336)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        ms = benchmark(f"blockwise {M}x{N}", lambda: blockwise_cast_to_fp8_triton(x))
        bw = M * N * 2 / (ms * 1e-3) / 1e12
        print(f"  blockwise {M:5d}x{N:5d}: {ms:.2f}ms ({bw:.2f} TB/s)")

    # ===== MoE Forward+Backward =====
    print("\n=== MoE Forward+Backward ===")
    from miles.backends.fsdp_utils.kernels.fused_experts import (
        DownProjFunction,
        GateUpProjFunction,
        MoeSumReduceFunction,
        SiluAndMulFunction,
        _DEFAULT_MOE_CONFIG,
    )

    print(f"  Config: {_DEFAULT_MOE_CONFIG}")

    E, N, K = 8, 9728, 2560
    topk = 2
    for num_tokens in [256, 512, 1024, 2048]:
        torch.manual_seed(42)
        h = torch.randn(num_tokens, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w1 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w2 = torch.randn(E, K, N // 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        tw = torch.randn(num_tokens, topk, device="cuda", dtype=torch.bfloat16).softmax(-1).requires_grad_(True)
        ti = torch.randint(0, E, (num_tokens, topk), device="cuda", dtype=torch.int32)

        def run():
            h.grad = w1.grad = w2.grad = tw.grad = None
            y1 = GateUpProjFunction.apply(h, w1, tw, ti)
            y2 = SiluAndMulFunction.apply(y1)
            y3 = DownProjFunction.apply(y2, w2, tw, ti)
            out = MoeSumReduceFunction.apply(y3, (num_tokens, K))
            out.sum().backward()

        ms = benchmark(f"MoE {num_tokens}", run, n_iters=20)
        print(f"  tokens={num_tokens:5d}: {ms:.2f}ms")

    # ===== Flash Attention =====
    print("\n=== Flash Attention ===")
    from flash_attn import flash_attn_func

    B, H, D = 4, 32, 80
    num_kv = 8
    for S in [256, 512, 1024, 2048]:
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, S, num_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, S, num_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def run():
            q.grad = k.grad = v.grad = None
            out = flash_attn_func(q, k, v, causal=True)
            out.sum().backward()

        ms = benchmark(f"FA S={S}", run, n_iters=30)
        print(f"  seq_len={S:5d}: {ms:.2f}ms")

    # ===== scaled_mm =====
    print("\n=== FP8 scaled_mm ===")
    for M, N, K in [(4096, 2560, 2560), (4096, 9728, 2560), (2048, 2560, 9728)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(fp8_dtype)
        b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(fp8_dtype)
        sa = torch.tensor(1.0, device="cuda")
        sb = torch.tensor(1.0, device="cuda")

        ms = benchmark(
            f"scaled_mm {M}x{K}x{N}",
            lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16),
        )
        tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
        print(f"  {M:5d}x{K:5d}x{N:5d}: {ms:.2f}ms ({tflops:.1f} TFLOP/s)")

    print("\nDone!")


if __name__ == "__main__":
    main()
