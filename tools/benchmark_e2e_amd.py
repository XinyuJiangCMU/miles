#!/usr/bin/env python3
"""End-to-end AMD MI300X benchmark for Miles FP8 training pipeline.

Tests training throughput, memory usage, and FP8 kernel performance.

Usage:
    HIP_VISIBLE_DEVICES=0 python tools/benchmark_e2e_amd.py
    HIP_VISIBLE_DEVICES=0 python tools/benchmark_e2e_amd.py --model /data/Qwen2.5-7B-Instruct
"""

import argparse
import os
import sys
import time

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12398")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def benchmark_fp8_kernel():
    """Benchmark FP8 quantization kernel."""
    from miles.utils.fp8_kernel import blockwise_cast_to_fp8_triton

    M, N = 4096, 2560
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        blockwise_cast_to_fp8_triton(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(100):
        blockwise_cast_to_fp8_triton(x)
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) / 100 * 1000
    bw = M * N * 2 / (us / 1e6) / 1e12
    return {"fp8_quant_us": us, "fp8_quant_tb_s": bw}


def benchmark_gemm():
    """Benchmark BF16 and FP8 GEMM."""
    from miles.utils.fp8_kernel import fp8_dtype, fp8_max

    M, K, N = 2048, 2560, 9728
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # BF16 GEMM
    for _ in range(10):
        torch.mm(a, b.t())
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        torch.mm(a, b.t())
    e.record()
    torch.cuda.synchronize()
    bf16_us = s.elapsed_time(e) / 50 * 1000

    # FP8 GEMM
    a8 = a.to(fp8_dtype)
    b8 = b.to(fp8_dtype)
    sa = torch.tensor(1.0, device="cuda")
    for _ in range(10):
        torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sa, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    s.record()
    for _ in range(50):
        torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sa, out_dtype=torch.bfloat16)
    e.record()
    torch.cuda.synchronize()
    fp8_us = s.elapsed_time(e) / 50 * 1000

    tflops_bf16 = 2 * M * K * N / (bf16_us / 1e6) / 1e12
    tflops_fp8 = 2 * M * K * N / (fp8_us / 1e6) / 1e12

    return {
        "bf16_gemm_us": bf16_us,
        "fp8_gemm_us": fp8_us,
        "bf16_tflops": tflops_bf16,
        "fp8_tflops": tflops_fp8,
        "fp8_speedup": bf16_us / fp8_us,
    }


def benchmark_training(model_path, mbs=4, seq_len=512, warmup=3, iters=5):
    """Benchmark FSDP training throughput."""
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard
    from transformers import AutoModelForCausalLM

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda()
    model.train()
    for layer in model.model.layers:
        fully_shard(layer, reshard_after_forward=False)
    fully_shard(model, reshard_after_forward=False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
    x = torch.randint(0, 1000, (mbs, seq_len), device="cuda")

    for _ in range(warmup):
        out = model(x, labels=x)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        out = model(x, labels=x)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    e.record()
    torch.cuda.synchronize()

    ms = s.elapsed_time(e) / iters
    toks = mbs * seq_len
    free, total = torch.cuda.mem_get_info(0)
    mem_gb = (total - free) / 1024**3

    del model, opt
    torch.cuda.empty_cache()

    return {
        "step_ms": ms,
        "tok_s": toks / ms * 1000,
        "mem_gb": mem_gb,
        "tokens_per_step": toks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/Qwen3-4B")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.version.hip is None:
        print("This benchmark requires AMD ROCm GPU")
        sys.exit(1)

    banner("AMD MI300X End-to-End Benchmark")
    gpu = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {gpu}")
    print(f"VRAM: {total/1024**3:.0f}GB total, {free/1024**3:.0f}GB free")
    print(f"ROCm: {torch.version.hip}")
    print(f"Model: {args.model}")

    banner("FP8 Kernel Benchmark")
    r = benchmark_fp8_kernel()
    print(f"  FP8 quantization: {r['fp8_quant_us']:.0f}us ({r['fp8_quant_tb_s']:.2f} TB/s)")

    banner("GEMM Benchmark")
    r = benchmark_gemm()
    print(f"  BF16 GEMM: {r['bf16_gemm_us']:.0f}us ({r['bf16_tflops']:.1f} TFLOP/s)")
    print(f"  FP8 GEMM:  {r['fp8_gemm_us']:.0f}us ({r['fp8_tflops']:.1f} TFLOP/s)")
    print(f"  FP8 speedup: {r['fp8_speedup']:.2f}x")

    banner(f"Training Benchmark ({args.model})")
    for mbs in [1, 2, 4]:
        try:
            r = benchmark_training(args.model, mbs=mbs)
            print(f"  MBS={mbs}: {r['step_ms']:.0f}ms ({r['tok_s']:.0f} tok/s) mem={r['mem_gb']:.0f}GB")
        except torch.cuda.OutOfMemoryError:
            print(f"  MBS={mbs}: OOM")
            torch.cuda.empty_cache()

    banner("Summary")
    print("  See docs/amd_training_optimization.md for full optimization guide")


if __name__ == "__main__":
    main()
