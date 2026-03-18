#!/usr/bin/env python3
"""Benchmark FSDP training throughput on AMD MI300X.

Tests different optimization configurations and reports results.

Usage:
    HIP_VISIBLE_DEVICES=0 python tools/benchmark_training_amd.py --model /data/Qwen3-4B
    HIP_VISIBLE_DEVICES=0 python tools/benchmark_training_amd.py --model /data/Qwen2.5-7B-Instruct
"""

import argparse
import os
import sys

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("NCCL_BUFFSIZE", "16777216")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist


def benchmark_config(model_path, mbs, seq_len, gc, fused, reshard, bf16_reduce, warmup=3, iters=5):
    """Benchmark a single configuration."""
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda()
    model.train()
    if gc:
        model.gradient_checkpointing_enable()

    reduce_dtype = torch.bfloat16 if bf16_reduce else torch.float32
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype)
    for layer in model.model.layers:
        fully_shard(layer, reshard_after_forward=reshard, mp_policy=mp)
    fully_shard(model, reshard_after_forward=reshard, mp_policy=mp)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=fused)
    x = torch.randint(0, 1000, (mbs, seq_len), device="cuda")

    try:
        for _ in range(warmup):
            out = model(x, labels=x)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            out = model(x, labels=x)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
        e.record()
        torch.cuda.synchronize()

        ms = s.elapsed_time(e) / iters
        free, total = torch.cuda.mem_get_info(0)
        mem_gb = (total - free) / 1024**3
        toks = mbs * seq_len
        tok_s = toks / ms * 1000
        return {"ms": ms, "tok_s": tok_s, "mem_gb": mem_gb, "ok": True}
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "error": "OOM"}
    finally:
        del model, opt
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/Qwen3-4B")
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    print(f"{'='*70}")
    print(f"FSDP Training Benchmark — {args.model}")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM: {total/1024**3:.0f}GB total, {free/1024**3:.0f}GB free")
    print(f"Seq length: {args.seq_len}")
    print()

    configs = [
        ("Baseline", dict(mbs=1, gc=False, fused=False, reshard=True, bf16_reduce=False)),
        ("+ Fused AdamW", dict(mbs=1, gc=False, fused=True, reshard=True, bf16_reduce=False)),
        ("+ reshard=False", dict(mbs=1, gc=False, fused=True, reshard=False, bf16_reduce=False)),
        ("+ BF16 reduce", dict(mbs=1, gc=False, fused=True, reshard=False, bf16_reduce=True)),
        ("+ MBS=2", dict(mbs=2, gc=False, fused=True, reshard=False, bf16_reduce=True)),
        ("+ MBS=4", dict(mbs=4, gc=False, fused=True, reshard=False, bf16_reduce=True)),
        ("+ GC + MBS=4", dict(mbs=4, gc=True, fused=True, reshard=False, bf16_reduce=True)),
    ]

    print(f"{'Config':25s} {'ms/step':>8s} {'tok/s':>8s} {'Memory':>8s} {'vs Base':>8s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    baseline_tok_s = None
    for name, cfg in configs:
        result = benchmark_config(args.model, seq_len=args.seq_len, **cfg)
        if result["ok"]:
            if baseline_tok_s is None:
                baseline_tok_s = result["tok_s"]
            speedup = result["tok_s"] / baseline_tok_s
            print(f"{name:25s} {result['ms']:8.0f} {result['tok_s']:8.0f} {result['mem_gb']:7.0f}GB {speedup:7.2f}x")
        else:
            print(f"{name:25s} {'OOM':>8s}")

    print()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
