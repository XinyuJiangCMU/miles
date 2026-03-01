#!/usr/bin/env python3
"""
Unified-only attention parity experiment.

Compare two unified-family representations on the same Q/K/V:

Route A (SGLang-style decode step as unified q_len=1):
  - For each position i, run extend_attention_fwd_unified with one query token.
  - Visible KV is [0..i] via unified metadata.

Route B (FSDP-side full-extend / zero-prefix):
  - Run extend_attention_fwd_unified once on the full sequence.
  - prefix_lens = [0], all Q/KV belong to extend.
  - Compare per-position outputs with Route A.

Scope:
  - batch=1, MHA only
  - no sliding window / sinks / custom mask
  - bf16 default
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch


def _ensure_sglang_import() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "..", "sglang", "python"),
        os.path.join("/app", "sglang", "python"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return


_ensure_sglang_import()

from sglang.srt.layers.attention.triton_ops.extend_attention import (  # noqa: E402
    extend_attention_fwd_unified,
)


def build_test_inputs(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Build random Q/K/V for batch=1 and return flattened [S,H,D] tensors.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q = torch.randn((1, seq_len, num_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((1, seq_len, num_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((1, seq_len, num_heads, head_dim), dtype=dtype, device=device)
    return q[0].contiguous(), k[0].contiguous(), v[0].contiguous(), head_dim ** -0.5


def run_route_a_sglang_q1(
    q_all: torch.Tensor,  # [S,H,D]
    k_buffer: torch.Tensor,  # [S,H,D]
    v_buffer: torch.Tensor,  # [S,H,D]
    sm_scale: float,
) -> torch.Tensor:
    """
    Route A: per-position unified q_len=1 (decode-step style).

    For each i:
      qo_indptr   = [0, 1]
      kv_indptr   = [0, i+1]
      kv_indices  = [0..i]
      prefix_lens = [i]
    """
    device = q_all.device
    seq_len, _h, _d = q_all.shape
    outs = []
    for i in range(seq_len):
        q_i = q_all[i : i + 1].contiguous()  # [1,H,D]
        o_i = torch.empty_like(q_i).contiguous()

        qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, i + 1], dtype=torch.int32, device=device)
        kv_indices = torch.arange(i + 1, dtype=torch.int64, device=device)
        prefix_lens = torch.tensor([i], dtype=torch.int32, device=device)

        extend_attention_fwd_unified(
            q=q_i,
            o=o_i,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            prefix_lens=prefix_lens,
            max_len_extend=1,
            custom_mask=None,
            mask_indptr=None,
            sm_scale=sm_scale,
            logit_cap=0.0,
            is_causal=True,
            sliding_window_size=-1,
            sinks=None,
            window_start_pos=None,
            xai_temperature_len=-1,
        )
        outs.append(o_i)

    return torch.cat(outs, dim=0)  # [S,H,D]


def run_route_b_fsdp_full_extend_zero_prefix(
    q_all: torch.Tensor,  # [S,H,D]
    k_buffer: torch.Tensor,  # [S,H,D]
    v_buffer: torch.Tensor,  # [S,H,D]
    sm_scale: float,
) -> torch.Tensor:
    """
    Route B: full-extend / zero-prefix unified representation.

      qo_indptr   = [0, S]
      kv_indptr   = [0, S]
      kv_indices  = [0..S-1]
      prefix_lens = [0]
    """
    device = q_all.device
    seq_len, _h, _d = q_all.shape
    o_full = torch.empty_like(q_all).contiguous()

    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, dtype=torch.int64, device=device)
    prefix_lens = torch.tensor([0], dtype=torch.int32, device=device)

    extend_attention_fwd_unified(
        q=q_all.contiguous(),
        o=o_full,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        prefix_lens=prefix_lens,
        max_len_extend=seq_len,
        custom_mask=None,
        mask_indptr=None,
        sm_scale=sm_scale,
        logit_cap=0.0,
        is_causal=True,
        sliding_window_size=-1,
        sinks=None,
        window_start_pos=None,
        xai_temperature_len=-1,
    )
    return o_full


def evaluate_parity(
    o_a: torch.Tensor,  # [S,H,D]
    o_b: torch.Tensor,  # [S,H,D]
    rtol: float,
    atol: float,
    show_failures: int,
) -> Dict[str, object]:
    seq_len = o_a.shape[0]
    diffs = (o_a.float() - o_b.float()).abs()
    per_pos_max = diffs.amax(dim=(1, 2))
    global_max = diffs.max().item()
    global_mean = diffs.mean().item()

    failed: List[Dict[str, object]] = []
    for i in range(seq_len):
        a_i = o_a[i].float()
        b_i = o_b[i].float()
        if not torch.allclose(a_i, b_i, rtol=rtol, atol=atol):
            d = (a_i - b_i).abs()
            flat = d.argmax().item()
            h = flat // d.shape[1]
            dd = flat % d.shape[1]
            failed.append(
                {
                    "pos": i,
                    "max_abs_diff": d.max().item(),
                    "mean_abs_diff": d.mean().item(),
                    "head_idx": int(h),
                    "dim_idx": int(dd),
                    "a_elem": a_i[h, dd].item(),
                    "b_elem": b_i[h, dd].item(),
                }
            )

    if failed:
        print(f"first {min(show_failures, len(failed))} failure(s):")
        for item in failed[:show_failures]:
            print(
                f"  - pos={item['pos']:4d}, max_abs_diff={item['max_abs_diff']:.6e}, "
                f"mean_abs_diff={item['mean_abs_diff']:.6e}, "
                f"head={item['head_idx']}, dim={item['dim_idx']}"
            )
            print(
                f"    routeA={item['a_elem']:.6e}, routeB={item['b_elem']:.6e}"
            )

    return {
        "seq_len": seq_len,
        "failed_positions": len(failed),
        "global_max_abs_diff": global_max,
        "global_mean_abs_diff": global_mean,
        "per_pos_max_abs_diff_mean": per_pos_max.mean().item(),
    }


def parse_sweep(v: str) -> List[int]:
    vals = []
    for part in v.split(","):
        p = part.strip()
        if p:
            vals.append(int(p))
    if not vals:
        raise ValueError("empty sweep list")
    return vals


def run_case(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    rtol: float,
    atol: float,
    show_failures: int,
) -> Dict[str, object]:
    q_all, k_buffer, v_buffer, sm_scale = build_test_inputs(
        seq_len, num_heads, head_dim, dtype, device, seed
    )
    o_a = run_route_a_sglang_q1(q_all, k_buffer, v_buffer, sm_scale)
    o_b = run_route_b_fsdp_full_extend_zero_prefix(q_all, k_buffer, v_buffer, sm_scale)

    has_nan = torch.isnan(o_a).any().item() or torch.isnan(o_b).any().item()
    has_inf = torch.isinf(o_a).any().item() or torch.isinf(o_b).any().item()
    metrics = evaluate_parity(o_a, o_b, rtol, atol, show_failures)
    metrics["has_nan"] = bool(has_nan)
    metrics["has_inf"] = bool(has_inf)
    metrics["shape_a"] = tuple(o_a.shape)
    metrics["shape_b"] = tuple(o_b.shape)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parity: unified q_len=1 (SGLang-style) vs unified full-extend (FSDP-style)."
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--sweep-seq-lens", type=str, default="")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--show-failures", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA/HIP GPU available. On ROCm, device string should still be 'cuda'."
        )

    dtype = getattr(torch, args.dtype)
    if args.sweep_seq_lens.strip():
        seq_lens = parse_sweep(args.sweep_seq_lens)
    else:
        seq_lens = [args.seq_len]

    print("=" * 80)
    print("Unified-only Attention Parity")
    print("=" * 80)
    print(
        f"config: B=1, H={args.num_heads}, D={args.head_dim}, dtype={args.dtype}, "
        f"device={args.device}, seed={args.seed}"
    )
    print(
        f"runtime: torch={torch.__version__}, cuda={torch.version.cuda}, hip={torch.version.hip}"
    )
    print(
        "routeA: unified q_len=1 per position (prefix_lens=[i], kv=[0..i])\n"
        "routeB: unified full-extend zero-prefix (prefix_lens=[0], kv=[0..S-1])"
    )

    all_results: List[Dict[str, object]] = []
    for s in seq_lens:
        print("-" * 80)
        print(f"running seq_len={s} ...")
        r = run_case(
            seq_len=s,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=args.device,
            seed=args.seed,
            rtol=args.rtol,
            atol=args.atol,
            show_failures=args.show_failures,
        )
        all_results.append(r)
        print(
            f"shapeA={r['shape_a']}, shapeB={r['shape_b']}, nan={r['has_nan']}, inf={r['has_inf']}"
        )
        print(
            f"failed_positions={r['failed_positions']}/{s}, "
            f"global_max_abs_diff={r['global_max_abs_diff']:.6e}, "
            f"global_mean_abs_diff={r['global_mean_abs_diff']:.6e}"
        )

    print("=" * 80)
    print("summary")
    print("=" * 80)
    print("seq_len | failed_positions | global_max_abs_diff | global_mean_abs_diff")
    for r in all_results:
        print(
            f"{r['seq_len']:7d} | {r['failed_positions']:16d} | "
            f"{r['global_max_abs_diff']:.6e} | {r['global_mean_abs_diff']:.6e}"
        )


if __name__ == "__main__":
    main()
