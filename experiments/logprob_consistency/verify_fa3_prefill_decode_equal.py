#!/usr/bin/env python3
"""
=============================================================================
验证 FA3 prefill 与 decode 路径是否比特级相等
=============================================================================

【测试目的】
  Miles True On-Policy 依赖 FA3 的 "bitwise equal between prefill and decode"。
  本脚本验证：对同一组 (Q_i, K_{1:i}, V_{1:i}) 的 causal attention，
  无论用 prefill 路径还是 decode 路径，结果是否完全一致。

【测试方法】
  1. 生成随机 Q, K, V，形状 [batch=1, seq_len, num_heads, head_dim]
  2. Prefill 路径：flash_attn_func(q, k, v, causal=True)
     - 一次性计算整段序列的 causal attention
     - 输出 out_prefill[i] = attention(Q[i], K[0:i+1], V[0:i+1])
  3. Decode 路径：对每个位置 i，调用 flash_attn_with_kvcache(q_i, k_cache, v_cache)
     - q_i = Q[i:i+1]，仅当前位置的 query
     - k_cache = K[0:i+1], v_cache = V[0:i+1]
     - 输出 out_decode = attention(Q[i], K[0:i+1], V[0:i+1])
  4. 逐位置比较：torch.equal(out_prefill[i], out_decode)

【用法】
  python verify_fa3_prefill_decode_equal.py
  python verify_fa3_prefill_decode_equal.py --output results/fa3_verify.txt

【依赖】
  pip install flash-attn
"""

import argparse
import os
import sys
from datetime import datetime

import torch


def verify_fa3(seq_len=32, num_heads=8, head_dim=64, seed=42):
    """使用 flash-attn 验证 prefill 与 decode 比特级相等。返回 (all_match, max_diff, failed, details) 或 None。"""
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    # [1] 构造输入：与真实 LLM attention 相同的形状
    #     batch=1 简化实验；seq_len=32；num_heads=8, head_dim=64 常见配置
    q = torch.randn(1, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, seq_len, num_heads, head_dim, dtype=dtype, device=device)

    sm_scale = head_dim**-0.5  # 1/sqrt(d)，attention 缩放因子

    try:
        from flash_attn import flash_attn_func, flash_attn_with_kvcache
    except ImportError:
        return None

    # [2] Prefill 路径：模拟「处理整段 prompt」的一次性前向
    #     flash_attn_func 内部对每个位置 i 计算：out[i] = softmax(Q[i]@K[0:i+1].T/sqrt(d)) @ V[0:i+1]
    #     等价于 SGLang prefill 或 FSDP 整段 teacher-forcing 的 attention
    out_prefill = flash_attn_func(q, k, v, causal=True, softmax_scale=sm_scale)
    # out_prefill: [1, seq_len, num_heads, head_dim]

    # [3] Decode 路径：模拟「每步生成 1 个 token」的 incremental decode
    #     对位置 i：只算 Q[i] 对 K[0:i+1], V[0:i+1] 的 attention，与 prefill 中 out[i] 数学上相同
    all_match = True
    max_diff = 0.0
    failed = []
    out_decode_list = []  # 存每个位置的 decode 输出，便于详细对比

    for i in range(seq_len):
        # 位置 i 的 query：仅 1 个 token
        q_i = q[:, i : i + 1, :, :].contiguous()  # [1, 1, H, D]
        # KV cache：到位置 i 为止的 K, V（causal：位置 i 只能看到 0..i）
        k_cache = k[:, : i + 1, :, :].contiguous()  # [1, i+1, H, D]
        v_cache = v[:, : i + 1, :, :].contiguous()
        cache_seqlens = torch.tensor([i + 1], dtype=torch.int32, device=device)

        out_decode = flash_attn_with_kvcache(
            q_i,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=sm_scale,
            causal=True,
        )
        out_decode = out_decode.squeeze(1)  # [1, H, D]
        out_decode_list.append(out_decode.clone())

        out_prefill_i = out_prefill[:, i, :, :]  # [1, H, D]

        if not torch.equal(out_prefill_i, out_decode):
            diff = (out_prefill_i.float() - out_decode.float()).abs().max().item()
            max_diff = max(max_diff, diff)
            all_match = False
            failed.append((i, diff))

    details = {
        "out_prefill": out_prefill,
        "out_decode_list": out_decode_list,
        "q": q,
        "k": k,
        "v": v,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "sm_scale": sm_scale,
    }
    return all_match, max_diff, failed, details


def _format_tensor_sample(t: torch.Tensor, head=0, dim_slice=4) -> str:
    """格式化张量的一小部分，便于阅读。"""
    t = t.detach().cpu().float()
    if t.dim() == 4:  # [B,S,H,D]
        s = t[0, :, head, :dim_slice]
    elif t.dim() == 3:  # [B,H,D]
        s = t[0, head, :dim_slice]
    else:
        s = t.flatten()[:dim_slice]
    return " ".join(f"{x:.6f}" for x in s.tolist())


def build_report(all_match, max_diff, failed, backend, details, seq_len=32) -> str:
    """生成详细报告字符串。"""
    lines = []
    lines.append("=" * 70)
    lines.append("FA3 Prefill vs Decode 比特级相等验证报告")
    lines.append("=" * 70)
    lines.append(f"时间: {datetime.now().isoformat()}")
    lines.append(f"Backend: {backend}")
    lines.append(f"序列长度: {details['seq_len']}, heads: {details['num_heads']}, head_dim: {details['head_dim']}")
    lines.append("")

    # 为什么重要
    lines.append("-" * 70)
    lines.append("【为什么重要】")
    lines.append("-" * 70)
    lines.append("Miles True On-Policy 要求：SGLang 推理与 FSDP 训练对同一 trajectory 的 logprob")
    lines.append("比特级一致。SGLang 推理时：prefill 处理 prompt，decode 逐 token 生成；FSDP")
    lines.append("训练时：整段序列一次前向（等价 prefill 风格）。若 prefill 与 decode 数值不同，")
    lines.append("则 FSDP 无法与 SGLang 对齐。FA3 的设计保证两者比特级相等，本实验验证之。")
    lines.append("")

    # 张量形状
    lines.append("-" * 70)
    lines.append("【张量形状】")
    lines.append("-" * 70)
    q, k, v = details["q"], details["k"], details["v"]
    op = details["out_prefill"]
    lines.append(f"  Q, K, V: {tuple(q.shape)} = (batch, seq_len, num_heads, head_dim)")
    lines.append(f"  out_prefill: {tuple(op.shape)}")
    lines.append("")

    # 测试方法说明
    lines.append("-" * 70)
    lines.append("【测试方法】")
    lines.append("-" * 70)
    lines.append("1. Prefill 路径: flash_attn_func(q, k, v, causal=True)")
    lines.append("   - 一次性计算整段 [0..seq_len-1] 的 causal attention")
    lines.append("   - 输出 out_prefill[i] = softmax(Q[i]@K[0:i+1].T)*V[0:i+1]")
    lines.append("")
    lines.append("2. Decode 路径: 对每个位置 i 调用 flash_attn_with_kvcache")
    lines.append("   - q_i = Q[i:i+1], k_cache = K[0:i+1], v_cache = V[0:i+1]")
    lines.append("   - 输出 out_decode = 同样的 causal attention")
    lines.append("")
    lines.append("3. 比较: torch.equal(out_prefill[i], out_decode) 逐位置")
    lines.append("")

    # 结果摘要
    lines.append("-" * 70)
    lines.append("【结果摘要】")
    lines.append("-" * 70)
    if all_match:
        lines.append("[PASS] 所有位置比特级一致")
    else:
        lines.append(f"[FAIL] max_diff={max_diff:.2e}, 不匹配位置数={len(failed)}")
        for pos, d in failed[:20]:
            lines.append(f"  位置 {pos}: diff={d:.2e}")
    lines.append("")

    # 采样位置详细数值（位置 0, seq_len//4, seq_len//2, seq_len-1）
    out_prefill = details["out_prefill"]
    out_decode_list = details["out_decode_list"]
    sample_positions = [0, max(1, seq_len // 4), seq_len // 2, seq_len - 1]
    sample_positions = sorted(set(p for p in sample_positions if p < seq_len))

    lines.append("-" * 70)
    lines.append("【采样位置数值对比】")
    lines.append("  (格式: prefill 与 decode 的 output[0, head=0, :4] 前 4 维)")
    lines.append("-" * 70)

    for pos in sample_positions:
        pf = out_prefill[:, pos, :, :]
        dc = out_decode_list[pos]
        eq = torch.equal(pf, dc)
        lines.append(f"位置 {pos}:")
        lines.append(f"  Prefill  output[0,:4] = {_format_tensor_sample(pf)}")
        lines.append(f"  Decode  output[0,:4] = {_format_tensor_sample(dc)}")
        lines.append(f"  比特级一致: {eq}")
        if not eq:
            diff = (pf.float() - dc.float()).abs().max().item()
            lines.append(f"  max |diff| = {diff:.2e}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="验证 FA3 prefill vs decode 比特级相等")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="结果保存到指定 txt 文件",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="测试序列长度（默认 32）",
    )
    args = parser.parse_args()

    seq_len = args.seq_len

    print("=" * 70)
    print("验证 FA3 prefill vs decode 比特级相等")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("需要 CUDA")
        return 1

    result = verify_fa3(seq_len=seq_len)
    if result is None:
        print("未找到 flash_attn，请安装: pip install flash-attn")
        return 1

    all_match, max_diff, failed, details = result
    report = build_report(all_match, max_diff, failed, "flash_attn", details, seq_len)

    print(report)

    if args.output:
        d = os.path.dirname(args.output)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n结果已保存到: {args.output}")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
