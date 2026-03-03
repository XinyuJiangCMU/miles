"""Kernel wrapper for SGLang unified extend attention."""

import torch


def run_unified_extend(
    q_varlen: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    batch: int,
    seq_len: int,
) -> torch.Tensor:
    """Run extend_attention_fwd_unified in teacher-forcing prefill mode."""
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd_unified,
    )

    device = q_varlen.device
    o = torch.empty_like(q_varlen)
    qo_indptr = torch.arange(0, batch + 1, device=device, dtype=torch.int32) * seq_len
    kv_indptr = qo_indptr.clone()
    kv_indices = torch.arange(batch * seq_len, device=device, dtype=torch.int64)
    prefix_lens = torch.zeros(batch, device=device, dtype=torch.int32)

    extend_attention_fwd_unified(
        q_varlen,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        max_len_extend=seq_len,
        is_causal=True,
    )
    return o

