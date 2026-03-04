"""
Miles alignment workspace local copy of the HF Triton attention shim.

This is used by the FSDP-side alignment experiments to call SGLang's
deterministic Triton unified extend kernel from the HF AttentionInterface path.

Semantics here are pure prefill / teacher-forcing / no-prefix-cache:
- query/key/value arrive as [B, H, S, D]
- tensors are reshaped to varlen [B*S, H, D]
- qo_indptr / kv_indptr / kv_indices are constructed locally
- prefix_lens are all zero
- is_causal=True

It uses the same `extend_attention_fwd_unified` kernel as SGLang server runtime,
but not the exact same metadata construction path as the server scheduler.
"""

from typing import Optional

import torch

_extend_attention_fwd_unified = None


def _maybe_get_dumper():
    try:
        return __import__("sglang.srt.debug_utils.dumper", fromlist=["dumper"]).dumper
    except Exception:
        return None


_DUMPER = _maybe_get_dumper()


def _maybe_dump(name: str, value) -> None:
    if _DUMPER is None:
        return
    try:
        _DUMPER.dump(name, value)
    except Exception:
        pass


def _get_extend_attention_fwd_unified():
    global _extend_attention_fwd_unified
    if _extend_attention_fwd_unified is None:
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd_unified,
        )

        _extend_attention_fwd_unified = extend_attention_fwd_unified
    return _extend_attention_fwd_unified


def triton_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    HF AttentionInterface signature.

    query/key/value: (B, num_heads, S, head_dim), GQA: key/value may have fewer heads.
    Returns (attn_output, None), attn_output shape (B, S, num_heads, head_dim).
    """

    extend_attention_fwd_unified = _get_extend_attention_fwd_unified()

    batch, num_q_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    device = query.device

    q = query.permute(0, 2, 1, 3).reshape(batch * seq_len, num_q_heads, head_dim).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(batch * seq_len, num_kv_heads, head_dim).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(batch * seq_len, num_kv_heads, head_dim).contiguous()
    out = torch.empty_like(q)

    total_tokens = batch * seq_len
    qo_indptr = torch.arange(0, batch + 1, device=device, dtype=torch.int32) * seq_len
    kv_indptr = qo_indptr.clone()
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)
    prefix_lens = torch.zeros(batch, device=device, dtype=torch.int32)
    if getattr(module, "layer_idx", None) == 0:
        _maybe_dump("hf_q_kernel_in", q)
        _maybe_dump("hf_qo_indptr", qo_indptr)
        _maybe_dump("hf_kv_indptr", kv_indptr)
        _maybe_dump("hf_kv_indices", kv_indices)
        _maybe_dump("hf_prefix_lens", prefix_lens)
        _maybe_dump(
            "hf_max_len_extend",
            torch.tensor([seq_len], device=device, dtype=torch.int32),
        )
        _maybe_dump(
            "hf_sm_scale",
            torch.tensor(
                [scaling if scaling is not None else 1.0 / (head_dim**0.5)],
                device=device,
                dtype=torch.float32,
            ),
        )
        _maybe_dump("hf_is_causal", torch.tensor([1], device=device, dtype=torch.int32))

    extend_attention_fwd_unified(
        q,
        out,
        k,
        v,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        max_len_extend=seq_len,
        is_causal=True,
    )

    attn_output = out.reshape(batch, seq_len, num_q_heads, head_dim)
    return attn_output, None
