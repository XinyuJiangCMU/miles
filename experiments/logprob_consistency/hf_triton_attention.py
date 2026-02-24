"""
HF Triton Attention - 对接 SGLang prefill_attention，
用于 Miles true on-policy (Triton backend) 的 FSDP 侧 attention 对齐。
"""
from typing import Optional

import torch

# 延迟导入，避免未用 triton 时加载 sglang
_context_attention_fwd = None


def _get_context_attention_fwd():
    global _context_attention_fwd
    if _context_attention_fwd is None:
        from sglang.srt.layers.attention.triton_ops.prefill_attention import (
            context_attention_fwd,
        )
        _context_attention_fwd = context_attention_fwd
    return _context_attention_fwd


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
    HF AttentionInterface 签名。
    query/key/value: (B, num_heads, S, head_dim)，GQA 时 key/value 的 heads 可能更少。
    返回 (attn_output, None)，attn_output 形状 (B, S, num_heads, head_dim)。
    """
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        raise ValueError(
            "triton attention 不支持 output_attentions 或 head_mask，请改用 eager"
        )

    context_attention_fwd = _get_context_attention_fwd()

    B, H, S, D = query.shape
    kv_heads = key.shape[1]
    kv_group_num = H // kv_heads

    q = query.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(B * S, kv_heads, D).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(B * S, kv_heads, D).contiguous()
    o = torch.empty_like(q)

    b_start_loc = torch.arange(0, B * S, S, device=q.device, dtype=torch.int32)
    b_seq_len = torch.full((B,), S, device=q.device, dtype=torch.int32)

    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len=S, is_causal=True)

    attn_output = o.reshape(B, S, H, D)
    return attn_output, None
