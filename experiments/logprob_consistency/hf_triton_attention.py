"""
HF Triton Attention - 对接 SGLang prefill_attention，
用于 Miles true on-policy (Triton backend) 的 FSDP 侧 attention 对齐。
"""
from typing import Optional

import torch

# 延迟导入，避免未用 triton 时加载 sglang
_context_attention_fwd = None
_extend_attention_fwd_unified = None


def _get_context_attention_fwd():
    global _context_attention_fwd
    if _context_attention_fwd is None:
        from sglang.srt.layers.attention.triton_ops.prefill_attention import (
            context_attention_fwd,
        )
        _context_attention_fwd = context_attention_fwd
    return _context_attention_fwd


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


def unified_replay_attention_forward(
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
    Unified-only replay attention backend for teacher-forcing experiments.

    It uses SGLang's extend_attention_fwd_unified in full-extend / zero-prefix mode.
    This keeps HF model forward (embedding/RoPE/MLP/lm_head) intact while replacing
    only the attention kernel path.
    """
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        raise ValueError(
            "unified_replay 不支持 output_attentions 或 head_mask，请改用 eager"
        )

    extend_attention_fwd_unified = _get_extend_attention_fwd_unified()

    B, H, S, D = query.shape
    kv_heads = key.shape[1]
    if H % kv_heads != 0:
        raise ValueError(
            f"Invalid GQA config: q_heads({H}) % kv_heads({kv_heads}) != 0"
        )
    if B != 1:
        raise NotImplementedError(
            "unified_replay 当前仅支持 batch=1 的最小 teacher-forcing 场景"
        )

    # Flatten to kernel layout [num_tokens, heads, head_dim].
    q = query.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
    k_buffer = key.permute(0, 2, 1, 3).reshape(B * S, kv_heads, D).contiguous()
    v_buffer = value.permute(0, 2, 1, 3).reshape(B * S, kv_heads, D).contiguous()
    o = torch.empty_like(q)

    # Full-extend / zero-prefix unified metadata.
    device = q.device
    qo_indptr = torch.tensor([0, S], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, S], dtype=torch.int32, device=device)
    kv_indices = torch.arange(S, dtype=torch.int64, device=device)
    prefix_lens = torch.tensor([0], dtype=torch.int32, device=device)
    sm_scale = scaling if scaling is not None else (D ** -0.5)

    extend_attention_fwd_unified(
        q=q,
        o=o,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        prefix_lens=prefix_lens,
        max_len_extend=S,
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

    attn_output = o.reshape(B, S, H, D)
    return attn_output, None
