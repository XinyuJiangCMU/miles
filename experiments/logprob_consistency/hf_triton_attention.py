"""
HF Triton Attention - 对接 SGLang Triton attention，
用于 Miles true on-policy (Triton backend) 的 FSDP 侧 attention 对齐。

实现 extend + decode 拆分：
  - extend 段（0..extend_len-1）：context_attention_fwd（prefill）
  - decode 段（extend_len..S-1）：decode_attention_fwd（decode）

extend_len 获取优先级：
  1. set_extend_len_context() 设置的 context 值（推荐，上层在 model forward 前调用）
  2. kwargs["extend_len"]
  3. 默认 14
"""
from contextvars import ContextVar
from typing import Optional

import torch

# extend_len 的 context，供上层在 model forward 前注入
_extend_len_ctx: ContextVar[Optional[int]] = ContextVar("triton_extend_len", default=None)

# 延迟导入，避免未用 triton 时加载 sglang
_context_attention_fwd = None
_decode_attention_fwd = None


def set_extend_len_context(extend_len: int):
    """
    在 model forward 前调用，注入 extend_len（prompt 长度）。
    返回 token，forward 后需调用 reset_extend_len_context(token) 恢复。
    """
    return _extend_len_ctx.set(extend_len)


def reset_extend_len_context(token) -> None:
    """forward 后调用，恢复 extend_len context。"""
    try:
        _extend_len_ctx.reset(token)
    except (LookupError, ValueError):
        pass


def _get_context_attention_fwd():
    global _context_attention_fwd
    if _context_attention_fwd is None:
        from sglang.srt.layers.attention.triton_ops.prefill_attention import (
            context_attention_fwd,
        )
        _context_attention_fwd = context_attention_fwd
    return _context_attention_fwd


def _get_decode_attention_fwd():
    global _decode_attention_fwd
    if _decode_attention_fwd is None:
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        _decode_attention_fwd = decode_attention_fwd
    return _decode_attention_fwd


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

    extend_len: kwargs 传入，默认 14。extend 段用 context_attention，decode 段用 decode_attention。
    """
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        raise ValueError(
            "triton attention 不支持 output_attentions 或 head_mask，请改用 eager"
        )

    # 优先级：context > kwargs > 默认 14
    extend_len = _extend_len_ctx.get()
    if extend_len is None:
        extend_len = kwargs.get("extend_len", 14)

    B, H, S, D = query.shape
    kv_heads = key.shape[1]
    D_v = value.shape[-1]  # GQA 时 value head_dim 可能与 D 不同

    q = query.permute(0, 2, 1, 3).contiguous()  # (B, S, H, D)
    k = key.permute(0, 2, 1, 3).contiguous()     # (B, S, kv_heads, D)
    v = value.permute(0, 2, 1, 3).contiguous()  # (B, S, kv_heads, D_v)

    o = torch.empty(B, S, H, D_v, dtype=q.dtype, device=q.device)

    if S <= extend_len:
        # 纯 prefill：全用 context_attention
        context_attention_fwd = _get_context_attention_fwd()
        q_flat = q.reshape(B * S, H, D)
        k_flat = k.reshape(B * S, kv_heads, D)
        v_flat = v.reshape(B * S, kv_heads, D_v)
        o_flat = torch.empty_like(q_flat)
        b_start_loc = torch.arange(0, B * S, S, device=q.device, dtype=torch.int32)
        b_seq_len = torch.full((B,), S, device=q.device, dtype=torch.int32)
        context_attention_fwd(
            q_flat, k_flat, v_flat, o_flat,
            b_start_loc, b_seq_len, max_input_len=S, is_causal=True
        )
        o = o_flat.reshape(B, S, H, D_v)
    else:
        # extend 段 + decode 段
        context_attention_fwd = _get_context_attention_fwd()
        decode_attention_fwd = _get_decode_attention_fwd()

        # 1) Extend: 0..extend_len-1
        q_ext = q[:, :extend_len].reshape(B * extend_len, H, D)
        k_ext = k[:, :extend_len].reshape(B * extend_len, kv_heads, D)
        v_ext = v[:, :extend_len].reshape(B * extend_len, kv_heads, D_v)
        o_ext = torch.empty_like(q_ext)
        b_start_loc = torch.arange(0, B * extend_len, extend_len, device=q.device, dtype=torch.int32)
        b_seq_len = torch.full((B,), extend_len, device=q.device, dtype=torch.int32)
        context_attention_fwd(
            q_ext, k_ext, v_ext, o_ext,
            b_start_loc, b_seq_len, max_input_len=extend_len, is_causal=True
        )
        o[:, :extend_len] = o_ext.reshape(B, extend_len, H, D_v)

        # 2) Decode: extend_len..S-1，逐 position 调用
        # decode kernel 需要的 buffer/indices 格式（见 Step 1/3）：
        #   k_buffer, v_buffer: [total_tokens, kv_heads, head_dim]，连续存储
        #   kv_indptr: [batch+1]，cumsum，kv_indptr[b+1]-kv_indptr[b]=batch b 的 seq_len
        #   kv_indices: [total_tokens]，kv_indices[kv_indptr[b]:kv_indptr[b+1]] 为 batch b 的 buffer 索引
        # HF key[:,:,:p+1] -> k_buffer: 按 batch 拼接，buffer[i]=batch(i//seq_len) 的 position(i%seq_len)
        sm_scale = scaling if scaling is not None else (1.0 / (D ** 0.5))
        max_kv_splits = 8
        num_kv_splits = torch.ones(B, dtype=torch.int32, device=q.device)

        for p in range(extend_len, S):
            seq_len = p + 1  # position p 需 attend 到 0..p 共 seq_len 个 token
            q_p = q[:, p, :, :]  # (B, H, D)
            # key[:,:,:p+1] -> k_buffer: (B, seq_len, kv_heads, D) -> (B*seq_len, kv_heads, D)
            k_p = k[:, :seq_len, :, :].reshape(B * seq_len, kv_heads, D)
            v_p = v[:, :seq_len, :, :].reshape(B * seq_len, kv_heads, D_v)
            # contiguous 存储：batch 0 占 slot 0..seq_len-1，batch 1 占 seq_len..2*seq_len-1
            kv_indptr = torch.arange(0, B + 1, device=q.device, dtype=torch.int32) * seq_len
            kv_indices = torch.arange(B * seq_len, device=q.device, dtype=torch.int32)

            attn_logits = torch.empty(
                B, H, max_kv_splits, D_v,
                dtype=torch.float32, device=q.device
            )
            attn_lse = torch.empty(
                B, H, max_kv_splits,
                dtype=torch.float32, device=q.device
            )

            decode_attention_fwd(
                q_p, k_p, v_p,
                o[:, p],
                kv_indptr, kv_indices,
                attn_logits, attn_lse,
                num_kv_splits, max_kv_splits,
                sm_scale,
            )

    attn_output = o
    if D_v != D:
        attn_output = attn_output.to(query.dtype)
    return attn_output, None
