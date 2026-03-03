"""Qwen3 semantic adapter for the SGLang Triton HF patch path."""

import torch

from ..kernels.triton_extend_attn_unified import run_unified_extend


def _resolve_rotary():
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        return apply_rotary_pos_emb
    except Exception:
        try:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            return apply_rotary_pos_emb
        except Exception:
            return None


APPLY_ROTARY_POS_EMB = _resolve_rotary()


def qwen3_triton_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Qwen3-like attention semantic path with unified extend kernel."""
    if past_key_values is not None:
        return self._sglang_triton_original_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    batch, seq_len, _ = hidden_states.shape
    head_dim = getattr(self, "head_dim", None)
    if head_dim is None and hasattr(self, "config"):
        head_dim = getattr(
            self.config, "head_dim", None
        ) or (self.config.hidden_size // self.config.num_attention_heads)
    if head_dim is None:
        head_dim = self.q_proj.out_features // getattr(
            self.config, "num_attention_heads", 32
        )

    num_heads = getattr(self, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(
            self.config, "num_attention_heads", self.q_proj.out_features // head_dim
        )
    num_kv_heads = getattr(self, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(self, "num_kv_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.k_proj.out_features // head_dim
        )

    # SGLang Qwen3 semantic order: qkv -> qk_norm -> rope -> attention core -> o_proj.
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = (
        self.v_proj(hidden_states)
        .view(batch, seq_len, num_kv_heads, head_dim)
        .transpose(1, 2)
    )

    if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
        q_by_head = query_states.reshape(-1, head_dim)
        k_by_head = key_states.reshape(-1, head_dim)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = self.k_norm(k_by_head)
        query_states = q_by_head.view_as(query_states)
        key_states = k_by_head.view_as(key_states)

    query_states = query_states.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    if position_embeddings is not None and APPLY_ROTARY_POS_EMB is not None:
        cos, sin = position_embeddings
        query_states, key_states = APPLY_ROTARY_POS_EMB(
            query_states, key_states, cos, sin
        )

    total_tokens = batch * seq_len
    q_varlen = query_states.permute(0, 2, 1, 3).contiguous().view(
        total_tokens, num_heads, head_dim
    )
    k_buffer = key_states.permute(0, 2, 1, 3).contiguous().view(
        total_tokens, num_kv_heads, head_dim
    )
    v_buffer = value_states.permute(0, 2, 1, 3).contiguous().view(
        total_tokens, num_kv_heads, head_dim
    )

    o = run_unified_extend(
        q_varlen=q_varlen,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        batch=batch,
        seq_len=seq_len,
    )

    attn_output = o.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None

