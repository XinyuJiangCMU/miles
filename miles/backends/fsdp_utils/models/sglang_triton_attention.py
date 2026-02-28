"""
Patch HuggingFace attention layers to use SGLang Triton unified extend attention
for true on-policy training when attn_implementation == "sglang_triton".

Uses extend_attention_fwd_unified (the same kernel as SGLang's deterministic
inference extend path) with prefix_len=0 and sequential kv_indices, so the
training-side attention is numerically identical to the inference-side extend.

Use: after loading the model, call apply_sglang_triton_attention_patch(model).
Supports Qwen2/Qwen3-style attention (forward with position_embeddings tuple).
"""

import types
import torch


def _make_triton_attention_forward(original_forward):
    """Build a forward that uses extend_attention_fwd_unified for teacher-forcing prefill."""

    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd_unified,
    )

    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except Exception:
        try:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        except Exception:
            apply_rotary_pos_emb = None

    def triton_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_values is not None:
            return original_forward(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        head_dim = getattr(self, "head_dim", None)
        if head_dim is None and hasattr(self, "config"):
            head_dim = getattr(
                self.config, "head_dim", None
            ) or (self.config.hidden_size // self.config.num_attention_heads)
        if head_dim is None:
            head_dim = self.q_proj.out_features // getattr(
                self.config, "num_attention_heads", 32
            )
        num_heads = getattr(
            self.config, "num_attention_heads", self.q_proj.out_features // head_dim
        )
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.k_proj.out_features // head_dim
        )

        hidden_shape = (batch, seq_len, -1, head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states)
            .view(batch, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(batch, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )

        if position_embeddings is not None and apply_rotary_pos_emb is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        total_tokens = batch * seq_len
        # Shape: [total_tokens, num_heads, head_dim]
        q_varlen = (
            query_states.permute(0, 2, 1, 3).contiguous().view(total_tokens, num_heads, head_dim)
        )
        # Shape: [total_tokens, num_kv_heads, head_dim] — used directly as k_buffer/v_buffer
        k_buffer = (
            key_states.permute(0, 2, 1, 3).contiguous().view(total_tokens, num_kv_heads, head_dim)
        )
        v_buffer = (
            value_states.permute(0, 2, 1, 3).contiguous().view(total_tokens, num_kv_heads, head_dim)
        )

        # qo_indptr: [batch+1], cumulative Q token counts per sequence
        # Each sequence has exactly seq_len tokens, so indptr = [0, seq_len, 2*seq_len, ...]
        qo_indptr = torch.arange(0, batch + 1, device=device, dtype=torch.int32) * seq_len

        # No prefix cache in teacher-forcing: all KV tokens are "extend" tokens.
        # kv_indptr == qo_indptr, kv_indices == sequential [0, 1, ..., total_tokens-1].
        kv_indptr = qo_indptr.clone()
        kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)

        # prefix_lens = 0 for every sequence (pure prefill, no radix cache)
        prefix_lens = torch.zeros(batch, device=device, dtype=torch.int32)

        o = torch.empty_like(q_varlen)

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

        attn_output = o.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return triton_forward


def apply_sglang_triton_attention_patch(model):
    """Replace attention layer forwards with unified extend attention for true on-policy training."""
    patched = 0
    for name, module in model.named_modules():
        if not (
            hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "o_proj")
        ):
            continue
        if getattr(module, "_sglang_triton_patched", False):
            continue
        original_forward = module.__class__.forward
        new_forward = _make_triton_attention_forward(original_forward)
        module.forward = types.MethodType(new_forward, module)
        module._sglang_triton_patched = True
        patched += 1
    return patched
