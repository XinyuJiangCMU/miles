"""Qwen3 semantic adapter for the SGLang Triton HF patch path."""

import torch

from ..hf_sglang_triton_patch import run_unified_extend

_dumper = None


def _maybe_dump(name: str, value: torch.Tensor) -> None:
    """Mirror of ppo_utils._maybe_dump — lazily import sglang dumper."""
    global _dumper
    if _dumper is False:
        return
    if _dumper is None:
        try:
            _dumper = __import__(
                "sglang.srt.debug_utils.dumper",
                fromlist=["dumper"],
            ).dumper
        except Exception:
            _dumper = False
            return
    _dumper.dump(name, value)


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
    hidden_states = hidden_states.to(torch.bfloat16)
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

    # Determine layer position for selective dumping (mirrors SGLang qwen3.py logic).
    layer_id = getattr(self, "layer_idx", None)
    num_hidden_layers = getattr(getattr(self, "config", None), "num_hidden_layers", None)
    is_layer0 = layer_id == 0
    is_last_layer = (
        layer_id is not None
        and num_hidden_layers is not None
        and layer_id == num_hidden_layers - 1
    )

    total_tokens = batch * seq_len

    # SGLang Qwen3 semantic order: qkv -> qk_norm -> rope -> attention core -> o_proj.
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    # Split v_proj output before in-place reshape to allow dumping [T, kv_size].
    v_raw = self.v_proj(hidden_states)

    if is_layer0:
        _maybe_dump("layer0_q_pre_norm", query_states.view(total_tokens, -1))
        _maybe_dump("layer0_k_pre_norm", key_states.view(total_tokens, -1))
        _maybe_dump("layer0_v_pre_norm", v_raw.view(total_tokens, -1))
    if is_last_layer:
        _maybe_dump("q_pre_norm", query_states.view(total_tokens, -1))
        _maybe_dump("k_pre_norm", key_states.view(total_tokens, -1))
        _maybe_dump("v_pre_norm", v_raw.view(total_tokens, -1))

    value_states = v_raw.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
        # Match SGLang on-policy semantic order:
        # q/k are normalized in fp32, then rotary is applied in fp32,
        # and q/k are cast back to bf16 right before attention kernel.
        q_by_head = query_states.reshape(-1, head_dim).float()
        k_by_head = key_states.reshape(-1, head_dim).float()
        q_by_head = self.q_norm(q_by_head)
        k_by_head = self.k_norm(k_by_head)
        query_states = q_by_head.view_as(query_states).float()
        key_states = k_by_head.view_as(key_states).float()
    else:
        query_states = query_states.float()
        key_states = key_states.float()

    if is_layer0:
        _maybe_dump("layer0_q_post_norm", query_states.view(total_tokens, -1))
        _maybe_dump("layer0_k_post_norm", key_states.view(total_tokens, -1))
    if is_last_layer:
        _maybe_dump("q_post_norm", query_states.view(total_tokens, -1))
        _maybe_dump("k_post_norm", key_states.view(total_tokens, -1))

    query_states = query_states.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    if position_embeddings is not None and APPLY_ROTARY_POS_EMB is not None:
        cos, sin = position_embeddings
        query_states, key_states = APPLY_ROTARY_POS_EMB(
            query_states, key_states, cos, sin
        )

    # Dump post-rope in [T, q_size] / [T, kv_size] to match SGLang's varlen format.
    if is_layer0:
        _maybe_dump(
            "layer0_q_post_rope",
            query_states.permute(0, 2, 1, 3).contiguous().view(total_tokens, -1),
        )
        _maybe_dump(
            "layer0_k_post_rope",
            key_states.permute(0, 2, 1, 3).contiguous().view(total_tokens, -1),
        )
    if is_last_layer:
        _maybe_dump(
            "q_post_rope",
            query_states.permute(0, 2, 1, 3).contiguous().view(total_tokens, -1),
        )
        _maybe_dump(
            "k_post_rope",
            key_states.permute(0, 2, 1, 3).contiguous().view(total_tokens, -1),
        )

    query_states = query_states.to(torch.bfloat16)
    key_states = key_states.to(torch.bfloat16)
    value_states = value_states.to(torch.bfloat16)

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

    if is_layer0:
        _maybe_dump("layer0_attn_context_before_o_proj", o.view(total_tokens, -1))
    if is_last_layer:
        _maybe_dump("attn_context_before_o_proj", o.view(total_tokens, -1))

    # `o` is varlen output laid out as [B*S, H, D]. Restore to [B, S, H, D]
    # then flatten heads -> hidden dim as [B, S, H*D] for `o_proj`.
    attn_output = o.view(batch, seq_len, num_heads, head_dim)
    attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if is_layer0:
        _maybe_dump("layer0_attn_out_after_o_proj", attn_output.view(total_tokens, -1))
    if is_last_layer:
        _maybe_dump("attn_out_last_layer", attn_output.view(total_tokens, -1))

    return attn_output, None
