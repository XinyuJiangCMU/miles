"""Minimal patch: replace HF attention kernel with SGLang Triton extend_attention_fwd_unified.

Instead of monkey-patching the entire attention forward (qkv proj, norm, rope, o_proj),
we register a custom attention backend via HF's ALL_ATTENTION_FUNCTIONS. This way HF
keeps its own qkv_proj, norm, rope, o_proj logic — we ONLY replace the core attention
computation (Q@K^T softmax V) with extend_attention_fwd_unified.

The extend_attention_fwd_unified kernel uses per-request indptrs, which naturally
gives batch-invariant results (each request is computed independently).
"""

import torch

_dumper = None


def _maybe_dump(name: str, value: torch.Tensor) -> None:
    """Lazily import sglang dumper and dump if enabled."""
    global _dumper
    if _dumper is False:
        return
    if _dumper is None:
        try:
            _dumper = __import__(
                "sglang.srt.debug_utils.dumper", fromlist=["dumper"]
            ).dumper
        except Exception:
            _dumper = False
            return
    _dumper.dump(name, value)


def _sglang_triton_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Drop-in replacement for HF's attention_interface.

    Input:  query [B, num_heads, S, D], key/value [B, num_kv_heads, S, D]
    Output: attn_output [B, S, num_heads, D], None

    Dumps intermediates for layer 0 and last layer (read-only, no computation change).
    """
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd_unified,
    )

    B, num_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    total = B * S

    # Determine layer position for selective dumping
    layer_idx = getattr(module, "layer_idx", None)
    num_layers = getattr(getattr(module, "config", None), "num_hidden_layers", None)
    is_layer0 = layer_idx == 0
    is_last = layer_idx is not None and num_layers is not None and layer_idx == num_layers - 1

    # Dump q/k (post-rope) and v (post-proj) as received from HF — no modification
    if is_layer0 or is_last:
        prefix = "layer0" if is_layer0 else ""
        # q/k are post-rope here; v is post-proj (no norm/rope applied to v)
        q_flat = query.permute(0, 2, 1, 3).contiguous().view(total, -1)
        k_flat = key.permute(0, 2, 1, 3).contiguous().view(total, -1)
        v_flat = value.permute(0, 2, 1, 3).contiguous().view(total, -1)
        if is_layer0:
            _maybe_dump("layer0_q_post_rope", q_flat)
            _maybe_dump("layer0_k_post_rope", k_flat)
            _maybe_dump("layer0_v_pre_norm", v_flat)
        if is_last:
            _maybe_dump("q_post_rope", q_flat)
            _maybe_dump("k_post_rope", k_flat)
            _maybe_dump("v_pre_norm", v_flat)

    orig_dtype = query.dtype
    q = query.to(torch.bfloat16).transpose(1, 2).contiguous().view(total, num_heads, D)
    k = key.to(torch.bfloat16).transpose(1, 2).contiguous().view(total, num_kv_heads, D)
    v = value.to(torch.bfloat16).transpose(1, 2).contiguous().view(total, num_kv_heads, D)

    o = torch.empty_like(q)
    device = q.device
    qo_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * S
    kv_indptr = qo_indptr.clone()
    kv_indices = torch.arange(total, device=device, dtype=torch.int64)
    prefix_lens = torch.zeros(B, device=device, dtype=torch.int32)

    extend_attention_fwd_unified(
        q, o, k, v,
        qo_indptr, kv_indptr, kv_indices, prefix_lens,
        max_len_extend=S,
        is_causal=True,
    )

    # Dump attention output before o_proj
    if is_layer0:
        _maybe_dump("layer0_attn_context_before_o_proj", o.view(total, -1))
    if is_last:
        _maybe_dump("attn_context_before_o_proj", o.view(total, -1))

    attn_output = o.view(B, S, num_heads, D).to(orig_dtype)
    return attn_output, None


def apply_sglang_triton_attention_patch(model):
    """Register SGLang Triton as attention backend and activate it on the model.

    Uses HF's ALL_ATTENTION_FUNCTIONS registry — no monkey-patching of forward methods.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["sglang_triton"] = _sglang_triton_attention
    model.config._attn_implementation = "sglang_triton"

    # Count attention modules for logging
    patched = sum(
        1 for _, m in model.named_modules()
        if hasattr(m, "q_proj") and hasattr(m, "o_proj")
    )
    return patched
