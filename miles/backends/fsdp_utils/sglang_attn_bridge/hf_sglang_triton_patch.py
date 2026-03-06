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


def _dump_view(x: torch.Tensor):
    # Dump raw dtype by default; never cast compute tensors for dump.
    out = x.detach()
    return out, out.dtype, out.dtype


def _flat_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 1:
        return x.contiguous().view(-1)
    if x.ndim == 2:
        return x.contiguous()
    if x.ndim == 3:
        # [B, S, H] -> [B*S, H]
        return x.contiguous().view(-1, x.shape[-1])
    if x.ndim == 4:
        # Normalize to [B, S, H, D], then flatten to [B*S, H*D].
        # q/k norm tensors are typically [B, H, S, D] in HF attention internals.
        if x.shape[1] <= x.shape[2]:
            y = x.permute(0, 2, 1, 3).contiguous()
        else:
            y = x.contiguous()
        return y.view(y.shape[0] * y.shape[1], y.shape[2] * y.shape[3])
    return x.contiguous().view(x.shape[0], -1)


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
    dump_value, _, _ = _dump_view(value)
    _dumper.dump(name, dump_value)


def _is_layer0_or_last(module):
    layer_idx = getattr(module, "layer_idx", None)
    num_layers = getattr(getattr(module, "config", None), "num_hidden_layers", None)
    is_layer0 = layer_idx == 0
    is_last = layer_idx is not None and num_layers is not None and layer_idx == num_layers - 1
    return is_layer0, is_last


def _make_norm_pre_hook(name: str):
    def _hook(_mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            _maybe_dump(name, _flat_last_dim(x))

    return _hook


def _make_norm_post_hook(name: str):
    def _hook(_mod, _inputs, output):
        if isinstance(output, torch.Tensor):
            _maybe_dump(name, _flat_last_dim(output))

    return _hook


def _register_norm_dump_hooks(model) -> int:
    hook_count = 0
    for _, module in model.named_modules():
        if not (hasattr(module, "q_norm") and hasattr(module, "k_norm")):
            continue
        if getattr(module, "_miles_norm_dump_hooks_registered", False):
            continue

        is_layer0, is_last = _is_layer0_or_last(module)
        if not (is_layer0 or is_last):
            continue

        if is_layer0:
            module.q_norm.register_forward_pre_hook(_make_norm_pre_hook("layer0_q_pre_norm"))
            module.q_norm.register_forward_hook(_make_norm_post_hook("layer0_q_post_norm"))
            module.k_norm.register_forward_pre_hook(_make_norm_pre_hook("layer0_k_pre_norm"))
            module.k_norm.register_forward_hook(_make_norm_post_hook("layer0_k_post_norm"))
            hook_count += 4

        if is_last:
            module.q_norm.register_forward_pre_hook(_make_norm_pre_hook("q_pre_norm"))
            module.q_norm.register_forward_hook(_make_norm_post_hook("q_post_norm"))
            module.k_norm.register_forward_pre_hook(_make_norm_pre_hook("k_pre_norm"))
            module.k_norm.register_forward_hook(_make_norm_post_hook("k_post_norm"))
            hook_count += 4

        module._miles_norm_dump_hooks_registered = True

    return hook_count


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

    del attention_mask, scaling, dropout, kwargs

    B, num_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    total = B * S

    # Determine layer position for selective dumping
    is_layer0, is_last = _is_layer0_or_last(module)

    # Dump q/k (post-rope) and v (post-proj) as received from HF — no modification
    if is_layer0 or is_last:
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

    q = query.transpose(1, 2).contiguous().view(total, num_heads, D)
    k = key.transpose(1, 2).contiguous().view(total, num_kv_heads, D)
    v = value.transpose(1, 2).contiguous().view(total, num_kv_heads, D)

    supported_dtypes = {torch.float16, torch.bfloat16}
    if q.dtype not in supported_dtypes or k.dtype not in supported_dtypes or v.dtype not in supported_dtypes:
        raise TypeError(
            "extend_attention_fwd_unified only supports fp16/bf16 in bridge compute path. "
            f"Got q={q.dtype}, k={k.dtype}, v={v.dtype}."
        )

    o = torch.empty_like(q)
    device = q.device
    qo_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * S
    kv_indptr = qo_indptr.clone()
    kv_indices = torch.arange(total, device=device, dtype=torch.int64)
    prefix_lens = torch.zeros(B, device=device, dtype=torch.int32)

    extend_attention_fwd_unified(
        q,
        o,
        k,
        v,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        max_len_extend=S,
        is_causal=True,
    )

    # Dump attention output before o_proj
    if is_layer0:
        _maybe_dump("layer0_attn_context_before_o_proj", o.view(total, -1))
    if is_last:
        _maybe_dump("attn_context_before_o_proj", o.view(total, -1))

    attn_output = o.view(B, S, num_heads, D)
    return attn_output, None


def apply_sglang_triton_attention_patch(model):
    """Register SGLang Triton as attention backend and activate it on the model.

    Uses HF's ALL_ATTENTION_FUNCTIONS registry — no monkey-patching of forward methods.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["sglang_triton"] = _sglang_triton_attention
    model.config._attn_implementation = "sglang_triton"
    _register_norm_dump_hooks(model)

    patched = sum(
        1
        for _, m in model.named_modules()
        if hasattr(m, "q_proj") and hasattr(m, "o_proj")
    )
    return patched
