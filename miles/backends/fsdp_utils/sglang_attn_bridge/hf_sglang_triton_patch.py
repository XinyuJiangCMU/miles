"""HF-side monkey patch entry for the SGLang Triton attention bridge."""

import types
import torch


def _is_patchable_attention(module) -> bool:
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
    )


def run_unified_extend(
    q_varlen: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    batch: int,
    seq_len: int,
) -> torch.Tensor:
    """Execute extend_attention_fwd_unified in teacher-forcing prefill mode."""
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


def apply_sglang_triton_attention_patch(model):
    """Patch HF attention modules to use the SGLang Triton unified-extend path."""
    from .models.qwen3 import qwen3_triton_forward

    patched = 0
    for _name, module in model.named_modules():
        if not _is_patchable_attention(module):
            continue
        if getattr(module, "_sglang_triton_patched", False):
            continue

        module.forward = types.MethodType(qwen3_triton_forward, module)
        module._sglang_triton_patched = True
        patched += 1
    return patched

