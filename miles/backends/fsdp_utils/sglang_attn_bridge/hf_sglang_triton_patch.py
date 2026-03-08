"""Minimal patch: replace HF attention kernel with SGLang Triton extend_attention_fwd_unified.

Instead of monkey-patching the entire attention forward (qkv proj, norm, rope, o_proj),
we register a custom attention backend via HF's ALL_ATTENTION_FUNCTIONS. This way HF
keeps its own qkv_proj, norm, rope, o_proj logic — we ONLY replace the core attention
computation (Q@K^T softmax V) with extend_attention_fwd_unified.

The extend_attention_fwd_unified kernel uses per-request indptrs, which naturally
gives batch-invariant results (each request is computed independently).

Gradient strategy
-----------------
Triton's extend_attention_fwd_unified has no fused backward kernel.  We solve this
with a custom autograd.Function (_TritonFwdSdpaBwd) that:

  - forward : runs extend_attention_fwd_unified (Triton) so the numerics match the
              SGLang inference-side Triton backend exactly (true on-policy alignment).
  - backward: re-runs F.scaled_dot_product_attention on the saved Q/K/V to obtain
              the exact analytic gradients dQ, dK, dV.

This is mathematically valid because both kernels compute the identical function
softmax(QK^T / sqrt(d)) @ V; the only difference is floating-point order.
The SDPA backward is therefore the correct gradient for the Triton forward.
"""

import torch
import torch.nn.functional as F


class _TritonFwdSdpaBwd(torch.autograd.Function):
    """Triton forward + SDPA recompute backward for causal self-attention.

    Tensors are in the flattened layout expected by extend_attention_fwd_unified:
        q : [B*S, num_heads,    D]
        k : [B*S, num_kv_heads, D]
        v : [B*S, num_kv_heads, D]
    Output:
        o : [B*S, num_heads,    D]
    """

    @staticmethod
    def forward(ctx, q, k, v, B, S):
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd_unified,
        )

        device = q.device
        qo_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * S
        kv_indptr = qo_indptr.clone()
        kv_indices = torch.arange(B * S, device=device, dtype=torch.int64)
        prefix_lens = torch.zeros(B, device=device, dtype=torch.int32)

        o = torch.empty_like(q)
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

        ctx.save_for_backward(q, k, v)
        ctx.B = B
        ctx.S = S
        return o

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [B*S, num_heads, D]  (same shape as forward output o)
        q, k, v = ctx.saved_tensors
        B, S = ctx.B, ctx.S
        nh = q.shape[1]
        nkv = k.shape[1]
        D = q.shape[2]

        # Reshape [B*S, H, D] -> [B, H, S, D] for SDPA
        q2 = q.view(B, S, nh, D).transpose(1, 2).detach().requires_grad_(True)
        k2 = k.view(B, S, nkv, D).transpose(1, 2).detach().requires_grad_(True)
        v2 = v.view(B, S, nkv, D).transpose(1, 2).detach().requires_grad_(True)

        with torch.enable_grad():
            # SDPA output: [B, H, S, D]; enable_gqa=True handles GQA (num_heads != num_kv_heads)
            o2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, enable_gqa=True)

        # grad_output [B*S, H, D] -> [B, H, S, D]
        go = grad_output.view(B, S, nh, D).transpose(1, 2).contiguous()
        o2.backward(go)

        # [B, H, S, D] -> [B*S, H, D]
        dq = q2.grad.transpose(1, 2).reshape(B * S, nh, D)
        dk = k2.grad.transpose(1, 2).reshape(B * S, nkv, D)
        dv = v2.grad.transpose(1, 2).reshape(B * S, nkv, D)

        # Return grads for (q, k, v, B, S); B and S are ints → None
        return dq, dk, dv, None, None


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

    Uses _TritonFwdSdpaBwd so that:
      - forward  is the Triton kernel (bitwise-aligned with SGLang inference)
      - backward flows through SDPA (correct dQ/dK/dV for all parameters)
    """
    del attention_mask, scaling, dropout, kwargs

    B, num_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    total = B * S

    q = query.transpose(1, 2).contiguous().view(total, num_heads, D)
    k = key.transpose(1, 2).contiguous().view(total, num_kv_heads, D)
    v = value.transpose(1, 2).contiguous().view(total, num_kv_heads, D)

    # Force kernel inputs to bf16 right before extend_unified to match
    # the SGLang inference-side Triton backend behavior.
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    o = _TritonFwdSdpaBwd.apply(q, k, v, B, S)

    attn_output = o.view(B, S, num_heads, D)
    return attn_output, None


def apply_sglang_triton_attention_patch(model):
    """Register SGLang Triton as attention backend and activate it on the model."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["triton"] = _sglang_triton_attention
    model.config._attn_implementation = "triton"

    patched = sum(
        1
        for _, m in model.named_modules()
        if hasattr(m, "q_proj") and hasattr(m, "o_proj")
    )
    return patched
