"""HF-side monkey patch entry for the SGLang Triton attention bridge."""

import types

from .models.qwen3 import qwen3_triton_forward


def _is_patchable_attention(module) -> bool:
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
    )


def apply_sglang_triton_attention_patch(model):
    """Patch HF attention modules to use the SGLang Triton unified-extend path."""
    patched = 0
    for _name, module in model.named_modules():
        if not _is_patchable_attention(module):
            continue
        if getattr(module, "_sglang_triton_patched", False):
            continue

        module._sglang_triton_original_forward = module.forward
        module.forward = types.MethodType(qwen3_triton_forward, module)
        module._sglang_triton_patched = True
        patched += 1
    return patched

