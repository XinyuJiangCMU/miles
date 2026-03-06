"""ExecutionPolicy: dtype mutation hooks to align HF FSDP forward pass with SGLang's dtype flow.

Copied from experiments/alignment_fsdp_sglang/capture_hf_fsdp_align.py and adapted
for integration into the MILES FSDP training pipeline.

SGLang uses fp32 for layernorm output (override_orig_dtype), then casts to bf16 before
attention kernels. HF's native Qwen3RMSNorm outputs bf16. These hooks force HF's dtype
flow to match SGLang's, enabling bitwise-identical intermediate tensors.
"""

from dataclasses import dataclass

import torch


@dataclass
class ExecutionPolicy:
    """Controls which dtype mutations are applied to the HF forward pass."""

    # input_layernorm / post_attention_layernorm run in fp32
    norm_io_fp32: bool = True

    # self_attn module receives bf16 input (SG attention kernels are bf16)
    attn_input_bf16: bool = True

    # q_norm / k_norm receive fp32 input
    qk_norm_input_fp32: bool = True

    # MLP module receives bf16 input
    mlp_input_bf16: bool = True

    # o_proj and lm_head inputs are cast to match their weight dtype
    linear_to_weight_dtype: bool = True

    # Each decoder layer receives fp32 hidden_states, so residual saves are fp32.
    # residual (fp32) + attn_out (bf16) auto-upcasts to fp32 in PyTorch.
    # This matches SG's fp32 residual stream.
    residual_fp32: bool = False

    # Each decoder layer output is cast back to bf16 after the final residual add.
    # SG keeps the block output in bf16; without this, residual_fp32 leaves it fp32.
    block_out_bf16: bool = False

    def summary(self) -> str:
        lines = ["[ExecutionPolicy]"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)


def register_execution_policy_hooks(model, policy: ExecutionPolicy) -> list:
    """Register all MUTATING hooks. Returns handles for later removal."""
    handles = []

    # --- norm fp32 ---
    if policy.norm_io_fp32:
        def _norm_pre_fp32(_module, args, kwargs):
            if args and isinstance(args[0], torch.Tensor):
                return (args[0].float(),) + tuple(args[1:]), kwargs
            return None

        def _norm_post_fp32(_module, args, kwargs, output):
            return output.float() if isinstance(output, torch.Tensor) else output

        for layer in model.model.layers:
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                mod = getattr(layer, norm_name, None)
                if mod is not None:
                    handles.append(mod.register_forward_pre_hook(_norm_pre_fp32, with_kwargs=True))
                    handles.append(mod.register_forward_hook(_norm_post_fp32, with_kwargs=True))

    # --- attn input bf16 ---
    if policy.attn_input_bf16:
        def _attn_pre_bf16(_module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None or not isinstance(hs, torch.Tensor):
                return None
            hs = hs.to(torch.bfloat16)
            if "hidden_states" in kwargs:
                return args, {**kwargs, "hidden_states": hs}
            new_args = list(args)
            new_args[0] = hs
            return tuple(new_args), kwargs

        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                handles.append(layer.self_attn.register_forward_pre_hook(_attn_pre_bf16, with_kwargs=True))

    # --- qk_norm input fp32 ---
    if policy.qk_norm_input_fp32:
        def _qk_norm_pre_fp32(_module, args):
            if args and isinstance(args[0], torch.Tensor):
                return (args[0].float(),) + tuple(args[1:])
            return None

        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            for norm_attr in ("q_norm", "k_norm"):
                mod = getattr(attn, norm_attr, None)
                if mod is not None:
                    handles.append(mod.register_forward_pre_hook(_qk_norm_pre_fp32))

    # --- mlp input bf16 ---
    if policy.mlp_input_bf16:
        def _mlp_pre_bf16(_module, args, kwargs):
            x = kwargs.get("hidden_states", kwargs.get("x", args[0] if args else None))
            if x is None or not isinstance(x, torch.Tensor):
                return None
            x = x.to(torch.bfloat16)
            if "hidden_states" in kwargs:
                return args, {**kwargs, "hidden_states": x}
            if "x" in kwargs:
                return args, {**kwargs, "x": x}
            new_args = list(args)
            new_args[0] = x
            return tuple(new_args), kwargs

        for layer in model.model.layers:
            if hasattr(layer, "mlp"):
                handles.append(layer.mlp.register_forward_pre_hook(_mlp_pre_bf16, with_kwargs=True))

    # --- residual stream fp32 ---
    if policy.residual_fp32:
        def _layer_pre_fp32(_module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None or not isinstance(hs, torch.Tensor):
                return None
            hs = hs.float()
            if "hidden_states" in kwargs:
                return args, {**kwargs, "hidden_states": hs}
            new_args = list(args)
            new_args[0] = hs
            return tuple(new_args), kwargs

        for layer in model.model.layers:
            handles.append(layer.register_forward_pre_hook(_layer_pre_fp32, with_kwargs=True))

    # --- block output bf16 ---
    if policy.block_out_bf16:
        def _layer_post_bf16(_module, args, kwargs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if isinstance(hs, torch.Tensor) and hs.dtype != torch.bfloat16:
                hs = hs.to(torch.bfloat16)
                return (hs,) + output[1:] if isinstance(output, tuple) else hs
            return output

        for layer in model.model.layers:
            handles.append(layer.register_forward_hook(_layer_post_bf16, with_kwargs=True))

    # --- linear input → weight dtype ---
    if policy.linear_to_weight_dtype:
        def _linear_to_weight_dtype(_module, args, kwargs):
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            x = args[0]
            if x.dtype != _module.weight.dtype:
                x = x.to(_module.weight.dtype)
            return (x,) + tuple(args[1:]), kwargs

        for layer in model.model.layers:
            o_proj = getattr(getattr(layer, "self_attn", None), "o_proj", None)
            if o_proj is not None:
                handles.append(o_proj.register_forward_pre_hook(_linear_to_weight_dtype, with_kwargs=True))
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            handles.append(model.lm_head.register_forward_pre_hook(_linear_to_weight_dtype, with_kwargs=True))

    return handles
