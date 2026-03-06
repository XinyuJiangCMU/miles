#!/usr/bin/env python3
"""
HF tensor capture aligned to SGLang for Miles alignment experiments.

Structure
---------
§1  Infrastructure       – dumper, dtype utils
§2  Model setup          – load, FSDP wrap, dist init, SG generate
§3  Execution policy     – hooks that MUTATE the real forward (dtype casts)
§4  Observers            – hooks that READ and dump only (no mutation)
§5  Reference builders   – hooks that re-compute values for HF↔SG comparison
§6  Capture entry-point  – wires everything together
§7  CLI
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch
import torch.distributed as dist

from batch_invariant_trace import enable_miles_batch_invariant


# ─────────────────────────────────────────────────────────────────────────────
# §1  Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_get_dumper():
    try:
        return __import__("sglang.srt.debug_utils.dumper", fromlist=["dumper"]).dumper
    except Exception:
        return None


_DUMPER = _maybe_get_dumper()


def _dump(name: str, value) -> None:
    if _DUMPER is None:
        return
    try:
        _DUMPER.dump(name, value)
    except Exception:
        pass


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


# ─────────────────────────────────────────────────────────────────────────────
# §2  Model setup
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, torch_dtype: torch.dtype, attn_implementation: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    ).cuda()


def apply_fsdp2_minimal(model, *, use_fp16: bool):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    layer_cls_to_wrap = getattr(model, "_no_split_modules", None)
    assert layer_cls_to_wrap and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for _, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    param_dtype = torch.float16 if use_fp16 else torch.bfloat16
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32),
        "mesh": None,
        "offload_policy": None,
    }
    for module in modules:
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    return model


def init_single_process_dist() -> bool:
    if dist.is_initialized():
        return False
    rendezvous = tempfile.NamedTemporaryFile(prefix="miles_fsdp_align_", delete=False)
    rendezvous.close()
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous.name}",
        rank=0,
        world_size=1,
    )
    return True


def destroy_dist(initialized_here: bool) -> None:
    if initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def sglang_generate_with_logprobs(
    host: str, port: int, prompt_ids: list[int], sampling_params: dict[str, Any]
) -> tuple[list[int], list[float]]:
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "stream": False,
    }
    response = requests.post(f"http://{host}:{port}/generate", json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"SGLang request failed: {response.status_code} {response.text}")
    ret = response.json()
    if isinstance(ret, list):
        ret = ret[0]
    output_ids = ret["output_ids"]
    token_lps = ret.get("meta_info", {}).get("output_token_logprobs", [])
    gen_logprobs = [float(x[0]) for x in token_lps[: len(output_ids)]]
    return prompt_ids + output_ids, gen_logprobs


# ─────────────────────────────────────────────────────────────────────────────
# §3  Execution policy
#
#   These hooks MUTATE the real forward pass.  Every dtype cast that affects
#   actual computation lives here – nothing else should silently change dtype.
#
#   Set any flag to False to disable that mutation and observe the effect.
# ─────────────────────────────────────────────────────────────────────────────

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
    # Pre-hook on each decoder layer: cast hidden_states to fp32 on entry.
    # Effect: `residual = hidden_states` inside forward saves fp32.
    # Then `residual + attn_out(bf16)` auto-upcasts to fp32 (PyTorch type promotion).
    # The residual stream stays fp32 throughout, matching SG behavior.
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
    # Post-hook on each decoder layer: cast block output back to bf16.
    # Must be registered BEFORE observer post-hooks so the observer sees bf16.
    # (Observer pre-hooks are registered before policy; observer post-hooks after.)
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


# ─────────────────────────────────────────────────────────────────────────────
# §4  Observers
#
#   Read-only hooks: dump real forward values, NEVER mutate tensors.
#
#   Registration order matters for correctness:
#     - Pre-hooks:  observers BEFORE policy  → see pre-mutation state
#       e.g. layer0_attn_input_after_prepare must see fp32 before attn_input_bf16 casts it
#     - Post-hooks: policy BEFORE observers  → see post-mutation state
#       e.g. layer0_block_out must see bf16 after block_out_bf16 casts it
#
#   Therefore observers are split into two functions called at different points
#   in hf_get_tensor_dumps:
#     1. register_observer_prehooks()   – called before policy hooks
#     2. register_observer_posthooks()  – called after policy hooks
# ─────────────────────────────────────────────────────────────────────────────

def _register_layer_observer_prehooks(handles, layer, hf_slice, prefix: str):
    """Register pre-hook observers for a given layer with the given name prefix."""

    # block raw input (before input_layernorm)
    def _obs_block_input(_module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is not None:
            _dump(f"{prefix}_attn_input_raw", hf_slice(hs))

    handles.append(layer.register_forward_pre_hook(_obs_block_input, with_kwargs=True))

    # self_attn input BEFORE policy cast
    def _obs_attn_pre_cast(_module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is not None:
            _dump(f"{prefix}_attn_input_after_prepare", hf_slice(hs))
            _dump(f"{prefix}_hidden_in", hf_slice(hs.to(torch.bfloat16)))
        return None

    handles.append(layer.self_attn.register_forward_pre_hook(_obs_attn_pre_cast, with_kwargs=True))

    # residual stream entering post_attention_layernorm
    def _obs_residual(_module, args, kwargs):
        x = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("hidden_states")
        if x is not None:
            _dump(f"{prefix}_residual", hf_slice(x))

    if hasattr(layer, "post_attention_layernorm"):
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                _obs_residual, with_kwargs=True
            )
        )

    # context tensor entering o_proj
    def _obs_pre_o_proj(_module, args, kwargs):
        if args and isinstance(args[0], torch.Tensor):
            _dump(f"{prefix}_attn_context_before_o_proj", hf_slice(args[0]))

    if hasattr(layer.self_attn, "o_proj"):
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                _obs_pre_o_proj, with_kwargs=True
            )
        )


def register_observer_prehooks(model, hf_slice) -> list:
    """Pre-hook observers. Must be registered BEFORE execution policy hooks."""
    handles = []
    _register_layer_observer_prehooks(handles, model.model.layers[0], hf_slice, "layer0")
    _register_layer_observer_prehooks(handles, model.model.layers[1], hf_slice, "layer1")
    return handles


def _register_layer_observer_posthooks(handles, layer, hf_slice, prefix: str):
    """Register post-hook observers for a given layer with the given name prefix."""

    # attention module output (after o_proj)
    def _obs_attn_out(_module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        if out is not None:
            _dump(f"{prefix}_attn_out", hf_slice(out))

    handles.append(layer.self_attn.register_forward_hook(_obs_attn_out, with_kwargs=True))

    # MLP output (= block delta before residual add)
    def _obs_mlp_out(_module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        if out is not None:
            _dump(f"{prefix}_block_out_before_residual_add", hf_slice(out))

    if hasattr(layer, "mlp"):
        handles.append(layer.mlp.register_forward_hook(_obs_mlp_out, with_kwargs=True))

    # full block output (after residual add + optional block_out_bf16 cast)
    def _obs_block_out(_module, args, kwargs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs is not None:
            _dump(f"{prefix}_block_out_after_residual_add", hf_slice(hs))
            _dump(f"{prefix}_block_out", hf_slice(hs))

    handles.append(layer.register_forward_hook(_obs_block_out, with_kwargs=True))


def register_observer_posthooks(model, hf_slice) -> tuple[list, dict]:
    """
    Post-hook observers. Must be registered AFTER execution policy hooks so they
    see the post-mutation output (e.g. block_out already cast to bf16).
    Returns (handles, last_attn_out_store).
    """
    handles = []
    _register_layer_observer_posthooks(handles, model.model.layers[0], hf_slice, "layer0")
    _register_layer_observer_posthooks(handles, model.model.layers[1], hf_slice, "layer1")

    # last layer: attn input + output (stored for post-forward dump)
    last_layer = model.model.layers[-1]
    last_attn_out_store: dict[str, torch.Tensor] = {}

    def _obs_last_attn(_module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        if out is not None:
            last_attn_out_store["value"] = out
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is not None:
            _dump("attn_input_last_layer", hf_slice(hs))

    handles.append(last_layer.self_attn.register_forward_hook(_obs_last_attn, with_kwargs=True))

    return handles, last_attn_out_store


# ─────────────────────────────────────────────────────────────────────────────
# §5  Reference builders
#
#   These re-compute values OUTSIDE the real forward (extra projection/norm
#   calls under no_grad) to produce reference tensors for HF↔SG comparison.
#   They do NOT affect the real forward path.
# ─────────────────────────────────────────────────────────────────────────────

def _sglang_style_qk_rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Exact fp32 QK-RMSNorm as executed by SGLang (reference, not HF native)."""
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normed = x_fp32 * torch.rsqrt(variance + eps)
    return weight.to(torch.float32) * normed


def _build_attn_references(prefix: str, hf_slice, _module, args, kwargs, output) -> None:
    """
    Re-runs QKV projections + QK norms + RoPE to produce reference dumps.
    Called from a post-hook; uses no_grad and does not affect real forward.

    Dumps (prefixed):
      q_pre_norm, k_pre_norm, v_pre_norm
      q_norm_input_native, k_norm_input_native   (native HF input to q/k norm)
      q_post_norm_native, k_post_norm_native       (native HF norm output)
      q_post_norm, k_post_norm                     (SG-style fp32 reference)
      q_post_rope, k_post_rope                     (SG-style fp32 + RoPE)
    """
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    if hidden_states is None:
        return

    with torch.no_grad():
        q = _module.q_proj(hidden_states)
        k = _module.k_proj(hidden_states)
        v = _module.v_proj(hidden_states)

    _dump(f"{prefix}q_pre_norm", hf_slice(q))
    _dump(f"{prefix}k_pre_norm", hf_slice(k))
    _dump(f"{prefix}v_pre_norm", hf_slice(v))

    num_heads = getattr(_module, "num_heads", None) or getattr(
        getattr(_module, "config", None), "num_attention_heads", None
    )
    num_kv_heads = getattr(_module, "num_key_value_heads", None) or getattr(
        getattr(_module, "config", None), "num_key_value_heads", num_heads
    )

    if not (hasattr(_module, "q_norm") and hasattr(_module, "k_norm") and num_heads and num_kv_heads):
        return

    q_head_dim = q.shape[-1] // num_heads
    k_head_dim = k.shape[-1] // num_kv_heads
    q_4d = q.view(q.shape[0], q.shape[1], num_heads, q_head_dim)
    k_4d = k.view(q.shape[0], q.shape[1], num_kv_heads, k_head_dim)

    # Native: dump inputs and actual HF norm output
    _dump(f"{prefix}q_norm_input_native", hf_slice(q_4d.reshape_as(q)))
    _dump(f"{prefix}k_norm_input_native", hf_slice(k_4d.reshape_as(k)))
    with torch.no_grad():
        qn_native = _module.q_norm(q_4d).reshape_as(q)
        kn_native = _module.k_norm(k_4d).reshape_as(k)
    _dump(f"{prefix}q_post_norm_native", hf_slice(qn_native))
    _dump(f"{prefix}k_post_norm_native", hf_slice(kn_native))

    # Reference: SGLang-style fp32 emulation
    qn = _sglang_style_qk_rmsnorm_fp32(q_4d, _module.q_norm.weight, _module.q_norm.variance_epsilon).reshape_as(q)
    kn = _sglang_style_qk_rmsnorm_fp32(k_4d, _module.k_norm.weight, _module.k_norm.variance_epsilon).reshape_as(k)
    _dump(f"{prefix}q_post_norm", hf_slice(qn))
    _dump(f"{prefix}k_post_norm", hf_slice(kn))

    # RoPE applied to reference Q/K
    position_embeddings = kwargs.get("position_embeddings")
    if not (isinstance(position_embeddings, tuple) and len(position_embeddings) == 2):
        return
    cos, sin = position_embeddings
    qn_heads = qn.view(q.shape[0], q.shape[1], num_heads, q_head_dim).transpose(1, 2)
    kn_heads = kn.view(k.shape[0], k.shape[1], num_kv_heads, k_head_dim).transpose(1, 2)
    with torch.no_grad():
        qr, kr = apply_rotary_pos_emb(qn_heads, kn_heads, cos, sin)
    _dump(f"{prefix}q_post_rope", hf_slice(qr.transpose(1, 2).reshape_as(qn)))
    _dump(f"{prefix}k_post_rope", hf_slice(kr.transpose(1, 2).reshape_as(kn)))


def register_reference_hooks(model, hf_slice) -> list:
    """Register reference-builder post-hooks. Returns handles."""
    handles = []
    layer0 = model.model.layers[0]
    layer1 = model.model.layers[1]
    last_layer = model.model.layers[-1]

    handles.append(
        layer0.self_attn.register_forward_hook(
            lambda m, a, k, o: _build_attn_references("layer0_", hf_slice, m, a, k, o),
            with_kwargs=True,
        )
    )
    handles.append(
        layer1.self_attn.register_forward_hook(
            lambda m, a, k, o: _build_attn_references("layer1_", hf_slice, m, a, k, o),
            with_kwargs=True,
        )
    )
    handles.append(
        last_layer.self_attn.register_forward_hook(
            lambda m, a, k, o: _build_attn_references("", hf_slice, m, a, k, o),
            with_kwargs=True,
        )
    )
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# §6  Capture entry-point
# ─────────────────────────────────────────────────────────────────────────────

def hf_get_tensor_dumps(
    model_path: str,
    token_ids: list[int],
    dtype: str,
    attn_implementation: str,
    batch_invariant: bool,
    true_on_policy: bool,
    policy: ExecutionPolicy | None = None,
) -> None:
    if policy is None:
        policy = ExecutionPolicy(residual_fp32=True, block_out_bf16=False)
    print(policy.summary())

    from transformers import AttentionInterface

    if attn_implementation == "triton":
        from hf_triton_attention import triton_attention_forward
        AttentionInterface.register("triton", triton_attention_forward)

    if batch_invariant and true_on_policy:
        enable_miles_batch_invariant(enable_bmm=False)

    torch_dtype = resolve_torch_dtype(dtype)
    model = load_model(model_path, torch_dtype, attn_implementation)
    model.eval()

    ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    _dump("input_ids_for_compare", ids)

    # Embeddings computed before FSDP so weight is a plain tensor (not DTensor).
    input_embeds = model.get_input_embeddings()(ids).float()
    _dump("embedding_output", input_embeds)

    layer0_positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.long).unsqueeze(0)
    _dump("layer0_positions", layer0_positions)

    init_here = init_single_process_dist()
    try:
        model = apply_fsdp2_minimal(model, use_fp16=(torch_dtype == torch.float16))

        def hf_slice(x: torch.Tensor) -> torch.Tensor:
            """Drop the last token from seq dim (HF sees full seq; SG sees prefix only)."""
            return x[:, :-1, ...] if x is not None and x.ndim >= 3 else x

        all_handles: list = []

        # §4a observer pre-hooks – see pre-mutation state (e.g. fp32 before attn bf16 cast)
        all_handles += register_observer_prehooks(model, hf_slice)

        # §3 execution policy – mutating pre+post hooks
        all_handles += register_execution_policy_hooks(model, policy)

        # §4b observer post-hooks – see post-mutation state (e.g. bf16 after block_out cast)
        obs_post_handles, last_attn_out_store = register_observer_posthooks(model, hf_slice)
        all_handles += obs_post_handles

        # §5 reference builders – post-hooks, run after forward regardless of order
        all_handles += register_reference_hooks(model, hf_slice)

        with torch.no_grad():
            outputs = model(
                input_ids=None,
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits

            if "value" in last_attn_out_store:
                _dump("attn_out_last_layer", last_attn_out_store["value"][:, :-1, :])
            _dump("final_hidden_before_lm_head", outputs.hidden_states[-1][:, :-1, :])
            _dump("lm_head_weight", model.lm_head.weight)
            _dump("next_token_logits_raw", logits[:, :-1, :])

        for h in all_handles:
            try:
                h.remove()
            except Exception:
                pass
    finally:
        destroy_dist(init_here)


# ─────────────────────────────────────────────────────────────────────────────
# §7  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Miles FSDP-side tensors aligned to SGLang.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--attn-implementation", type=str, default="triton")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--true-on-policy", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dumper-root", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def load_prompt_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = load_prompt_config(args.prompt_config)
    prompt_ids = list(config["prompt_ids"])
    sampling_params = dict(config["sampling_params"])
    if args.max_new_tokens is not None:
        sampling_params["max_new_tokens"] = args.max_new_tokens

    full_ids, gen_logprobs = sglang_generate_with_logprobs(args.host, args.port, prompt_ids, sampling_params)
    print(f"[capture] sglang_full_ids={full_ids}")
    print(f"[capture] sglang_logprobs={gen_logprobs}")

    hf_get_tensor_dumps(
        model_path=args.model_path,
        token_ids=full_ids,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        batch_invariant=args.batch_invariant,
        true_on_policy=args.true_on_policy,
    )


if __name__ == "__main__":
    main()
