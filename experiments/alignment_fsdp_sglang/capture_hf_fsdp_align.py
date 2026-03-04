#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import requests
import torch
import torch.distributed as dist

from batch_invariant_trace import enable_miles_batch_invariant


def _maybe_get_dumper():
    try:
        return __import__("sglang.srt.debug_utils.dumper", fromlist=["dumper"]).dumper
    except Exception:
        return None


_DUMPER = _maybe_get_dumper()


def _maybe_dump(name: str, value) -> None:
    if _DUMPER is None:
        return
    try:
        _DUMPER.dump(name, value)
    except Exception:
        pass


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


def load_prompt_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
        ),
        "mesh": None,
        "offload_policy": None,
    }

    for module in modules:
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    return model


def sglang_generate_with_logprobs(host: str, port: int, prompt_ids: list[int], sampling_params: dict[str, Any]) -> tuple[list[int], list[float]]:
    base_url = f"http://{host}:{port}"
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "stream": False,
    }
    response = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"SGLang request failed: {response.status_code} {response.text}")
    ret = response.json()
    if isinstance(ret, list):
        ret = ret[0]
    output_ids = ret["output_ids"]
    meta = ret.get("meta_info", {})
    token_lps = meta.get("output_token_logprobs", [])
    gen_logprobs = [float(x[0]) for x in token_lps[: len(output_ids)]]
    return prompt_ids + output_ids, gen_logprobs


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


def hf_get_tensor_dumps(
    model_path: str,
    token_ids: list[int],
    dtype: str,
    attn_implementation: str,
    batch_invariant: bool,
    true_on_policy: bool,
) -> None:
    from transformers import AttentionInterface, AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    if attn_implementation == "triton":
        from hf_triton_attention import triton_attention_forward

        AttentionInterface.register("triton", triton_attention_forward)

    if batch_invariant and true_on_policy:
        enable_miles_batch_invariant(enable_bmm=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype),
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    ).cuda()
    model.eval()

    ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    _maybe_dump("input_ids_for_compare", ids)
    # Compute embeddings before FSDP wrapping so embedding.weight is still a plain tensor,
    # not a DTensor. Later forward uses inputs_embeds only.
    input_embeds = model.get_input_embeddings()(ids).float()
    _maybe_dump("embedding_output", input_embeds)

    init_here = init_single_process_dist()
    try:
        model = apply_fsdp2_minimal(model, use_fp16=(resolve_torch_dtype(dtype) == torch.float16))

        def _hf_slice(x: torch.Tensor) -> torch.Tensor:
            if x is None:
                return x
            if x.ndim >= 3:
                return x[:, :-1, ...]
            return x

        layer0_positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.long).unsqueeze(0)
        _maybe_dump("layer0_positions", layer0_positions)

        def _norm_pre_fp32(_module, args, kwargs):
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                return (args[0].float(),) + tuple(args[1:]), kwargs
            return None

        def _norm_post_fp32(_module, args, kwargs, output):
            if isinstance(output, torch.Tensor):
                return output.float()
            return output

        def _qk_norm_pre_fp32(_module, args):
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                return (args[0].float(),) + tuple(args[1:])
            return None

        def _sglang_style_qk_rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
            normed = x_fp32 * torch.rsqrt(variance + eps)
            return weight.to(torch.float32) * normed

        def _capture_norm_outputs(
            prefix: str,
            q: torch.Tensor,
            k: torch.Tensor,
            q_4d: torch.Tensor,
            k_4d: torch.Tensor,
            q_norm_module,
            k_norm_module,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Native inputs are the exact tensors passed into HF q_norm/k_norm,
            # flattened back to the compare-friendly layout.
            _maybe_dump(f"{prefix}q_norm_input_native", _hf_slice(q_4d.reshape_as(q)))
            _maybe_dump(f"{prefix}k_norm_input_native", _hf_slice(k_4d.reshape_as(k)))

            # Native outputs come from the actual HF Qwen3RMSNorm.forward().
            qn_native = q_norm_module(q_4d).reshape_as(q)
            kn_native = k_norm_module(k_4d).reshape_as(k)
            _maybe_dump(f"{prefix}q_post_norm_native", _hf_slice(qn_native))
            _maybe_dump(f"{prefix}k_post_norm_native", _hf_slice(kn_native))

            # Aligned outputs preserve the existing SGLang-style fp32 emulation path.
            qn = _sglang_style_qk_rmsnorm_fp32(
                q_4d,
                q_norm_module.weight,
                q_norm_module.variance_epsilon,
            ).reshape_as(q)
            kn = _sglang_style_qk_rmsnorm_fp32(
                k_4d,
                k_norm_module.weight,
                k_norm_module.variance_epsilon,
            ).reshape_as(k)
            _maybe_dump(f"{prefix}q_post_norm", _hf_slice(qn))
            _maybe_dump(f"{prefix}k_post_norm", _hf_slice(kn))
            return qn, kn

        def _self_attn_pre_bf16(_module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            if hs is None or not isinstance(hs, torch.Tensor):
                return None
            try:
                first_layer = model.model.layers[0]
                if getattr(first_layer, "self_attn", None) is _module:
                    _maybe_dump("layer0_attn_input_after_prepare", _hf_slice(hs))
                    _maybe_dump("layer0_hidden_in", _hf_slice(hs.to(torch.bfloat16)))
            except Exception:
                pass
            hs = hs.to(torch.bfloat16)
            if "hidden_states" in kwargs:
                new_kwargs = dict(kwargs)
                new_kwargs["hidden_states"] = hs
                return args, new_kwargs
            new_args = list(args)
            new_args[0] = hs
            return tuple(new_args), kwargs

        def _mlp_pre_bf16(_module, args, kwargs):
            x = kwargs.get("hidden_states", kwargs.get("x", args[0] if len(args) > 0 else None))
            if x is None or not isinstance(x, torch.Tensor):
                return None
            x = x.to(torch.bfloat16)
            if "hidden_states" in kwargs:
                new_kwargs = dict(kwargs)
                new_kwargs["hidden_states"] = x
                return args, new_kwargs
            if "x" in kwargs:
                new_kwargs = dict(kwargs)
                new_kwargs["x"] = x
                return args, new_kwargs
            new_args = list(args)
            new_args[0] = x
            return tuple(new_args), kwargs

        def _linear_input_to_weight_dtype(_module, args, kwargs):
            if len(args) == 0 or not isinstance(args[0], torch.Tensor):
                return None
            x = args[0]
            target_dtype = _module.weight.dtype
            if x.dtype != target_dtype:
                x = x.to(target_dtype)
            return (x,) + tuple(args[1:]), kwargs

        hook_handles = []
        first_layer = model.model.layers[0]
        last_layer = model.model.layers[-1]

        for layer in model.model.layers:
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                norm_mod = getattr(layer, norm_name, None)
                if norm_mod is not None:
                    hook_handles.append(norm_mod.register_forward_pre_hook(_norm_pre_fp32, with_kwargs=True))
                    hook_handles.append(norm_mod.register_forward_hook(_norm_post_fp32, with_kwargs=True))
            if hasattr(layer, "self_attn"):
                hook_handles.append(layer.self_attn.register_forward_pre_hook(_self_attn_pre_bf16, with_kwargs=True))
                q_norm_mod = getattr(layer.self_attn, "q_norm", None)
                if q_norm_mod is not None:
                    hook_handles.append(q_norm_mod.register_forward_pre_hook(_qk_norm_pre_fp32))
                k_norm_mod = getattr(layer.self_attn, "k_norm", None)
                if k_norm_mod is not None:
                    hook_handles.append(k_norm_mod.register_forward_pre_hook(_qk_norm_pre_fp32))
                o_proj_mod = getattr(layer.self_attn, "o_proj", None)
                if o_proj_mod is not None:
                    hook_handles.append(o_proj_mod.register_forward_pre_hook(_linear_input_to_weight_dtype, with_kwargs=True))
            if hasattr(layer, "mlp"):
                hook_handles.append(layer.mlp.register_forward_pre_hook(_mlp_pre_bf16, with_kwargs=True))

        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            hook_handles.append(
                model.lm_head.register_forward_pre_hook(_linear_input_to_weight_dtype, with_kwargs=True)
            )

        def _capture_layer0_raw(_module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            if hs is not None:
                _maybe_dump("layer0_attn_input_raw", _hf_slice(hs))

        hook_handles.append(first_layer.register_forward_pre_hook(_capture_layer0_raw, with_kwargs=True))

        def _capture_layer0_post_attn_residual(_module, args, kwargs):
            x = args[0] if len(args) > 0 and isinstance(args[0], torch.Tensor) else kwargs.get("hidden_states")
            if x is not None:
                _maybe_dump("layer0_residual", _hf_slice(x.to(torch.bfloat16)))

        if hasattr(first_layer, "post_attention_layernorm"):
            hook_handles.append(
                first_layer.post_attention_layernorm.register_forward_pre_hook(
                    _capture_layer0_post_attn_residual, with_kwargs=True
                )
            )

        if hasattr(first_layer.self_attn, "o_proj"):
            def _capture_layer0_o_proj_input(_module, args, kwargs):
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    _maybe_dump("layer0_attn_context_before_o_proj", _hf_slice(args[0]))
            hook_handles.append(
                first_layer.self_attn.o_proj.register_forward_pre_hook(
                    _capture_layer0_o_proj_input, with_kwargs=True
                )
            )

        def _capture_layer0_attn_out(_module, args, kwargs, output):
            out = output[0] if isinstance(output, tuple) else output
            if out is not None:
                _maybe_dump("layer0_attn_out", _hf_slice(out))
        hook_handles.append(first_layer.self_attn.register_forward_hook(_capture_layer0_attn_out, with_kwargs=True))

        def _capture_layer0_mlp_out(_module, args, kwargs, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            if mlp_out is not None:
                _maybe_dump("layer0_block_out_before_residual_add", _hf_slice(mlp_out.to(torch.bfloat16)))
        if hasattr(first_layer, "mlp"):
            hook_handles.append(first_layer.mlp.register_forward_hook(_capture_layer0_mlp_out, with_kwargs=True))

        def _capture_layer0_block_out(_module, args, kwargs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs is not None:
                _maybe_dump("layer0_block_out_after_residual_add", _hf_slice(hs.to(torch.bfloat16)))
                _maybe_dump("layer0_block_out", _hf_slice(hs.to(torch.bfloat16)))
        hook_handles.append(first_layer.register_forward_hook(_capture_layer0_block_out, with_kwargs=True))

        def _capture_attn_details(prefix: str, layer, _module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            if hidden_states is None:
                return
            q = _module.q_proj(hidden_states)
            k = _module.k_proj(hidden_states)
            v = _module.v_proj(hidden_states)
            _maybe_dump(f"{prefix}q_pre_norm", _hf_slice(q))
            _maybe_dump(f"{prefix}k_pre_norm", _hf_slice(k))
            _maybe_dump(f"{prefix}v_pre_norm", _hf_slice(v))

            num_heads = getattr(_module, "num_heads", None)
            if num_heads is None and hasattr(_module, "config"):
                num_heads = getattr(_module.config, "num_attention_heads", None)
            num_kv_heads = getattr(_module, "num_key_value_heads", None)
            if num_kv_heads is None and hasattr(_module, "config"):
                num_kv_heads = getattr(_module.config, "num_key_value_heads", num_heads)
            if (
                hasattr(_module, "q_norm")
                and hasattr(_module, "k_norm")
                and num_heads is not None
                and num_kv_heads is not None
            ):
                q_head_dim = q.shape[-1] // num_heads
                k_head_dim = k.shape[-1] // num_kv_heads
                q_4d = q.view(q.shape[0], q.shape[1], num_heads, q_head_dim)
                k_4d = k.view(q.shape[0], q.shape[1], num_kv_heads, k_head_dim)
                qn, kn = _capture_norm_outputs(
                    prefix=prefix,
                    q=q,
                    k=k,
                    q_4d=q_4d,
                    k_4d=k_4d,
                    q_norm_module=_module.q_norm,
                    k_norm_module=_module.k_norm,
                )

                position_embeddings = kwargs.get("position_embeddings")
                if isinstance(position_embeddings, tuple) and len(position_embeddings) == 2:
                    cos, sin = position_embeddings
                    qn_heads = qn.view(q.shape[0], q.shape[1], num_heads, q_head_dim).transpose(1, 2)
                    kn_heads = kn.view(k.shape[0], k.shape[1], num_kv_heads, k_head_dim).transpose(1, 2)
                    qr, kr = apply_rotary_pos_emb(qn_heads, kn_heads, cos, sin)
                    _maybe_dump(f"{prefix}q_post_rope", _hf_slice(qr.transpose(1, 2).reshape_as(qn)))
                    _maybe_dump(f"{prefix}k_post_rope", _hf_slice(kr.transpose(1, 2).reshape_as(kn)))

            if prefix == "":
                out = output[0] if isinstance(output, tuple) else output
                if out is not None:
                    _maybe_dump("attn_context_before_o_proj", _hf_slice(out))

        hook_handles.append(
            first_layer.self_attn.register_forward_hook(
                lambda m, a, k, o: _capture_attn_details("layer0_", first_layer, m, a, k, o),
                with_kwargs=True,
            )
        )
        hook_handles.append(
            last_layer.self_attn.register_forward_hook(
                lambda m, a, k, o: _capture_attn_details("", last_layer, m, a, k, o),
                with_kwargs=True,
            )
        )

        attn_out_last_layer = {}

        def _capture_last_attn_out(_module, args, kwargs, output):
            out = output[0] if isinstance(output, tuple) else output
            attn_out_last_layer["value"] = out
            hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            if hs is not None:
                _maybe_dump("attn_input_last_layer", _hf_slice(hs))
        hook_handles.append(last_layer.self_attn.register_forward_hook(_capture_last_attn_out, with_kwargs=True))

        with torch.no_grad():
            outputs = model(input_ids=None, inputs_embeds=input_embeds, output_hidden_states=True, return_dict=True)
            logits = outputs.logits
            if "value" in attn_out_last_layer:
                _maybe_dump("attn_out_last_layer", attn_out_last_layer["value"][:, :-1, :])
            _maybe_dump("final_hidden_before_lm_head", outputs.hidden_states[-1][:, :-1, :])
            _maybe_dump("lm_head_weight", model.lm_head.weight)
            _maybe_dump("next_token_logits_raw", logits[:, :-1, :])

        for handle in hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
    finally:
        destroy_dist(init_here)


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
