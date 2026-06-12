# AMD gfx950 (MI350X/MI355X) per-commit CI fixes — 2026-06-12

Verified fixes for the MI350-class CI error classes. Branch wip/miles-ci-20260612
is a single consolidated backup; split into per-fix PRs as appropriate.

## In-tree miles source changes (live on this branch)

- Error B — miles_plugins/megatron_bridge/__init__.py: guard
  megatron.bridge.remove_non_pickleables against ProcessGroup (incl. miles
  ReloadableProcessGroup) so colocate + --megatron-to-hf-mode bridge
  update_weights does not die with "cannot pickle ReloadableProcessGroup".
- Error D — tests/e2e/conftest_dumper.py: replace fragile multi-line dumper
  source-patch anchors with stable single-line anchors (fixes
  PatchApplicationError: match text not found on unpinned sglang drift).

## Cross-repo patch files (apply at image build; NOT yet wired into Dockerfile)

These target repos other than miles, staged here as patch files for review/
splitting. They are NOT applied by Dockerfile.rocm_MI350-5 yet — wire them next
to the existing megatron.patch apply step when promoting to a PR.

- transformer_engine_wgrad_dbias.patch (Error A) -> ROCm/TransformerEngine v2.8_rocm,
  transformer_engine/pytorch/module/layernorm_linear.py. Verified GPU before/after:
  "Unable to find any suitable algorithms" repro -> 8/8 OK. Also PR
  XinyuJiangCMU/TransformerEngine#1.
- megatron_fused_adam_import.patch (Error E) -> radixark/Megatron-LM miles-main,
  megatron/core/optimizer/fused_adam_patch.py. Verified import negative-control
  (wrong path ModuleNotFoundError -> .tensor. path OK).
- sglang_dsv4_qk_rmsnorm_fallback.patch (Error G) -> sglang image base (AMD
  if _use_aiter block), forward_mla.py. Reproduces import crash when aiter lacks
  the fused_qk_rmsnorm op; pure-torch fallback bitwise-equal to
  RMSNorm.forward_native. Lightweight: avoids bumping AITER_COMMIT / rebuilding base.
- sglang_mla_kv_a_proj_tp.patch (Error I) -> sgl-project/sglang,
  python/sglang/srt/models/transformers.py. Verified before/after:
  "Unsupported TP style mla_kv_a_proj" -> replicate; surgical.

## Not included (need more)

- F (session logprobs): current sglang chat endpoint DOES return
  meta_info.output_token_logprobs with len==completion_tokens (verified on a
  Qwen3-0.6B gfx950 server) — F is a CI runtime condition, needs the exact
  UpstreamResponseError text from CI logs to pinpoint. No blind fix.
- C (rollout_routed_experts): needs gfx950 DSv4-server instrumentation.
- H (DeepseekV32): already a registered alias in current sglang; deep V3.2
  rollout needs NSA-on-ROCm (out of scope).
