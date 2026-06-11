# miles CI Results on AMD MI350X (gfx950) — 2026-06-11

Per-test results for the miles CI tests exercised this session on AMD Instinct
MI350X. Only tests that were actually run are listed; each row reports what was
observed and the fix applied to make it pass on ROCm.

## Environment
- Hardware: AMD Instinct MI350X (gfx950), ROCm 7.2, hipBLASLt 7.2.26015
- Image: `xinyujiangcmu/miles:rocm720-mi35x-20260610`
- Source: run inside the container (`/root/miles`), Python 3.10
- TransformerEngine: `2.8.0+a365f2de`

## Fixes applied to run
1. **NVFP4 reference import (ROCm relocation)** — `test_nvfp4_quantizer.py`, the
   NVFP4 processor, and `convert_hf_to_nvfp4.py` import
   `transformer_engine.pytorch.custom_recipes.quantization_nvfp4`, which does not
   exist on ROCm TE. ROCm ships the same `NVFP4QuantizerRef` under
   `transformer_engine.pytorch.experimental.quantization_microblock_ref`. Added a
   `te_nvfp4_compat` shim (NVIDIA path first, ROCm experimental fallback) and routed
   the three call sites through it. No quantization logic changed.
2. **TE wgrad fused bias-grad on ROCm fp32-accumulate** — `layernorm_linear.py`
   backward unconditionally fuses dbias into the wgrad GEMM (BGRADB epilogue). On
   gfx950, hipBLASLt has no algorithm for a `bf16 -> fp32-accumulate` GEMM with that
   epilogue, so it raises `Unable to find any suitable algorithms`. Patched TE to
   skip the fusion on the ROCm fp32-accumulate path and reduce grad_bias separately
   (`grad_output.sum(tokens, fp32).to(bias.dtype)`, mathematically identical). CUDA
   and all other paths unchanged.
3. **Run prerequisites (env)** — `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` (Ray blanks
   `HIP_VISIBLE_DEVICES` on the 0-GPU job entrypoint otherwise →
   `No HIP GPUs are available`); `ulimit -n 524288` (default 1024 exhausts fds and
   the gateway dies with `Too many open files` after ~80 rollouts);
   `SGLANG_USE_AITER=0` (aiter rmsnorm path lacks the ROCm fp32-norm fallback).

## Branches / PRs
- `XinyuJiangCMU/miles:amd-fp4-quant-rocm` — NVFP4 ROCm import shim + new AMD MXFP4
  quantizer & test.
- `XinyuJiangCMU/TransformerEngine#1` — TE wgrad fused-dbias fix (fork `dev`).

## Error-type legend
- **A** `RuntimeError: Unable to find any suitable algorithms` — root-caused this
  session to the **TE `layernorm_linear` backward wgrad GEMM with a fused
  bias-gradient (BGRADB) epilogue on an fp32-accumulate output**, not the attention
  path. Isolated repro: a LayerNormLinear fails only when `bias` **and**
  `fuse_wgrad_accumulation` are both on (e.g. Qwen2.5 QKV `--add-qkv-bias` +
  `--accumulate-allreduce-grads-in-fp32`); plain GEMM or no-bias layers pass. Fixed
  by Fix #2.

---

## stage-b-2-gpu  (suite: stage-b-2-gpu-h200)

| # | Test path | Ran | Result | Notes |
|---|---|---|---|---|
| 1 | `tests/fast-gpu/test_nvfp4_quantizer.py` | yes | FAIL → **PASS (213 passed)** | Before: `ModuleNotFoundError: transformer_engine.pytorch.custom_recipes`. After Fix #1 (ROCm experimental `NVFP4QuantizerRef`), all 213 cases pass bitwise (`rtol=0 atol=0`) on gfx950. |
| 2 | `tests/fast-gpu/test_mxfp4_quantizer.py` | yes | **PASS (58 passed)** | New this session. Tests the new AMD MXFP4 (OCP microscaling, block 32, E8M0 scale) quantizer — pure torch, no TE dependency. Bitwise vs an independent OCP reference across 13 shapes × {random,boundary,zeros,maxes}, plus round-trip dequant cross-checked against SGLang's loader-side dequantize. |

---

## stage-c-2-gpu  (suite: stage-c-2-gpu-h200)

| # | Test path | Ran | Result | Notes |
|---|---|---|---|---|
| 1 | `tests/e2e/long/test_qwen2.5_0.5B_gsm8k_async.py` | yes | FAIL (Error **A**) → **PASS** | 2-GPU Megatron async GRPO. Before: crashed on the first train step with Error **A**. After Fix #2, training runs end-to-end; the async pipeline (rollout → train → cross-GPU weight sync) works on ROCm. CI metric check (`eval/gsm8k ≥ 0.55`, latched-OR over evals) passed: `[MetricChecker] check_success=True actual_value=0.5572` at rollout 200. eval trajectory: 0.5064 → 0.5140 → 0.5216 → 0.5322 → 0.5337 → 0.5406 → 0.5474 → 0.5367 → 0.5390 → **0.5572 (≥0.55)**. |
