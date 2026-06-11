# miles CI Results on AMD MI300X (gfx942) — 2026-06-11

Per-test results for the miles CI suites exercised this session on AMD Instinct
MI300X. Each row reports what was observed and the fix applied to make it pass on
ROCm. MI300X runs the ROCm 7.0 (`rocm700`) stack; FP4/FP6 hardware paths are
MI350 (gfx950)-only, so NV-format quant tests are out of scope here (legend **D**).

## Environment
- Hardware: 8x AMD Instinct MI300X (gfx942), ROCm 7.0, torch hip `7.0.51831`
- Image: `rocm/sgl-dev:miles-rocm700-mi30x-20260610`
- Source: run inside the container, Python 3.10
- TransformerEngine: `2.8.0+a365f2de`

## Fixes applied to run
1. **Py3.10 `StrEnum` guard** — `miles/utils/chat_template_utils/tito_tokenizer.py`
   and `miles/utils/test_utils/session_verify_agent.py` do a bare
   `from enum import StrEnum` (3.11+). Guarded with
   `try: from enum import StrEnum / except ImportError: from backports.strenum import StrEnum`.
   This is a conftest-import prerequisite (without it the StrEnum-dependent tests fail
   at collection).
2. **Py3.10 `Exception.add_note` guard** — 3 sites (`sglang_engine.py:261`,
   `sample_utils.py:88`, `reloadable_process_group.py:282`) call `e.add_note(...)`
   (a 3.11 API, PEP 678). Guarded with `getattr(e, "add_note", lambda *a: None)(...)`
   so the original exception still propagates on 3.10. No control-flow change.
3. **`hypothesis>=5.40`** — the ROCm base ships hypothesis `5.35.1`, whose
   `HealthCheck` enum predates the `function_scoped_fixture` member (added in 5.40).
   `requirements.txt` listed an unpinned `hypothesis`, so `pip install -r` kept the
   stale base version. Pinned `hypothesis>=5.40` (+ `backports.strenum; python_version < "3.11"`).
4. **Rollout colocate GPU placement (ROCm)** — `_to_local_gpu_id` in
   `miles/backends/sglang_utils/sglang_engine.py` mis-maps ray-logical GPU ids when
   `HIP_VISIBLE_DEVICES` is a **non-0-based subset** (e.g. `"1,2"` to avoid a busy
   GPU 0): a logical id that numerically collides with a physical id in the visible
   list is routed through the `.index()` (physical) path, so both rollout engines
   collapse onto the same physical GPU (`base_gpu_id` → 0 for both) → KV-cache OOM.
   Fixed by preferring the **local-id interpretation first**. Root cause of the gsm8k
   `Server process terminated unexpectedly` / rollout-engine `Not enough memory`.
   **Same root cause** also hits `get_local_gpu_id` in `miles/ray/train_actor.py`
   (`cvd.split(",").index(str(ray.get_gpu_ids()[0]))` → `ValueError: '0' is not in list`
   when the train actor starts under `HIP_VISIBLE_DEVICES="1,2"`); fixed the same way
   (local-id first). Both call sites must be patched together.

## Fix branches
- `dev/amd-rocm-ci-0611` — merged MI300/MI350 ROCm Dockerfile + Py3.10 StrEnum/add_note fixes.
- `compat/py310-strenum-add-note` (PR #8) — standalone Py3.10 StrEnum + add_note compat.

## Error-type legend
- **A** `AttributeError: function_scoped_fixture` — hypothesis 5.35.1 (ROCm base)
  predates the 5.40 `HealthCheck` member; crashes at collection. Fixed by #3.
- **B** `AttributeError: '...' object has no attribute 'add_note'` — Py3.11
  `Exception.add_note` is absent on Py3.10. Fixed by #2.
- **C** rollout colocate placement — both sglang engines land on the same physical
  GPU (`base_gpu_id` collapses to 0) → `RuntimeError: Not enough memory`. Fixed by #4.
- **D** NV-only quant — `transformer_engine.pytorch.custom_recipes.quantization_nvfp4`
  (NVFP4) / `fake_int4_quant_cuda` (int4 CUDA kernel) are absent / NV-proprietary on
  ROCm. NVFP4 is an NV block format (MI350 hardware uses **MXFP4**, OCP microscaling).
  Keep `register_cuda_ci` so the AMD suite does not collect them.
- **E** loss-snapshot numeric tolerance — gfx942 recompute vs CUDA-saved `.pt`
  snapshot under a bitwise compare; a platform tolerance issue, not a bug.

---

## stage-a-cpu

| Bucket | Count | Notes |
|---|---|---|
| passed | 2615 | majority of CPU unit tests pass on gfx942 |
| ERROR (**A**) | 33 | `tests/fast/ray/rollout/...` + `test_train_data_conversion` + `test_mock_sglang_engine`: collection-time `function_scoped_fixture`. **Fixed by #3** (hypothesis upgraded to 6.x; `test_train_data_conversion` 26 passed, real_ray batch 37 collected). |
| FAILED (**B**) | 7 | `test_sample_utils` (4) + `test_openai_endpoint_utils` (3): `except ... as e: e.add_note(...)` masked the expected exception with `AttributeError`. **Fixed by #2** (those files: 7 failed → 28 passed). |
| FAILED (**E**) | 6 | `test_loss_snapshot` (grpo/sft variants): gfx942 vs CUDA `.pt` snapshot numeric diff. Platform tolerance — needs a tolerance policy or a gfx942 baseline. |

stage-b-cpu: **124 passed / 0 failed.**

---

## stage-b-2-gpu  (suite: `stage-b-2-gpu-h200`)

| # | Test path | Ran | Result | Notes |
|---|---|---|---|---|
| 1 | `tests/fast-gpu/test_run_megatron_worker_main.py` | yes | **PASS** | Generic `worker/main.py` unit test (MagicMock-stubbed megatron/sglang imports), no NV dependency. Currently `register_cuda_ci` but AMD-capable → should be re-registered `register_amd_ci`. |
| 2 | `tests/fast-gpu/test_nvfp4_quantizer.py` | yes | **EXCLUDE (D)** | Module-level `from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef` → `ModuleNotFoundError` on ROCm TE. ROCm does ship an equivalent `NVFP4QuantizerRef` under `transformer_engine.pytorch.experimental.quantization_microblock_ref`, but its output is **not bitwise-identical** to the NV reference (re-pointed import → 5 passed / 208 failed on the `rtol=0 atol=0` bitwise cases). NVFP4 is NV-proprietary; keep `register_cuda_ci`. |
| 3 | `tests/fast-gpu/test_quantizer_ci.py` | — | **EXCLUDE (D)** | `fake_int4_quant_cuda` is an int4 CUDA kernel, absent on ROCm. Keep `register_cuda_ci`. |
| 4 | `tests/fast-gpu/test_mxfp8_quantizer.py` | no | **DISABLED** | Registered `register_amd_ci(..., disabled="ROCm TE wheel has no MXFP8Quantizer")`. **The disabled reason is inaccurate**: ROCm TE *does* have `MXFP8Quantizer` (in `transformer_engine.pytorch.tensor.mxfp8_tensor`, signature `(fp8_dtype, *, rowwise, columnwise)`); it is just not re-exported from the top-level `transformer_engine.pytorch`. Re-pointing the import is likely enough to enable it (MXFP8 is an OCP standard AMD supports). |
| 5 | `tests/fast-gpu/test_semaphore.py` | no | **DISABLED** | `disabled="FIXME: re-enable after shared HTTP client concurrency is reset between cases."` Generic sglang server-concurrency test, no NV dependency; disabled due to a **scheduler-dependent / flaky assertion**, unrelated to arch. |

---

## stage-c-2-gpu  (suite: `stage-c-2-gpu-h200`, long)

| # | Test path | Ran | Result | Notes |
|---|---|---|---|---|
| 1 | `tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py` | yes | 2 GPU-placement bugs **FIXED & verified**; now reaches colocate weight-sync | 2-GPU colocate GRPO. Originally crashed at sglang rollout init (one engine `2.02 GB`, the other `114 GB` → `Not enough memory`): both rollout engines collapsed onto the **same physical GPU** (Fix #4). **After Fix #4** (both `_to_local_gpu_id` *and* `train_actor.get_local_gpu_id`): rollout engines split correctly (`base_gpu_id 0,1`, each 1GB + own 56GB KV, no OOM) **and** train-actor init passes (no more `ValueError: '0' is not in list`). Training then proceeds rollout → train-init → first `update_weights` (megatron→HF `Qwen2Bridge` conversion), where the ray job exits 1 **with no Python traceback** (last logs: Rank-0 `Converting to HuggingFace 4%`, Rank-1 `Reloading 6 process groups`, `[Gloo] Rank 0 connected to 0 peer ranks`). Deeper colocate weight-sync issue (same area as FINAL_REPORT `cannot pickle ReloadableProcessGroup` / Error **B**); needs separate investigation. |

> Note (placement): the same root cause (Fix #4) is the likely explanation for the
> FINAL_REPORT rollout-engine failures (`Request is disconnected from the client side`
> / `no_available_workers`) on the gsm8k short/async and ckpt e2e tests, which all
> stage through the same colocate rollout-engine startup.

---

## Notes on the broader GPU e2e set
The 4-/8-GPU e2e failures recorded earlier (`~/mi300_ci_results/FINAL_REPORT.md`)
group into ROCm feature-adaptation gaps that are **not** Dockerfile/merge regressions:
colocate `ReloadableProcessGroup` pickle, hipBLASLt training-GEMM algorithm
selection, GLM-MLA `Unsupported TP style 'mla_kv_a_proj'` on the sglang Transformers
backend, routing-replay `rollout_routed_experts`, session-verifier driver events, and
oversized-model rollout-engine stability. The merged ROCm Dockerfile itself builds
and smoke-tests cleanly (build / CLI / modelopt / NVTE all green).
