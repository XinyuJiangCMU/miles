# Miles FSDP vs SGLang Alignment Lab

This workspace is a Miles-local experiment area for comparing a minimal FSDP-side
forward path against an SGLang server forward path under true-on-policy style
settings.

Goals:

- reuse Miles defaults for true-on-policy and batch-invariant ops
- keep experiment outputs isolated under a per-run directory
- compare tensor dumps with stable names and repeatable slicing rules

Primary entrypoints:

- `run_full_align.py`: one-click runner
- `capture_hf_fsdp_align.py`: FSDP-side tensor capture against an already-running server
- `compare_tensors.py`: dump-to-dump compare tool

Default example:

```bash
PYTHONPATH=/app/sglang/python:/app/true_on_policy/miles:$PYTHONPATH \
python /app/true_on_policy/miles/experiments/alignment_fsdp_sglang/run_full_align.py \
  --server-gpu 4 \
  --fsdp-gpu 5 \
  --host 127.0.0.1 \
  --port 30000 \
  --model-path Qwen/Qwen3-0.6B \
  --server-attention-backend triton \
  --fsdp-attn-implementation triton \
  --dtype bf16 \
  --batch-invariant \
  --true-on-policy \
  --deterministic \
  --max-new-tokens 1 \
  --output-root /app/results/miles_alignment_runs
```

Notes:

- First version focuses on a minimal single-process, single-sample, single-step run.
- The FSDP-side script wraps the HF model with Miles `apply_fsdp2(...)` under a
  single-rank process group.
- The FSDP-side Triton attention shim is vendored locally as
  `hf_triton_attention.py`, originally derived from the experiment workspace,
  so this lab no longer depends on `/app/true_on_policy/experiment/attention_test`.
- Tensor names intentionally mirror the existing `experiment/` alignment names to
  reduce compare friction.
