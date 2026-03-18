# AMD MI300X Training Optimization Guide

## Quick Start

```bash
# Set GPU devices
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training (auto-detects AMD and applies optimizations)
python train.py --hf-checkpoint /data/Qwen3-4B \
    --no-offload-train \
    --gradient-checkpointing \
    --micro-batch-size 2
```

## Performance Optimizations (Cumulative)

Benchmarked on Qwen3-4B, 1x MI300X:

| Optimization | tok/s | vs Baseline |
|---|---|---|
| Baseline (no opts) | 1,423 | 1.0x |
| + Fused AdamW | 1,484 | 1.04x |
| + reshard=False (DP=1) | 1,719 | 1.21x |
| + BF16 reduced precision | 1,727 | 1.21x |
| + Micro-batch-size 2 | 3,458 | 2.43x |
| + Micro-batch-size 4 | 5,929 | **4.17x** |

### 1. Fused AdamW (automatic)
Enabled by default in FSDP actor. Combines per-parameter optimizer operations into fused kernels. ~10% speedup.

### 2. reshard_after_forward=False (automatic for DP=1)
When only 1 data-parallel rank, skip unnecessary resharding after forward. Saves ~42% by avoiding redundant all-gather in backward.

### 3. BF16 Reduced Precision Reduction (automatic)
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True` for faster GEMM operations.

### 4. Gradient Checkpointing (`--gradient-checkpointing`)
Saves 47% GPU memory (163GB → 86GB at BS=32). Enables larger batch sizes or more KV cache for rollout.

### 5. No Offload Train (`--no-offload-train`)
MI300X has 192GB VRAM - keep model + optimizer on GPU. Eliminates CPU↔GPU transfer overhead. **Saves ~48% step time** (12.4s → 6.5s).

### 6. Micro-Batch Size (`--micro-batch-size 2` or `4`)
Larger micro-batches are more efficient. MBS=2 gives 56% more tok/s than MBS=1 with minimal memory increase.

### 7. RCCL Tuning (automatic via execute_train)
- `NCCL_BUFFSIZE=16MB`: 20% faster all-reduce
- `HIP_FORCE_DEV_KERNARG=1`: Reduce kernel launch overhead
- `HSA_NO_SCRATCH_RECLAIM=1`: Prevent scratch memory reclaim stalls

### 8. GEMM Pre-tuning (optional, +12%)
```bash
bash scripts/pretune_gemm_amd.sh /data/Qwen3-4B
export PYTORCH_TUNABLEOP_ENABLED=1
```

### 9. BF16 Gradient Reduce (`--bf16-reduce`)
Use BF16 for gradient all-reduce communication. Halves bandwidth. Acceptable for RL training.

## Memory Guide

| Model | FSDP DP=1 | + Gradient Ckpt | + Ref Model | Available for KV |
|---|---|---|---|---|
| Qwen3-4B | 40GB | 52GB | 60GB | 132GB |
| Qwen2.5-7B | 65GB | 55GB | 70GB | 122GB |
| Qwen3-30B-MoE | ~120GB | ~80GB | ~90GB | 102GB |

## Training Scripts

- `scripts/run-qwen3-4B-fp8-amd.sh` - FP8 training with optimizations
- `scripts/run-qwen3-4B-amd.sh` - BF16 training
- `scripts/run-qwen25-7B-amd.sh` - 7B model training
- `scripts/pretune_gemm_amd.sh` - GEMM kernel pre-tuning

## Inference Performance (SGLang on MI300X)

| Model | BF16 | FP8 | Improvement |
|---|---|---|---|
| Qwen2.5-7B (2x MI300X) | 1,382 tok/s | 1,703 tok/s | +23% |
| Qwen3-30B-MoE FP8 (2x MI300X) | N/A | 5,344 tok/s | - |

## Multi-GPU FSDP Tuning (DP>1)

For multi-GPU data-parallel training on MI300X:

```bash
# RCCL tuning for multi-GPU
export NCCL_BUFFSIZE=16777216  # 16MB buffer
export NCCL_MIN_NCHANNELS=112  # More channels for XGMI
export NCCL_ALGO=Ring           # Ring is faster than Tree for 2-8 GPUs

# FSDP settings
--bf16-reduce    # Halves gradient communication volume
# Note: reshard_after_forward stays True for DP>1 (memory savings)

# Memory considerations for colocate mode:
# Actor model: ~8GB (Qwen3-4B in BF16)
# Ref model: ~8GB (kept on GPU with --no-offload-train)
# Optimizer: ~16GB (AdamW with momentum)
# Activations: ~10-50GB (depends on batch size and GC)
# SGLang inference: ~30-80GB (KV cache + CUDA graphs)
# Total: ~72-162GB / 192GB available
```

## Known Limitations

1. **FP8 forward training**: Not beneficial for dense models <30B (quantization overhead > GEMM savings)
2. **Custom all-reduce**: AITER AR is slower than NCCL on MI300X. Keep disabled.
3. **TunableOp**: +12% GEMM speedup but requires 10-30 min pre-tuning per model
4. **Gradient checkpointing at DP=1**: Costs 24% speed for only 3% memory savings. Skip for DP=1.
5. **FP8 KV cache**: Not supported (AITER mha_batch_prefill doesn't support FP8 input)

## Troubleshooting

### Common Issues

**1. "normalize_e4m3fn_to_e4m3fnuz: assertion error"**
```
AssertionError: weight.dtype == torch.float8_e4m3fn
```
Fix: Update to latest fp8-miles-amd-dev branch. We handle both e4m3fn and e4m3fnuz.

**2. "hipIpcOpenMemHandle failed"**
```
RuntimeError: hipIpcOpenMemHandle failed
```
Fix: Add `--sglang-disable-custom-all-reduce` to training args. Custom all-reduce has issues with non-contiguous GPU IDs.

**3. OOM during CUDA graph capture**
```
RuntimeError: HIP out of memory during graph capture
```
Fix: Add `--cuda-graph-max-bs 256` to SGLang server args, or `--disable-cuda-graph` for MoE models.

**4. "fused_set_kv_buffer_arg" error**
```
ValueError: fused_set_kv_buffer requires CUDA sgl_kernel op
```
Fix: This is expected on AMD. The `utils.py _is_cuda` check disables fused KV buffer on AMD.

**5. Slow training with --offload-train**
```
Step time: 12+ seconds
```
Fix: Use `--no-offload-train` on MI300X (192GB VRAM is sufficient for most models).

**6. "pynvml not available"**
```
Warning: pynvml not available, skipping NUMA affinity setup
```
This is informational only. On AMD, NUMA affinity is set via ROCm sysfs instead.

### Performance Checklist

- [ ] `HIP_VISIBLE_DEVICES` is set correctly
- [ ] `SGLANG_USE_AITER=1` for AITER attention backend
- [ ] `NCCL_BUFFSIZE=16777216` for 16MB RCCL buffer
- [ ] `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
- [ ] `--no-offload-train` for MI300X 192GB
- [ ] `--micro-batch-size 2` or higher
- [ ] `--sglang-disable-custom-all-reduce` to avoid hipIpc issues
- [ ] Flash Attention 2 is being used (`attn_implementation="flash_attention_2"`)

## Multi-GPU Benchmark Results (Qwen3-4B, 2x MI300X DP=2)

| MBS/GPU | Total tok/s | vs 1-GPU | Scaling |
|---|---|---|---|
| 4 | 9,762 | 1.16x | 58% |
| 8 | 14,766 | 1.75x | 88% |
| **16** | **20,162** | **2.39x** | **120%** |

## Multi-GPU Benchmark Results (Qwen2.5-7B, 2x MI300X DP=2)

| MBS/GPU | Total tok/s | vs 1-GPU | Scaling |
|---|---|---|---|
| 2 | 6,360 | 1.12x | 56% |
| 4 | 10,703 | 1.88x | 94% |
| **8** | **14,516** | **2.56x** | **128%** |

Note: Super-linear scaling at large batch sizes is due to each GPU
getting half the model parameters (more efficient GEMM with smaller
per-rank weight matrices) + larger per-GPU batch = better HBM bandwidth
utilization.

## RL Training Pipeline Breakdown (Qwen3-4B, 1x MI300X)

| Phase | Time | % of Step |
|---|---|---|
| Ref model forward (no_grad) | 58ms | 16% |
| Actor model forward (no_grad) | 74ms | 20% |
| Training (fwd+bwd+opt) | 232ms | 64% |
| **Total per RL step** | **364ms** | **100%** |
| **Throughput** | **5,626 tok/s** | |

Weight update: ~2s (dominated by FSDP DTensor materialization).
This happens once per training step, not per microbatch.

## When to Use Each Optimization

| Optimization | Use When | Don't Use When |
|---|---|---|
| --no-offload-train | MI300X 192GB VRAM | Small GPU (<64GB) |
| --gradient-checkpointing | Colocate mode (need room for SGLang) | DP=1 standalone training |
| --micro-batch-size 4+ | Single GPU training | Already memory-constrained |
| --bf16-reduce | DP>1 with bandwidth constraints | Accuracy-sensitive tasks |
| --compile-log-probs | Many rollouts (60+) | Quick tests (<10 rollouts) |

## Complete Scaling Results (Session 6)

### Qwen3-4B on MI300X (tok/s)

| GPUs | MBS=4 | MBS=8 | MBS=16 | MBS=32 |
|---|---|---|---|---|
| 1 | 7,923 | 10,078 | 11,607 | 12,726 |
| 2 | 9,807 | 14,719 | 20,024 | 23,508 |
| 4 | 26,259 | 37,337 | 44,218 | 48,889 |
| **8** | **57,654** | **76,391** | **88,612** | **96,430** |

### Qwen2.5-7B on MI300X (tok/s)

| GPUs | MBS=4 | MBS=8 | MBS=16 | MBS=32 |
|---|---|---|---|---|
| 2 | 6,344 | 10,737 | 14,522 | 16,925 |
| 4 | 16,322 | 26,731 | 33,116 | 36,081 |
| **8** | **43,460** | **58,242** | **66,258** | **71,210** |

### Full RL Training (colocate mode)

| GPUs | step_time | train_time | update_wt | wait_time |
|---|---|---|---|---|
| 2 | 5.95s | 1.39s | 0.37s | 4.55s |
| 4 | 4.28s | 0.99s | 0.28s | 3.28s |

### Megatron Backend RL Training (2x MI300X TP=2)

| Metric | Value |
|---|---|
| step_time | 4.99s |
| actor_train | 27.4 TFLOP/s |
| log_probs | 22.7 TFLOP/s |
| update_weights | 0.23s |
| wait_time | 3.65s |

Note: Requires `get_device_arch_version` patch for AMD/Ray compatibility.
Use `--load` with pre-converted Megatron checkpoint.

### Non-Colocate Mode (4 train + 4 inference GPUs)

| Metric | Value |
|---|---|
| step_time | 3.62s |
| train_time | 0.92s |
| update_weights | 0.70s |
| wait_time | 2.70s |
| prefix_cache_hit | 94.5% |

Non-colocate is 15% faster than colocate mode because there's no
memory saver pause/resume overhead.

### Async Training (train_async.py, non-colocate)

| Metric | Sync | Async | Improvement |
|---|---|---|---|
| step_time | 3.62s | **2.63s** | **-27%** |
| wait_time | 2.70s | 1.69s | -37% |
| train_time | 0.92s | 0.94s | same |

Async mode overlaps next rollout with current training,
hiding most of the rollout latency.

## Recommended RL Training Configurations

### Small Model (4B) on 2 GPUs
```bash
# Colocate mode (shared GPUs)
HIP_VISIBLE_DEVICES=0,1 bash scripts/run-qwen3-4B-fp8-amd.sh
# Expected: ~5.95s/step
```

### Small Model (4B) on 8 GPUs
```bash
# Non-colocate async (4 train + 4 inference)
# Use train_async.py for best performance
# Expected: ~2.63s/step
```

### Medium Model (7B) on 2 GPUs
```bash
# Colocate mode
HIP_VISIBLE_DEVICES=0,1 bash scripts/run-qwen25-7B-amd.sh
# Expected: ~6.78s/step
```

### Megatron Backend (4B, TP=2)
```bash
# 1. Convert model first
HIP_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 tools/convert_hf_to_torch_dist.py ...
# 2. Then train with --load
HIP_VISIBLE_DEVICES=0,1 bash scripts/run-qwen3-4B-megatron-amd.sh
# Expected: ~4.99s/step
```
