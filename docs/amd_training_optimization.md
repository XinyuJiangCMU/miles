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
