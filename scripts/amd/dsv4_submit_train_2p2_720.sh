#!/bin/bash
# DSv4-Flash 2+2 DISAGGREGATED async GRPO train -- 4 node rocm720 (lmsys blog recipe adapted).
#   train (actor) = 2 nodes = 16 GPU Megatron, TP1/PP4/EP4 (DP=4 shards optimizer -> fits 16 GPU).
#   rollout       = 2 nodes = 16 GPU sglang (4 engines x TP4/EP4).
#   submit on head (node-4). Ray cluster must contain all 4 nodes {4,6,8,9}.
#
# WHY this works on 4 nodes where TP8/PP4/EP8 could not:
#   - TP8/PP4/EP8/DP1 welds the model to all 32 GPU (each GPU carries full Adam, no split possible).
#   - TP1/PP4/EP4/DP4: EP4+PP4 shard the MoE across 16 GPU, and DP=4 shards the optimizer 4-way
#     -> 291B training fits on 16 GPU (2 nodes), freeing 2 nodes for a dedicated rollout pool
#     -> spatial disaggregation -> async overlap becomes possible (the blog proved 100+ steps).
#   - broadcast weight-sync (RCCL, not hipIpcGetMemHandle) sidesteps the ROCm-7.2 IPC leak saga.
#
# PHASE 1 (this script): disaggregated-SYNC first -- validate TP1/PP4/EP4 16-GPU training comes up,
#   broadcast weight-transfer works on ROCm, step-0 grad_norm healthy. NO overlap yet (train.py).
#   Keeping FP8 actor (fp8_training default True) -- we already run FP8 on TP8; blog used bf16 only
#   because they hadn't validated FP8 actor. FP8 is LIGHTER than bf16, so 16-GPU fit is easier.
# PHASE 2 (after Phase 1 healthy): wire train_async.py entrypoint + --use-tis + --max-weight-staleness 1
#   for the actual rollout/train overlap (~35% step cut).
source /opt/shared/hai/dsv4_env.sh
export PYTHONUNBUFFERED=1
cd /root/miles && export PYTHONPATH=/root/miles
python scripts/amd/run_deepseek_v4.py train \
  --model-name DeepSeek-V4-Flash-FP8 \
  --hf-checkpoint /workspace/models/DeepSeek-V4-Flash-FP8 \
  --model-dir /workspace/models --model-local-dir /workspace/models \
  --data-dir /opt/shared/hai/datasets \
  --num-nodes 4 --num-gpus-per-node 8 --rollout-num-nodes 2 --skip-saving \
  --extra-args "--update-weight-transfer-mode disk-delta --update-weight-disk-dir /workspace/weight_delta_2p2 --update-weight-local-checkpoint-dir /tmp/dsv4_weight_local --sglang-mem-fraction-static 0.85 --optimizer-offload-fraction 0.25 --distributed-timeout-minutes 120" \
  > /workspace/train_2p2_4node_rocm720.log 2>&1
#
# TUNING NOTES:
#   - offload-fraction squeeze (measured step-0 optimizer-init on TP2/PP4/EP4/DP2, 16 GPU):
#       0.5 -> train-node HOST OOM (~880GB/node optimizer fp32 + disk-delta snapshot approaches 1.5TB)
#       0.0 -> GPU OOM in fused_adam _initialize_state (GPU 282.6/288 GiB, short by <1GiB)
#       0.25 -> sweet spot: frees several GiB/GPU, host uses ~440GB/node + snapshot, safe under 1.5TB.
#     Compute path already proven end-to-end at 0.0: log_probs 680s + actor_train fwd+bwd 870s OK.
#   - sglang mem-fraction 0.85: rollout nodes dedicated, give KV cache room.
#   - cuda-graph is ON by default (our R7 deploy) -> rollout nodes get the 2x directly.
#   - OPEN: (1) TP1 FP8 blockwise GEMM shapes valid? (2) ray role assignment train vs rollout;
#     (3) broadcast weight-sync DSv4-Flash ROCm untested -> watch step-0 grad_norm.
