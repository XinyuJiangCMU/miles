#!/bin/bash
# DSv4-Flash FP8 colocate GRPO train — 4 node {4,6,8,9} rocm720, submit on head(node-4).
source /opt/shared/hai/dsv4_env.sh
export PYTHONUNBUFFERED=1
cd /root/miles && export PYTHONPATH=/root/miles
python scripts/amd/run_deepseek_v4.py train \
  --model-name DeepSeek-V4-Flash-FP8 \
  --hf-checkpoint /workspace/models/DeepSeek-V4-Flash-FP8 \
  --model-dir /workspace/models --model-local-dir /workspace/models \
  --data-dir /opt/shared/hai/datasets \
  --num-nodes 4 --num-gpus-per-node 8 --skip-saving \
  --extra-args "--sglang-mem-fraction-static 0.6 --distributed-timeout-minutes 120 --optimizer-offload-fraction 0.75 --offload-rollout-level kv_cache" \
  > /workspace/train_4node_rocm720.log 2>&1
# NOTE: --optimizer-offload-fraction 0.75 == run_deepseek_v4.py:395 code default for 4 actor nodes
# (75% of the DP=1 full fp32 Adam state offloaded to CPU, keep ~25% on GPU -> ~16GB GPU headroom).
# 0.25 was a WRONG override (2026-07-14): it pinned 75% of the optimizer on GPU -> filled GPU during
# step-0 optimizer.step -> starved step-1 update_weights -> 60x slow + OOM abort. If 0.75 hits host
# RAM OOM (~1509/1511 on rocm720), drop to 0.6 (fits both host and GPU; see JOURNEY TL;DR 2026-07-14).
