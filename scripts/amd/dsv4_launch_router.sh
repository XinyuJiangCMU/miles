#!/bin/bash
# 4-node full DSv4-Flash-FP8 train on ROUTER image (sgl-model-gateway), --use-miles-router dropped
# (recipe edited: if is_hip(): pass). Tests whether the new router preserves routed_experts for R3.
source /opt/shared/hai/dsv4_env.sh
export PYTHONUNBUFFERED=1
cd /root/miles
export PYTHONPATH=/root/miles
python scripts/amd/run_deepseek_v4.py train \
  --model-name DeepSeek-V4-Flash-FP8 \
  --hf-checkpoint /workspace/models/DeepSeek-V4-Flash-FP8 \
  --model-dir /workspace/models --model-local-dir /workspace/models \
  --data-dir /opt/shared/hai/datasets \
  --num-nodes 4 --num-gpus-per-node 8 --skip-saving \
  --extra-args "--sglang-mem-fraction-static 0.6 --distributed-timeout-minutes 120" \
  > /workspace/train_4node_router.log 2>&1
