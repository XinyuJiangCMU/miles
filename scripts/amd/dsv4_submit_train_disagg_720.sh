#!/bin/bash
# DSv4-Flash FP8 DISAGGREGATED GRPO train — 6 node rocm720.
#   train (actor) = {4,6,8,9} = 32 GPU Megatron ;  rollout = {5,7} = 16 GPU sglang.
#   submit on head (node-4). Ray cluster must contain all 6 nodes {4,5,6,7,8,9}.
#
# Why disaggregated (vs the colocate submit):
#   - train nodes carry NO sglang pool  -> optimizer has full GPU + host RAM slack
#     (kills the 0.25-GPU-OOM / 0.75-host-OOM squeeze that crashed colocate at step 1).
#   - --update-weight-transfer-mode broadcast  -> RCCL, NOT hipIpcGetMemHandle
#     -> sidesteps the entire ROCm-7.2 IPC-export leak saga (legacy/hsapatch all moot).
#   - rollout on separate GPUs  -> can overlap train (async, phase 2 below).
#
# --rollout-num-nodes 2 with --num-nodes 6  ->  colocate=False, actor_num_nodes=4,
# rollout_num_gpus=16 (4 engines x TP4/EP4).  See run_deepseek_v4.py:76-116.
source /opt/shared/hai/dsv4_env.sh
export PYTHONUNBUFFERED=1
cd /root/miles && export PYTHONPATH=/root/miles
python scripts/amd/run_deepseek_v4.py train \
  --model-name DeepSeek-V4-Flash-FP8 \
  --hf-checkpoint /workspace/models/DeepSeek-V4-Flash-FP8 \
  --model-dir /workspace/models --model-local-dir /workspace/models \
  --data-dir /opt/shared/hai/datasets \
  --num-nodes 6 --num-gpus-per-node 8 --rollout-num-nodes 2 --skip-saving \
  --extra-args "--update-weight-transfer-mode broadcast --sglang-mem-fraction-static 0.85 --distributed-timeout-minutes 120" \
  > /workspace/train_disagg_6node_rocm720.log 2>&1
#
# NOTES on the extra-args (vs colocate):
#   - DROPPED --optimizer-offload-fraction  -> code auto-sets 0.75 for actor_num_nodes==4
#     (run_deepseek_v4.py:392-402, the value the working 30-step #1607 run used). With no sglang
#     on the train nodes this is safe on host RAM; can be lowered later for speed once it runs.
#   - DROPPED --offload-rollout-level kv_cache  -> colocate-only concept; disaggregated rollout
#     is always resident on its own nodes, nothing to pause.
#   - sglang mem-fraction 0.6 -> 0.85: rollout nodes are dedicated, give KV cache more room.
#
# OPEN before first launch: (1) verify --rollout-num-nodes is accepted (run ... train --help);
# (2) how Ray assigns physical nodes to rollout vs actor (want {5,7}=rollout) — NodeAffinity;
# (3) nodes {5,7} are dn's [1,3,5,7] — coordinate before borrowing; (4) broadcast weight-sync
# untested for DSv4-Flash on ROCm — validate step-0 grad_norm first.
#
# PHASE 2 (async overlap, AFTER disaggregated-sync gives healthy multi-step grad_norm):
#   add to extra-args:  --max-weight-staleness 1   (allow rollout to run 1 step ahead of train)
#   + the off-policy TIS importance-sampling correction (arguments.py:1185-1190) for correctness
#   under staleness. Tune staleness up for more overlap.
