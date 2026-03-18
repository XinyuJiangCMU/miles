#!/bin/bash
# Qwen2.5-7B-Instruct FSDP training on AMD MI300X
set -e

# AMD Performance Tuning
export HIP_FORCE_DEV_KERNARG=1
export NCCL_BUFFSIZE=16777216  # 16MB for MI300X HBM3
export HSA_NO_SCRATCH_RECLAIM=1
export GPU_MAX_HW_QUEUES=2
export SGLANG_USE_AITER=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export PYTHONBUFFERED=16

# Model args - auto-detected from HF config
MODEL_DIR="${MODEL_DIR:-/data}"
DATA_DIR="${DATA_DIR:-/data}"
HF_CHECKPOINT="${MODEL_DIR}/Qwen2.5-7B-Instruct"

MODEL_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/test_math.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 60
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
)

GRPO_ARGS=(
   --algorithm grpo
   --global-batch-size 2
   --micro-batch-size 2
   --seq-length 1024
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.25
   --sglang-disable-custom-all-reduce
   --sglang-cuda-graph-max-bs 256
)

FSDP_ARGS=(
   --update-weight-buffer-size 1073741824
   --gradient-checkpointing
   --no-offload-train
)

# Launch ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"SGLANG_MEMORY_SAVER_CUDA_GRAPH\": \"true\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   --no-offload-train \
   --gradient-checkpointing \
   ${MODEL_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${FSDP_ARGS[@]}
