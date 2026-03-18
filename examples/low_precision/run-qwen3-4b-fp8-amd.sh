#!/bin/bash
# Qwen3-4B FP8 training on AMD MI300X
# Demonstrates FP8 quantized model training with FSDP
#
# Prerequisites:
#   1. Convert model to FP8: python tools/convert_hf_to_fp8.py --model-dir /data/Qwen3-4B --save-dir /data/Qwen3-4B-FP8 --strategy tensor
#   2. Set HIP_VISIBLE_DEVICES to target GPUs
#
# Usage:
#   HIP_VISIBLE_DEVICES=0,1 bash examples/low_precision/run-qwen3-4b-fp8-amd.sh

pkill -9 sglang 2>/dev/null; sleep 2
ray stop --force 2>/dev/null; pkill -9 ray 2>/dev/null; sleep 2

set -ex

# AMD Performance Tuning
export HIP_FORCE_DEV_KERNARG=1
export NCCL_BUFFSIZE=16777216
export HSA_NO_SCRATCH_RECLAIM=1
export GPU_MAX_HW_QUEUES=2
export SGLANG_USE_AITER=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"

# Use FP8 checkpoint
MODEL_DIR="${MODEL_DIR:-/data}"
HF_CHECKPOINT="${MODEL_DIR}/Qwen3-4B-FP8"
if [ ! -d "$HF_CHECKPOINT" ]; then
    echo "FP8 model not found at $HF_CHECKPOINT"
    echo "Convert first: python tools/convert_hf_to_fp8.py --model-dir ${MODEL_DIR}/Qwen3-4B --save-dir $HF_CHECKPOINT --strategy tensor"
    exit 1
fi

DATA_DIR="${DATA_DIR:-/data}"
DATA_FILE="${DATA_DIR}/test_math.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    DATA_FILE="$(dirname "$SCRIPT_DIR")/../scripts/sample_data/test_math.jsonl"
fi

CKPT_ARGS=(--hf-checkpoint ${HF_CHECKPOINT})

ROLLOUT_ARGS=(
    --prompt-data ${DATA_FILE}
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout 10
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --rollout-max-response-len 512
    --rollout-temperature 1
    --global-batch-size 8
    --micro-batch-size 2
    --balance-data
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
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
    --sglang-mem-fraction-static 0.3
    --sglang-disable-custom-all-reduce
    --sglang-cuda-graph-max-bs 256
)

FSDP_ARGS=(
    --update-weight-buffer-size 1073741824
    --no-offload-train
)

# Launch
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats

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
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${FSDP_ARGS[@]}
