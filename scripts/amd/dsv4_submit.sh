#!/bin/bash
# ============================================================================
# 统一 DSv4-Flash 4 节点 colocate GRPO 训练 submit 入口(一个脚本,参数化)。
# 在 head 容器内跑:  bash /root/miles/scripts/amd/dsv4_submit.sh [flags]
# head / 集群由旁边的 dsv4_env.sh 决定(CLUSTER_MGMT_IPS),这里不写死节点。
#
#   --mem-frac F   sglang KV 池 fraction         (默认 0.5)
#   --offload  F   optimizer-offload-fraction    (默认 0.6)
#   --seq-len  N   rollout-max-response-len      (默认 8192)
#   --wandb on|off 打不打 wandb                  (默认 on;需容器内 /root/.wandb_env)
#   --tag NAME     log 名 + wandb group          (默认按参数自动生成)
#   --mtp          开冻结 MTP(EAGLE 投机解码)   (默认关)
#   --extra "..."  额外透传给 --extra-args 的串   (默认空)
#
# 注:cuda-graph 默认就是开的(run_deepseek_v4.py:446 设 SGLANG_MEMORY_SAVER_CUDA_GRAPH=1),不用管。
# mem-frac 0.5 + offload 0.6 是 JOURNEY P1 在 16k 上验过 30 步的组合(三个旋钮是一套,别单独动)。
# 序列长度 2026-08-04 从 16k 降到 8k,内存旋钮沿用不变 —— 8k 更省,不会更紧。
# ============================================================================
MEM_FRAC=0.5; OFFLOAD=0.6; SEQLEN=8192; WANDB=on; TAG=""; EXTRA=""; MTP=""
while [[ $# -gt 0 ]]; do case "$1" in
  --mem-frac) MEM_FRAC=$2; shift 2;;
  --offload)  OFFLOAD=$2;  shift 2;;
  --seq-len)  SEQLEN=$2;   shift 2;;
  --wandb)    WANDB=$2;    shift 2;;
  --tag)      TAG=$2;      shift 2;;
  --mtp)      MTP="--enable-mtp"; shift 1;;
  --extra)    EXTRA=$2;    shift 2;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done
[ -z "$TAG" ] && TAG="mf${MEM_FRAC}_off${OFFLOAD}_$((SEQLEN/1024))k$([ "$WANDB" = on ] && echo _wb)"
LOG=/workspace/train_4node_${TAG}.log

source "$(dirname "$(readlink -f "$0")")/dsv4_env.sh"
if [ "$WANDB" = on ]; then
  if [ -f /root/.wandb_env ]; then source /root/.wandb_env
  else echo "[submit] WARN: --wandb on 但 /root/.wandb_env 缺失 → wandb 会 skip"; fi
fi
export PYTHONUNBUFFERED=1
cd /root/miles && export PYTHONPATH=/root/miles
echo "[submit] tag=$TAG mem-frac=$MEM_FRAC offload=$OFFLOAD seq=$SEQLEN wandb=$WANDB wandb_key=$([ -n "$WANDB_API_KEY" ] && echo set || echo unset) log=$LOG"

python scripts/amd/run_deepseek_v4.py train \
  --model-name DeepSeek-V4-Flash-FP8 \
  --hf-checkpoint /workspace/models/DeepSeek-V4-Flash-FP8 \
  --model-dir /workspace/models --model-local-dir /workspace/models \
  --data-dir /opt/shared/hai/datasets \
  --num-nodes 4 --num-gpus-per-node 8 --skip-saving $MTP --run-id "$TAG" \
  --extra-args "--sglang-mem-fraction-static $MEM_FRAC --distributed-timeout-minutes 120 --optimizer-offload-fraction $OFFLOAD --rollout-max-response-len $SEQLEN $EXTRA" \
  > "$LOG" 2>&1
