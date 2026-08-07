#!/bin/bash
# ============================================================================
# 统一 DSv4-Flash 4 节点 colocate GRPO 训练 submit 入口(一个脚本,参数化)。
# 在 head 容器内跑:  bash /root/miles/scripts/amd/dsv4_submit.sh [flags]
# head / 集群由旁边的 dsv4_env.sh 决定(CLUSTER_MGMT_IPS),这里不写死节点。
#
#   --mem-frac F   sglang KV 池 fraction         (默认 colocate 0.5 / disagg 0.85)
#   --offload  F   optimizer-offload-fraction    (默认 colocate 0.6 / disagg 0.25)
#   --seq-len  N   rollout-max-response-len      (默认 8192)
#   --wandb on|off 打不打 wandb                  (默认 on;需容器内 /root/.wandb_env)
#   --tag NAME     log 名 + wandb group          (默认按参数自动生成)
#   --mtp          开冻结 MTP(EAGLE 投机解码)   (默认关)
#   --rollout-nodes N  改跑 disaggregated:N 个节点专跑 rollout,其余跑训练(默认 0 = colocate)
#   --extra "..."  额外透传给 --extra-args 的串   (默认空)
#
# colocate 下 mem-frac 0.5 + offload 0.6 是一对:0.6 躲 host 墙,0.5 躲 step-2 resume 的 GPU 墙。
# 别单独动一个。disagg 下这对耦合消失(两边不共卡),默认换成 0.85 / 0.25 —— 0.25 是 2+2 上
# 实测的甜点(0.5 host OOM、0.0 差不到 1GiB GPU OOM)。
# ============================================================================
MEM_FRAC=""; OFFLOAD=""; SEQLEN=8192; WANDB=on; TAG=""; EXTRA=""; MTP=""; ROLLOUT_NODES=0
while [[ $# -gt 0 ]]; do case "$1" in
  --mem-frac) MEM_FRAC=$2; shift 2;;
  --offload)  OFFLOAD=$2;  shift 2;;
  --seq-len)  SEQLEN=$2;   shift 2;;
  --wandb)    WANDB=$2;    shift 2;;
  --tag)      TAG=$2;      shift 2;;
  --mtp)      MTP="--enable-mtp"; shift 1;;
  --rollout-nodes) ROLLOUT_NODES=$2; shift 2;;
  --extra)    EXTRA=$2;    shift 2;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done
if [ "$ROLLOUT_NODES" -gt 0 ]; then
  DISAGG="--rollout-num-nodes $ROLLOUT_NODES"
  : "${MEM_FRAC:=0.85}"; : "${OFFLOAD:=0.25}"
else
  DISAGG=""
  : "${MEM_FRAC:=0.5}";  : "${OFFLOAD:=0.6}"
fi
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

# NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME are dropped here on purpose. execute_train copies
# whichever of them is set on the submitting host into the ray runtime env, which then overrides
# every worker. The mgmt NIC is not named the same on every node (enp81s0f1np1 on the XAI boxes,
# enx00e04c680080 on des2-2), so one broadcast value makes gloo fail with "Unable to find address"
# on the odd one out. Each raylet already sourced dsv4_env.sh and holds its own correct value, and
# workers inherit it as long as the runtime env does not overwrite it.
env -u NCCL_SOCKET_IFNAME -u GLOO_SOCKET_IFNAME \
python scripts/amd/run_deepseek_v4.py train \
  --model-name DeepSeek-V4-Flash-FP8 \
  --hf-checkpoint /workspace/models/DeepSeek-V4-Flash-FP8 \
  --model-dir /workspace/models --model-local-dir /workspace/models \
  --data-dir /opt/shared/hai/datasets \
  --num-nodes 4 --num-gpus-per-node 8 --skip-saving $MTP $DISAGG --run-id "$TAG" \
  --extra-args "--sglang-mem-fraction-static $MEM_FRAC --distributed-timeout-minutes 120 --optimizer-offload-fraction $OFFLOAD --rollout-max-response-len $SEQLEN $EXTRA" \
  > "$LOG" 2>&1
