#!/bin/bash
# DSv4-Flash cluster lifecycle. Everything here runs OUTSIDE or BEFORE the miles python entry
# point, which is why it is not part of run_deepseek_v4.py.
#
#   container   start the training container      (run on the HOST, per node)
#   head        start the ray head                (run INSIDE the container, on the head node)
#   worker      join the ray head                 (run INSIDE the container, on every other node)
#   stop        kill the run and tear ray down    (run INSIDE the container, per node)
#
# Usage:
#   ssh amd-mi355x-1 "bash $PWD/dsv4_cluster.sh container"
#   ssh amd-mi355x-1 "docker exec dsv4-train bash /root/miles/scripts/amd/dsv4_cluster.sh head"
#   ssh amd-mi355x-3 "docker exec dsv4-train bash /root/miles/scripts/amd/dsv4_cluster.sh worker"
#
# Cluster/head selection comes from dsv4_env.sh (CLUSTER_MGMT_IPS / MASTER_ADDR); nothing is
# hardcoded per cluster here.
set -e
VERB=$1
if [ -z "$VERB" ]; then echo "usage: dsv4_cluster.sh {container|head|worker|stop}" >&2; exit 2; fi
HERE=$(dirname "$(readlink -f "$0")")
ENV_FILE=${ENV_FILE:-$HERE/dsv4_env.sh}

case "$VERB" in

container)
  # Host-side. RDMA-complete recipe (JOURNEY 4.3): all /dev/infiniband/* must be passed through,
  # or RCCL with NCCL_IB_HCA=ionic_* hangs ~40min instead of failing.
  IMG=${IMG:-xinyujiangcmu/miles:rocm720-mi35x-20260717}
  NAME=${NAME:-dsv4-train}
  docker rm -f "$NAME" 2>/dev/null || true
  docker run -d --name "$NAME" \
    --network host --ipc host --shm-size 128g \
    --device /dev/kfd --device /dev/dri \
    $(for d in /dev/infiniband/*; do printf -- '--device=%s ' "$d"; done) \
    --group-add video --cap-add CAP_SYS_PTRACE \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /opt/shared:/opt/shared -v /mnt/data/data/hai:/workspace \
    "$IMG" sleep infinity
  echo "container up on $(hostname); ibv check (want 9):"
  docker exec "$NAME" bash -lc 'ibv_devices 2>/dev/null | tail -n +3 | wc -l' || echo "ibv_devices failed"
  ;;

head)
  source "$ENV_FILE"
  ray stop --force >/dev/null 2>&1 || true
  sleep 3
  # --node-ip-address <mgmt IP> : announce on the mgmt net; if Ray picks a fabric /31 address the
  #                               raylet is judged dead. (bash forbids a comment after a line
  #                               continuation, hence this block.)
  # --dashboard-host 0.0.0.0    : the job-submit port must listen on all interfaces, not localhost
  ray start --head --node-ip-address "${MASTER_ADDR}" --port 6379 --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  echo "=== ray status ==="
  ray status
  ;;

worker)
  source "$ENV_FILE"
  # Detect this node's mgmt IP (visible inside --network host) and match it against the cluster
  # list, so one script works on every worker with nothing hardcoded.
  MYIP=$(ip -4 -o addr show "${NCCL_SOCKET_IFNAME}" 2>/dev/null | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | grep -Fx -f <(echo "$CLUSTER_MGMT_IPS" | tr ' ' '\n') | head -1)
  [ -z "$MYIP" ] && { echo "ERROR: no mgmt IP on ${NCCL_SOCKET_IFNAME} matches CLUSTER_MGMT_IPS=[$CLUSTER_MGMT_IPS]"; exit 1; }
  echo "worker mgmt IP = $MYIP  ->  head = ${MASTER_ADDR}:6379"
  ray stop --force >/dev/null 2>&1 || true
  sleep 3
  ray start --address="${MASTER_ADDR}:6379" --node-ip-address "${MYIP}" \
    --num-gpus 8 --disable-usage-stats
  ;;

stop)
  # The pkill patterns live in a script file on purpose: this process's cmdline is
  # "bash dsv4_cluster.sh stop", which does not contain run_deepseek_v4, so -f cannot match self.
  set +e
  pkill -9 -f run_deepseek_v4 2>/dev/null
  pkill -9 -f sglang 2>/dev/null
  ray stop --force 2>/dev/null
  pkill -9 ray 2>/dev/null
  sleep 2
  echo "[$(hostname)] stopped. residual python: $(pgrep -fc 'run_deepseek_v4|sglang::' 2>/dev/null || echo 0)"
  ;;

*)
  echo "unknown verb '$VERB'; want {container|head|worker|stop}" >&2
  exit 2
  ;;
esac
