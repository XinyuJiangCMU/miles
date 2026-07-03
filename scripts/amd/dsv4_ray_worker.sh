#!/bin/bash
# Ray WORKER —— 每个 worker 节点(node-8/9)容器内跑。自动探测本机管理 IP,不硬编。
# 用法: ssh amd-mi355x-8 'docker exec dsv4-train bash /opt/shared/hai/dsv4_ray_worker.sh'
source /opt/shared/hai/dsv4_env.sh    # MASTER_ADDR(=head IP)、NCCL_SOCKET_IFNAME(管理网卡)、RoCE env

# 探测本机在管理网卡上的 IP(--network host 下容器能看到宿主机接口)
MYIP=$(ip -4 -o addr show "${NCCL_SOCKET_IFNAME}" 2>/dev/null | grep -oE '172\.30\.160\.[0-9]+' | head -1)
[ -z "$MYIP" ] && { echo "ERROR: 探测不到本机管理 IP (${NCCL_SOCKET_IFNAME})"; exit 1; }
echo "worker mgmt IP = $MYIP  ->  head = ${MASTER_ADDR}:6379"
ray stop --force >/dev/null 2>&1 || true
sleep 3

# ray start 参数:
#   --address <head管理IP>:6379 : 连 head 的 GCS 协调口
#   --node-ip-address <本机管理IP> : 用管理网IP亮身份,别让Ray猜到fabric(否则raylet判死)
ray start --address="${MASTER_ADDR}:6379" --node-ip-address "${MYIP}" \
  --num-gpus 8 --disable-usage-stats
