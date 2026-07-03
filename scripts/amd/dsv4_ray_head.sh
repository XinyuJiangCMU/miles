#!/bin/bash
# Ray HEAD —— head 节点(node-6)容器内跑。
# 用法: ssh amd-mi355x-6 'docker exec dsv4-train bash /opt/shared/hai/dsv4_ray_head.sh'
source /opt/shared/hai/dsv4_env.sh    # MASTER_ADDR(=head管理IP)、RoCE env、通道数、Ray心跳
ray stop --force >/dev/null 2>&1 || true
sleep 3

# ray start 参数(注:bash 行尾 \ 后不能跟注释,故说明写这):
#   --node-ip-address <管理IP> : 用管理网IP亮身份,别让Ray猜到fabric /31(否则 raylet 判死)
#   --port 6379                : GCS 集群协调口,worker 连这
#   --dashboard-host 0.0.0.0   : job 提交口开在所有网卡(不只localhost),提交才连得上
#   --dashboard-port 8265      : ray job submit 用的端口
ray start --head --node-ip-address "${MASTER_ADDR}" --port 6379 --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "=== ray status ==="
ray status
