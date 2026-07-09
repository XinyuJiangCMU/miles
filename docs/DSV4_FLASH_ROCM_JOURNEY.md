# 在 AMD MI355X 上把 DeepSeek-V4-Flash 的 FP8 强化学习训练跑起来

一次工程记录:从"让模型在 AMD 卡上能加载",到 4 台机器 32 张卡把 291B 的 DeepSeek-V4-Flash 全量 FP8 RL 训练稳定跑起来。

**结果**:4 节点 × 8 卡 = 32 GPU(AMD MI355X / gfx950),全量 DeepSeek-V4-Flash(291B、43 层 MoE、DSA 稀疏注意力),FP8 训练 + FP8 rollout 的 GRPO 强化学习,**已连续稳定跑过 17 步、零崩溃(手动停止,优化器保持 fp32,reward 稳定 ~0.5、grad_norm 健康)**,wandb 全程追踪。

---

## 用到的分支/组件

| 仓库 | 分支 | 角色 |
|---|---|---|
| TransformerEngine (Jessica) | [`JessicaJiang-123/TransformerEngine@amd-qwen3-30b-a3b-fp8-dev`](https://github.com/JessicaJiang-123/TransformerEngine/tree/amd-qwen3-30b-a3b-fp8-dev) | **ROCm 上的 FP8 blockwise GEMM**(fp8 训练的地基) |
| miles | [`amd/dsv4-flash-rocm-enablement-20260701`](https://github.com/XinyuJiangCMU/miles/tree/amd/dsv4-flash-rocm-enablement-20260701) | 让 DSv4 在 AMD 卡上能加载/使能 |
| Megatron-LM | [`amd/dsv4-swiglu-fused-clamp-20260630`](https://github.com/XinyuJiangCMU/Megatron-LM/tree/amd/dsv4-swiglu-fused-clamp-20260630) | 训练侧激活 kernel |
| miles | [`wip/dsv4-flash-full-train-20260701`](https://github.com/XinyuJiangCMU/miles/tree/wip/dsv4-flash-full-train-20260701) | 主训练分支:多机并行 + 显存 |
| sglang | [`amd/dsv4-flash-rocm-rollout-20260703`](https://github.com/XinyuJiangCMU/sglang/tree/amd/dsv4-flash-rocm-rollout-20260703) | rollout(推理)侧 ROCm 修复 |

下面按时间顺序讲。

---

## 一、先让模型在 AMD 卡上"能加载"

**初始权重**:[`sgl-project/DeepSeek-V4-Flash-FP8`](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8)(deepseek-ai/[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 的公开 FP8 repackage,291B / 43 层)。磁盘占用:**fp8 safetensors ~297GB(46 shards)**;转成训练用的 torch_dist 格式后 **~530GB**(fp32 优化器态展开,更大)。放各节点本地 NVMe。


DeepSeek-V4 用了一套 CUDA-only 的 kernel 库(deepseek-ai/TileKernels,只支持 NVIDIA SM90/SM100)。在 AMD gfx950 上一 import 就让整个模型定义加载失败,连"把 FP8 权重转成训练用的 torch_dist 格式"这一步都过不了。

- 修法:把这些 CUDA-only 的 import 用 try/except 包起来,替换成一个"你要真调用我才报错"的占位实现(commit `1577f3b21`)。这样加载模型时不碰它、只有真正 forward 用到才抛错——而 DSv4-Flash 的路径根本不走它。
- 在这个基线上还叠了一批让 DSv4-Flash 在 AMD 上跑顺的使能改动:MoE permute 融合(`6fc0600`)、把 q-RMSNorm 和交叉熵 log-prob 换成 Liger 的 triton kernel(`f13349f`、`688f2dd`)、MoE 的 SwiGLU 融合(`c356fc1`)、以及 R3 路由重放走 miles 自己的 python router(`b9047e3`)。

**这一步用 4-layer 版验证**:先不上 291B,用一个 4 层的小版本(单机 8 卡)把"能加载 → FP8 转 torch_dist → 能跑一个训练 step"整条链路走通,确认没问题,再上全量。后面所有全量训练都长在这个基线上。

## 二、训练侧的一个激活 kernel

DeepSeek-V4 的 MoE 用的是带 clamp 的 SwiGLU 激活(silu,截断到 10,再乘 per-token 的 routing 概率)。Megatron 原本这条路走的是慢的、逐算子的 autograd 实现。

- 修法(commit `262b82419`):给融合版的 `weighted_bias_swiglu` 加上 clamp+offset,让它走 `torch.compile` → ROCm inductor 的融合 triton kernel(前向+反向都融),而不是慢路径。纯增量、默认行为不变,NVIDIA 侧不受影响。验证:fp32 数值误差 7e-7、4 层端到端一个 step 76 秒、无 NaN。
- 训练命令里对应开关:`--no-bias-swiglu-fusion --activation-func-clamp-value 10`。

## 三、集群与网络

**机器**:4 台 MI355X 节点,每台 8 卡(每卡 288GB HBM),每台 1.5TB 主机内存 → **全集群共 6TB 主机内存**。这个 6TB 后面很关键。

**两张物理网络,各干各的**:
- 一张**管理网**(普通以太网):跑 ray 的控制/心跳。
- 一张**RDMA 高速网**(RoCE):跑集合通信(梯度/权重)的大流量。
  - 小注:AMD 上这套集合通信库其实叫 **RCCL**(ROCm 版的 NCCL,API/环境变量都兼容 `NCCL_*`),下文为方便沿用"NCCL"叫法,实际跑的是 RCCL。
- 为什么要分开:ray 的心跳是 TCP,如果让它走 RDMA 网会不稳、容易误判"节点死了";而 NCCL 的大流量必须走 RDMA 才快。所以**管理网管控制,RDMA 网管数据**。

**ray 集群怎么起的**:4 台机器组成 **1 个 head + 3 个 worker** 的 ray 集群——node-6 起 head(`ray start --head`,它是集群大脑),另外 3 台起 worker(`ray start --address=<head 管理IP>:6379`)连到 head。每台在自己容器里各跑一次 `ray start`。

**这里有个容易踩的坑——ray 用哪张网报身份**。miles 默认自己起 ray;我们改成手动起(开关 `MILES_SCRIPT_EXTERNAL_RAY=1` = "别自己起,连我起好的"),就是为了给每个节点显式指定 `--node-ip-address <管理IP>`,**强制 ray 用管理网的 IP 报身份**。否则 ray 可能挑到 RDMA 网的 IP 去跑心跳(head↔worker 的控制通信)→ RDMA 网不适合这种碎 TCP 心跳 → 心跳丢/超时 → head 误判 worker 死了 → 踢掉 → 集群散架。所以管理网 IP 给 ray 控制,RDMA 网留给 NCCL 数据。

**NCCL 的一串环境变量怎么定的**:先用 `ip addr` 看清楚哪张网卡是管理网、哪张是 RDMA;RoCE 要认版本(GID index 选 1 = RoCEv2);RDMA 网卡有多条 rail(ionic 各 rail)要都告诉 NCCL;再开 GPU Direct RDMA(GDR)。

**带宽实测**:用 nccl-tests `all_reduce_perf` 测过——**卡内 8 卡(xGMI/Infinity Fabric)= 392 GB/s**,很高、正常。跨节点那个早先冒烟只有 ~46 GB/s、疑似偏低(像只用了 ~1 rail),但**训练节点在跑、没法直接复测**;拿空闲节点试又不在同一 fabric 段、跨节点 comm 建不起来。所以**跨节点带宽 + 是否用满所有 rail,留到训练告一段落在训练节点上正经复测**(待办)。注:训练的跨节点梯度 all-reduce 本身是成功的(grad_norm 正常),这是性能问题不是正确性问题。

**存储**:模型权重(torch_dist 格式)一开始放 NFS,但 NFS 读只有 ~90MB/s、扛不住 32 卡同时加载,改放**每台机器的本地 NVMe**。(NFS 慢的问题后面可以再查,列待办。)

## 四、并行配置 + 首次点火

把 291B / 43 层摊到 32 卡上,用 **TP8 / PP4 / EP8**(张量并行 8、流水并行 4、专家并行 8;没有数据并行)。43 层按流水切成 11+11+11+10。训练和 rollout **共用同一批 32 卡分时复用**(colocate),训练和推理都用 FP8 blockwise。

- 相关配置进 commit `31baa31`(`run_deepseek_v4.py` + 多机启动脚本)。
- 首次点火 → 走通引擎起、加载权重、跨节点连接、权重同步 → **打印出第一个 grad_norm**。这一步意义重大:grad_norm 是把 32 卡的梯度跨 4 台机器 all-reduce 出来的,**它出现 = 跨节点通信端到端打通**,是整个多机训练的验收锚点。

## 五、把 rollout 的三个坑填上(sglang 侧)

镜像用的是 triton 3.4 / ROCm,DSv4 的推理路径一路踩坑,全部修好并 commit 进 sglang fork(不打临时补丁):

1. **稀疏索引重放的限制**:DSv4 的 DSA 索引重放(记录推理时注意力选了哪些 KV、训练时重放)在 sglang 里由一个 capturer 实现,它**要求 rollout 的注意力张量并行 = 1**(`assert attn_tp_size==1`,DP-only)。而 `attn_tp = sglang_tp / sglang_dp`——launcher 的 Flash 分支写死 `tp4/dp1` → **attn_tp = 4**,不满足,直接 assert 崩。解决:关掉 `--use-rollout-indexer-replay`(miles 侧)。
   - **这不是 NV/AMD 的差别**:这个 `tp4/dp1` 是 launcher 里 Flash 的默认,NV 用同一份跑 Flash 也是 attn_tp=4、一样撞。attn_tp=1 只在 `tp==dp` 时出现(比如 Pro 分支 tp32/dp32 + 显式开 DP 注意力 → attn_tp=1)。**真要在 attn_tp=4 下开 indexer-replay**,得去掉那个 assert + 从 rank0 广播 top-k(`SGLANG_DSA_TOPK_BROADCAST=1`)——早先 4-layer bringup 就是这么让 TP 捕获跑通的。我们这轮为省事直接关了(且 TP-attention 下实测无 measurable 收益)。
2. **一个属性缺失**:上面关掉索引重放后,`indexer.py` 里有段代码仍**无条件去读**一个只有开了 capturer 才会写上的属性(`c4_sparse_raw_indices`)——没开时这属性根本不存在 → `AttributeError` 直接崩。修法:把直接读改成 `getattr(..., None)`,读不到就当 None(本来非 capture 路径就该是 None),不崩(commit `f46525c85a`,`dsv4/indexer.py`)。
3. **triton 3.4 的广播变严**:KV 压缩 kernel 里一个 `tl.load` 的指针形状是 `(1, D)`、mask 是 `(BLOCK, D)`,老 triton 自动广播、triton 3.4 不干了直接报错。修法:加一个"乘 0 保形状"的项 `(slot_offs*0)[:,None]` 把指针撑成 `(BLOCK, D)`,不改任何地址、纯骗过形状检查(commit `914dd3d93d`,`dsv4/fused_compress_triton.py`)。
   - 这个 fix 是通用的 triton 3.4 兼容修复(不是 AMD 专属),**值得往上游提 PR**。

### 后续:R1 —— triton 3.4 → 3.7 升级(以及一个必打的 PyTorch patch)

为解锁 DSA paged-MQA 的 preshuffle 快路(`page_size=64`,替掉慢的 legacy `page_size=1` 回退),把镜像里的 triton 从 3.4 升到 **AMD ROCm-7.0.0 原生的 3.7**(`triton==3.7.0+amd.rocm7.0.0` + 同 hash 的 `triton-kernels`),固化进 `docker/Dockerfile.rocm`(commit `6f4733c`)。

**代价:必须给 PyTorch 打一个 inductor 补丁。** torch 2.9.0a0 的 `_inductor` 会去读 triton 的 `cluster_dims` / `num_ctas`(CUDA thread-block-cluster 概念),而 triton **3.6+ 删了这俩属性** → inductor 生成 elementwise kernel 时崩(`'KernelMetadata' object has no attribute 'cluster_dims'`)。补丁把那行改成 `getattr(..., 默认值)` 兜底(`docker/amd_patch/latest/torch_inductor_triton37.patch`,以 git-apply 方式打,和 megatron/miles 的 patch 一个套路)。

**关键:这个 patch 不是永久的,是"老 torch × 新 triton"不匹配造成的。** sgl-dev 的 **rocm720(ROCm 7.2)base** 已经把 triton 顶到 **3.6** 且配套了新 torch **2.9.1+rocm7.2.0**(两者是 sglang 测过的兼容组合)——用那个 base 的话 **triton 安装和这个 inductor patch 两样都能省**。我们现在用不了它、只能留在 rocm700(triton 3.4)+ 自己装 triton + 打 patch,唯一原因是 **ROCm 7.2 的 RCCL colocate 显存泄漏**(legacy IPC-export 路径,VMM capability 被 stub 成 0,每轮 rollout↔train cycle 漏 ~3.6GB → colocate 训练用不了)。**一旦 ROCm 7.2 修好这个泄漏,就能切到 rocm720 base,把这套 triton 安装 + inductor patch 全删掉。**

**实测结果(R1)**:paged-MQA 快路解锁后数值正确(raw_reward 0.5547,落在基线区间内),稳态 rollout **6.9min vs 基线 7.4min ~7% faster**(step0 的 14.2min 是 triton JIT 冷启动,不可比)。固化进镜像 `miles-rocm700-mi35x-triton37-20260704`。paged-MQA gate 在 `dsa/utils.py:46-56 aiter_can_use_preshuffle_paged_mqa()`,升级后返回 True。

## 六、显存这道坎(4 节点的主要难点)

我们只有 4 节点,而 DeepSeek 官方这个模型的标定配置是 **8 节点**。少一半机器 → 每台机器要装 2 倍的层 → 优化器(FP8 训练但优化器状态是 fp32)光 offload 到主机内存这一项,**每台就吃掉 ~77% 的 1.5TB**。于是内存卡得很死,踩了两轮:

**第一轮:主机内存 OOM。** rollout 时还要把模型权重临时挪一份到主机内存,叠上去顶到 95% → OOM,死在第 0 步到第 1 步的切换。
- 试过把优化器矩(Adam 的 m/v)存成 bf16 想省内存——**没用**:`--optimizer-cpu-offload` 在主机上那份 buffer 恒为 fp32,dtype 开关只影响 GPU 侧的路径(grad_norm 变了证明它生效了,但主机内存一点没省)。
- 真正管用的:① 优化器**不要 100% offload**,留一部分在 GPU(`--optimizer-offload-fraction`);② rollout 时**别把模型权重挪去主机**(`--no-offload-train`)。主机内存降到 ~74%。

**第二轮:换成 GPU 显存吃紧。** 优化器留回 GPU 的部分太多,rollout 时显存顶到 99%(只剩 3GB),权重同步那一下瞬时波动就把一个 sglang 引擎挤崩(CUDA coredump)、连接断掉、任务挂。
- 修法:把 offload 比例从 0.66 调到 **0.75**——多挪一点优化器回主机(主机还有余量),给 GPU 留出 ~16GB margin。主机 79% / GPU 94%,两边都稳。

**一句话**:4 节点是"主机内存和 GPU 显存双双吃紧",`--optimizer-offload-fraction 0.75` 是在两者之间找到的平衡点,**保持 fp32 优化器、不降精度**。(这类调参就记录结论即可;后续可整理成一个讲"4 节点显存平衡"的 PR/文章。)

## 七、接上 wandb、eval,稳定多步

- 开 wandb(API key 走 ray 的 runtime-env 注入,不落到任何日志/脚本明文)。
- eval 用 **AIME-2024**,每 20 步评一次。
- **连续 17 步稳定,零崩溃**(第 17 步手动停止,非故障;wandb 因此标 Crashed):grad_norm 一直健康(0.08~0.17);reward 围 0.5 波动(raw_reward ~0.496)、还看不出学习趋势(lr 只有 1e-6,是 bring-up 的保守设定,早期正常);eval 的 AIME ~0.44——这个数被 eval 的 4096 token 上限压低了(57% 的题没写完就被截断)。
- **训推一致性(fp8 关键健康指标)**:`train_rollout_logprob_abs_diff ≈ 0.044`、`train_rollout_kl ≈ 0.0087`(全程稳定)。这俩量的是"Megatron 重算的 log-prob"和"sglang fp8 rollout 的 log-prob"差多少——**~0.044 极小,说明 fp8 训练侧和 fp8 rollout 侧数值高度一致**(fp8 blockwise 量化 + R3 路由重放共同保证的)。这是判断 fp8 RL 有没有训推漂移的核心指标,越小越好。

---

## 各阶段耗时(一步 ≈ 21 分钟;wandb 全程 11 步平均)

| 阶段 | 平均耗时 | 说明 |
|---|---|---|
| **rollout 生成**(train_wait) | **445s ≈ 7.4min** | sglang 生成 256 条(每条最多 4096 token);占一步 ~40%(wait_time_ratio 0.4) |
| log_probs | 235s ≈ 3.9min | Megatron 算 rollout 数据的 log-prob |
| **actor_train**(前向+反向+optimizer) | **572s ≈ 9.5min** | 含 optimizer.step;~1408 tok/s、~4.7 TFLOPS(偏低) |
| ├ train 总(log_probs+actor_train) | 807s ≈ 13.5min | |
| update_weights(Megatron→sglang) | 38s | |
| data_preprocess | 0.2s | |
| **一整步(step_time)** | **1252s ≈ 20.9min** | |

> **反直觉点:训练(~13.5min)比 rollout(~7.4min)还久。** 早期 rollout 是大头;换成 `offload-fraction 0.75` 后更多优化器放主机、`optimizer.step` 的 D2H/H2D 变多,**actor_train 涨到 ~572s** —— 这是"显存换时间"的代价。所以时间大头在训练侧;但 rollout 的**生成吞吐**(~250 token/s/引擎)仍是另一条独立可优化线(见待办)。数据取自 wandb `perf/*` 指标(step_time/train_wait_time/actor_train_time/… 11 步均值)。

## 训练反向 + 优化器详解(为什么 actor_train ~572s 是大头)

一步 actor_train 里,"反向"不是纯反传,它串了 5 件事:

1. **激活重算(recompute)** — `--recompute-granularity full --recompute-num-layers 1`:每层只存输入,backward 时**先把该层前向重跑一遍**再算梯度。省 HBM(中间激活全丢),代价 +~33% 算力,是 bring-up 期为塞进显存主动付的。
2. **两类 fp8 GEMM** — dgrad(`dx=dY·W`,往上传梯度)+ wgrad(`dW=dYᵀ·x`,权重梯度),走 TE 的 hipBLASLt / blockwise-triton。**`--no-gradient-accumulation-fusion` 必须开**:ROCm 上 TE 的 blockwise grouped-fp8(MoE grouped linear)不支持把 wgrad 融合累加进 main_grad(抛 `NotImplementedError`),所以关融合,wgrad 当普通梯度返回、由 DDP hook 单独累加进 fp32 buffer(数值等价)。NV 支持融合、保留。
3. **梯度 fp32 累积** — `--accumulate-allreduce-grads-in-fp32`:即使 fp8 算,梯度一律 cast 成 fp32 累积/规约(bf16/fp8 尾数太短,多微批小梯度累加会"大数吃小数")。每 rank ~0.5GB fp32 grad buffer(和 param 是 fp8/bf16 无关)。
4. **跨节点通信其实最少** — DP=1 → 没有经典的 data-parallel 梯度 all-reduce(那条大流量消失);TP8/EP8 都在节点内(xGMI);**只有 PP4 的激活梯度 p2p 出节点**(走 RDMA,点对点、量小)。所以瓶颈不在跨节点。
5. **优化器 step = 真正的大头(~380s→~572s 主因)** — `--optimizer-cpu-offload --optimizer-offload-fraction 0.75 --overlap-cpu-optimizer-d2h-h2d`:75% 参数在 **host 上用 CPU AdamW** 更新 + 每步 grad D2H / param H2D 穿 PCIe;只有 25% 留 GPU 用 TE FusedAdam。overlap 用两条 stream 把传输和计算流水,但**藏不掉 host Adam 的计算本身** → 被 CPU 吞吐 + PCIe 带宽两头卡,+~190s。
6. **precision-aware optimizer 配套省字节** — `--use-precision-aware-optimizer` + store_param_remainders:master 权重存成 **bf16 + int16 remainder(2 字节,非 fp32 的 4 字节)**,Adam 时重建 fp32 等价精度算再写回;m/v 也可低精度存。目的:每步 D2H/H2D 要搬的优化器状态更少 → 缓解 offload 传输瓶颈(所以和 cpu-offload 强制配套)。

> **一句话**:actor_train 慢 = "用算力换显存(重算)+ 用墙钟换显存(75% CPU offload)"两笔账叠加,**不是跨节点通信** —— DP=1 下反向的跨节点流量反而最少。

**backward kernel 对照(AMD vs NV)** — 已对着容器实际代码逐行核实:

| 计算 | AMD(我们) | NV | 差别 |
|---|---|---|---|
| 激活重算 checkpoint | Megatron→TE `te_checkpoint`(fp8 走此路,`transformer_block.py:626`) | 同一份 TE 代码 | **同源零分支** |
| dgrad GEMM(fp8 blockwise) | **TE Triton kernel** `blockwise_fp8_gemm`(dense)/ grouped 变体(MoE);IS_HIP 时短路 return,**不经 hipBLASLt**(`gemm.py:412`) | cuBLASLt(dense)/ cutlass(grouped);**此 fork 无 DeepGEMM** | 后端不同(Triton vs cuBLASLt/cutlass) |
| wgrad GEMM(fp8 blockwise) | 同上 Triton;MoE 走独立 variable-K grouped Triton(`grouped_linear_blockwise.py`) | cuBLASLt / cutlass | 后端不同 |
| **wgrad 累加融合** | **关**(`--no-gradient-accumulation-fusion` 是**全局** flag,dense 也连带关)。真实缺口:ROCm blockwise-**grouped**(MoE)硬 `raise NotImplementedError`(`grouped_linear_blockwise.py:85`)→ wgrad 当普通梯度、DDP post-hook `main_grad.add_()` 累进 fp32(`distributed_data_parallel.py:478`) | **开**;注:APEX `wgrad_gemm_accum_fp32` 走 core-TP-linear,与 TE GroupedLinear 是两套 fusion;apex 在 ROCm 上**也装了**,非 NV 独有 | ⚠️ **主要实质差别**(仅 MoE-grouped 路径) |
| 梯度 fp32 累积 | fp32(`--accumulate-allreduce-grads-in-fp32`) | fp32 | 同 |
| 跨节点梯度规约 | TP8×PP4×EP8×DP1,world=32;**每 PP stage=1 节点**(TP8 填满)→ TP all-reduce 与 **EP8 all-to-all 都在节点内**,DP=EDP=1 无跨节点 all-reduce → **唯一出节点=PP p2p**。RCCL over RoCEv2 RDMA | NCCL;同拓扑 | 同(RCCL=ROCm 版 NCCL) |
| attn/indexer 反向 | 自定义 autograd + **TileLang JIT**(`tilelang_sparse_mla.py`/`_indexer.py`),**非 Flash**;q-RMSNorm 走 Liger triton | 同一份 TileLang 源;q-RMSNorm 走 torch | **同源**(仅 RMSNorm 后端小差) |
| 优化器 GPU 部分(25%) | TE `FusedAdam`(HIP);HDO GPU 组=TE Adam(`optimizer/__init__.py:336`) | TE `FusedAdam`(CUDA) | 同源 |
| 优化器 offload(75%) | host CPU `torch.optim.AdamW` + PCIe;按 numel 累到 0.75 阈值切(`hybrid_optimizer.py:258`) | 同代码(NV 8 节点通常不需) | 配置差别,非 arch |
| master 权重 | **fp32 全量拷贝**(offload+precision-aware 下 `param_update_in_fp32=True`);`multi_tensor_adam_param_remainder` int16 余数**不在本 run 计算路径**(仅非-offload 纯 precision-aware 才用) | 同 | 同(本 run 无 int16-remainder) |
| comm-overlap | 未启用(无 `--tp-comm-overlap`/`--overlap-grad-reduce`) | 未启用 | 两侧都关 |

> backward 侧 **AMD/NV 绝大多数同源**(TE + Megatron + TileLang 同一份源)。差别集中在 **① blockwise fp8 GEMM 后端**(AMD 走 TE 自带 Triton kernel、hipBLASLt 不参与;NV 走 cuBLASLt/cutlass、无 DeepGEMM),和 **② MoE-grouped wgrad 累加融合**(ROCm 该路径硬 raise → 关全局融合、用 DDP hook 补一次 fp32 累加,数值等价多一步)。其余(重算/attn/FusedAdam/offload/master)是同代码换 HIP 后端或纯配置差别。

## 当前 recipe(关键配置说明)

```
并行:      训练 TP8 / PP4 / EP8;colocate;FP8 blockwise(e4m3)
优化器:     --optimizer-cpu-offload --optimizer-offload-fraction 0.75 --no-offload-train(保持 fp32)
rollout:    8 个 sglang 引擎,每个 TP4/EP4;--sglang-mem-fraction-static 0.6
RL:         GRPO,lr 1e-6,num-rollout 3000,每步 32 prompt × 8 采样
数据/eval:  dapo-math-17k 训练 / AIME-2024 评测
```

- **为什么 rollout 用 TP4 不是 TP8**:训练要 TP8 把大模型切开算;但 rollout 是逐 token decode、访存瓶颈,**小张量并行 + 多引擎并行**吞吐更高。32 卡拆成 8 个 TP4 引擎并行生成,比 4 个 TP8 引擎快(每 token 的跨卡通信更少、并行序列更多)。
- **R3(路由重放)开了没**:开了一半。MoE 的专家路由重放 `--use-rollout-routing-replay` **开着**(它记录推理时每个 token 路由到哪些专家、训练时重放,消除"训推不一致"导致的 MoE RL 崩溃);它的 DSA 稀疏版 indexer-replay 因为要求 rollout TP=1、我们是 TP4,**关掉了**。

## 和 NVIDIA 官方 64 卡(8×H200)配置对比

| 维度 | NV 官方 8 节点 / 64 H200 | 我们 4 节点 / 32 MI355X |
|---|---|---|
| GPU 显存 | H200 141GB/卡 | MI355X **288GB/卡**(更大) |
| 并行 TP/PP/EP | 8 / **8** / 8 | 8 / **4** / 8 |
| 每节点层数 | ~5 层 | ~11 层(2 倍) |
| 每节点优化器 offload | ~38% 主机内存 | ~77%(逼近内存墙) |
| offload 策略 | 全 offload 就够 | 必须部分 offload(0.75)+ 不 park |
| rollout 显存 | 宽松 | 99% 走钢丝、要留 margin |
| cuda graph | 可开(更快) | 挤,暂时关着 |

> **核心差别就一个 PP(4 vs 8)**:摊到一半机器 → 每台内存翻倍 → 主机/GPU 双紧。8 节点是根本解;4 节点是靠 offload 比例硬平衡出来的。
> (NV 侧的主机内存规格未知,故不列。)

**rollout 侧对比**(部分为推断,待与 NV 配置核对):

| 维度 | NV 官方 | 我们 4 节点 |
|---|---|---|
| 注意力并行 | 同一 launcher 的 Flash 分支也是 tp4/dp1 → **attn_tp=4** | tp4/dp1 → **attn_tp=4**,EP4,8 引擎 |
| indexer-replay(DSA 版 R3) | Flash 下同样撞 `attn_tp==1` assert(非 NV 专属);要开需去 assert + rank0 广播 top-k | **关**(未去 assert;TP-attention 下也无 measurable 收益) |
| routing-replay(MoE R3) | 开 | 开(一致) |
| cuda graph | 开 | 关(显存挤) |
| topk / 索引器 kernel | CUDA 原生 | AMD:原生 HIP topk + tilelang logits(部分因 triton 版本退 legacy,见待办) |

## 全流程 kernel 对照(AMD gfx950 vs NV CUDA)

> 前提:V4-Flash rollout 走 sglang 的 `dsv4/` 稀疏注意力栈(AMD `deepseek_v4_backend_hip_radix.py` / NV `deepseek_v4_backend.py`)。另一条硬约束:**DeepGEMM 在 ROCm 结构性不可用**(`deep_gemm_wrapper/configurer.py:24`),所以 NV 用 DeepGEMM 的地方 AMD 都落到 aiter。

**Rollout(sglang):**

| 计算 | AMD(我们) | NV |
|---|---|---|
| DSA 索引器 logits | tilelang `tilelang_fp8_paged_mqa_logits`(或 aiter gluon) | DeepGEMM `fp8_paged_mqa_logits` |
| DSA paged-MQA 取数 | aiter gluon;triton 太老退 legacy page_size=1(慢) | DeepGEMM(同上) |
| DSA top-k | 原生 HIP op `deepseek_v4_topk_transform_512` | CUDA-JIT `TopKKernel::transform` |
| Sparse MLA attention | triton(默认 tilelang;被 hack 强退 triton) | FlashMLA `flash_mla_with_kvcache` |
| fp8 GEMM(a8w8 blockscale) | aiter `ck/triton gemm_a8w8_blockscale`(gfx950 tuned) | DeepGEMM→CUTLASS→TRTLLM→Triton |
| MoE grouped gemm | aiter `fused_moe` | DeepGEMM grouped / CUTLASS |
| RMSNorm / RoPE | aiter `rmsnorm2d_fwd` / `get_rope` | flashinfer / sgl_kernel |

**训练(Megatron + miles):**

| 计算 | AMD(我们) | NV |
|---|---|---|
| fp8 训练 GEMM | **TE**(Jessica ROCm fork)`Float8BlockScaling`,CK/hipBLASLt;需 `--no-gradient-accumulation-fusion` | **TE** 官方,cuBLAS/cutlass(保留 wgrad-accum fusion) |
| SwiGLU fused clamp | 同源 Megatron `jit_fuser`→ROCm Inductor Triton | 同源,CUDA Inductor Triton |
| q-RMSNorm / CE / MHC | **Liger triton**(fwd+bwd) | naive torch(MHC 需 CUDA-only tile_kernels) |
| DSv4 自定义 RoPE | naive torch `view_as_complex` | **同为 naive torch**(NV 也没融合) |
| MoE permute / q_norm/kv_norm | TE(HIP binding) | TE(CUDA binding),同源 |

**一句话**:
- **两边同源、只是 backend 不同**:fp8 训练 GEMM(都是 TE)、SwiGLU、MoE permute、q/kv_norm、DSv4 RoPE(两边都 naive torch)。
- **AMD 仍走慢/替代路(NV 有更优 CUDA 原生)**:DSA logits+取数、top-k、sparse MLA、fp8/MoE GEMM(NV 首选 DeepGEMM/CUTLASS/FlashMLA,AMD 因 DeepGEMM 在 ROCm 禁用而落 aiter/triton)。
- **少数 AMD 反而更优**:q-RMSNorm/CE/MHC 那组——NV 走原生 torch,AMD 用 Liger triton 顶上(是 enablement 加速,不是 fallback)。

## kernel 库落点选型:量化算子的 SOTA landscape 调研(aiter vs Primus-Turbo vs Liger)

> **目的**:为"DSv4-Flash 的优化往哪个 kernel 库贡献"给一个能站得住的调研 —— 不是"什么能用先上",而是 holistic 权衡 maintenance / popularity / 开放度 / license / quant 支持(FP8 block/group、FP4/MXFP4)/ activity,兼顾 **AMD benefit vs community benefit**,配论据 + proposal。候选:leadership 点名的 **Primus-Turbo**、**Liger**,加一个关键第三方 **aiter**(AMD 官方手写算子库,会议里被听成 "Valim")。数据截至 **2026-07**,均带日期/证据。

### 方法论:怎么判断一个库"是不是 SOTA"

SOTA 不是单一分数,是**沿几条轴、锚定前沿、带日期**地比:
1. **量化粒度轴**(越靠右越新):per-tensor → per-token/row → **block 128×128**(DeepSeek)→ **microscaling MX 1×32**(OCP MX,E8M0)→ **NVFP4 1×16**(NVIDIA,两级 scale)。
2. **精度轴**:FP8 e4m3 → MXFP8 → MXFP4 → NVFP4。
3. **训练 vs 推理**:只有前向,还是有 **fwd + bwd**(dgrad/wgrad 量化 GEMM)。
4. **硬件原生度**:arch 特调(MFMA/ASM/shape-tuned config)vs 通用 Triton(无调优)。
5. **锚定前沿**:以 NVIDIA **TransformerEngine / CUTLASS / DeepGEMM** + **torchao** 为参照系,看目标库支持啥、**何时上的**,相对量落后多少。
6. **成熟度 ≠ 有格式**:有 blockscale 代码 ≠ SOTA —— 要在目标硬件(gfx950)上**数值正确 + 调优过**。
7. **"对的指标"随负载变**:DSv4 是混精栈,"谁 FP8 覆盖最多"是错指标;对的是"热点 GEMM 快 + 数值正确 + 训推数值一致"。

### 前沿标杆(参照系):现在什么算 SOTA

| 库 | FP8 per-tensor | FP8 row/token | FP8 128×128 block | MXFP8 (1×32) | MXFP4 (1×32) | NVFP4 (1×16) | 训/推 |
|---|---|---|---|---|---|---|---|
| **DeepGEMM**(DeepSeek)| — | ✅ 1×128 | ✅ **核心** | UE8M0(SM100)| FP8×FP4 混合(26-04)| — | 都有(wgrad 25-05)|
| **CUTLASS**(NV)| ✅ | ✅ | ✅ | ✅(3.8,25-01)| ✅ | ✅(prod 4.2,25-09)| 构件 |
| **TransformerEngine**(NV)| ✅ | ✅ | ✅ Float8BlockScaling(v2.3,25-04)| ✅ v2.0 25-02(Blackwell 默认)| via NVFP4 | ✅ NVFP4(v2.8,25-10)| **训练优先** |
| **torchao**(PyTorch)| ✅ | ✅ rowwise(v0.9,25-02 生产默认)| — | ✅ 原型 | ✅ PTQ/QAT | ⚠️ 训练仍原型(# 3293)| 都有 |
| **AMD aiter** | ✅ | ✅ | ✅(gfx950 **有 bug**)| via TE-fork(实验)| ✅ w4a4/w4a8 MoE(新)| — | **仅推理** |
| **AMD Primus-Turbo** | ✅ | ✅ | ✅(转调 aiter)| ✅ | ✅ gfx950 原生 ASM | — | **训练** |

微缩放标准:**OCP MX v1.0(2023)** = 32 元素块 + 共享 **E8M0** 幂次 scale → MXFP8/MXFP4。**NVFP4**(NVIDIA/Blackwell)= 16 元素块 + **两级 scale**(块级 FP8 E4M3 + 张量级 FP32),等位宽下比 MXFP4 精度高。

**四个关键结论:**
1. **FP8 训练 SOTA scheme**:已从 per-tensor 彻底转向细粒度,按硬件分 —— Hopper = **rowwise/per-token**(torchao 生产默认)或 **DeepSeek 128×128 block**;Blackwell = **MXFP8 1×32**(TE v2.0 已设默认)。per-tensor 过时。
2. **FP4 前沿**:**NVFP4 在赢**(尤其训练;arXiv 2509.25149,12B×10T token 逼近 FP8 loss)。MXFP4 赢在可移植(OCP 标准)。生产级 FP4 训练目前只有 TE(NVFP4,v2.8)。
3. **DSv4 混精栈——"FP8 覆盖最多"是错指标**:DSv4 已把 FP8 限定在 forward/rollout,BF16 主训、FP32 敏感项。价值在热点 block-scale FP8 GEMM 快 + 正确 + **rollout↔train 量化边界一致**(on-policy)。
4. **AMD 落后多少**:硬件(CDNA4/gfx950:FP4/FP6 + block-scaled MFMA)+ 地基(hipBLASLt/UE8M0)**与 Blackwell 齐平**;差在 kernel 库层 —— 推理 GEMM **~6-9 月**,FP8/MX 训练 **~9-12 月**。缺口最大:可靠 gfx950 block-scale FP8、TE-fork 的 MXFP4/NVFP4、scaled-MFMA tuned GEMM。

### 三个候选逐个体检

#### ① aiter(ROCm 官方)—— AMD 侧前沿,但仅推理

前向 GEMM:`gemm_a8w8_blockscale`(FP8 e4m3,act per-1×128 / **weight per-128×128 UE8M0**)、`_bpreshuffle`(per-token)、`gemm_a8w8`(INT8 per-token)、`gemm_a8w4`(FP8 act / **MXFP4** weight per-1×32)、`gemm_a4w4`(**MXFP4** e2m1)、fused MoE(FP8/INT8/MXFP4/MXFP8)。**无 NVFP4;e5m2 非一等 GEMM 路。**
- **训练**:❌ **仅前向,无 dgrad/wgrad 量化 GEMM**(ATOM 文档明确);AMD 的训练反向走 hipBLASLt/TE,不在 aiter → **miles 的 FP8 训练反向不能靠 aiter**。
- **时效**:极活。v0.1.16.post3(26-06-26)、v0.1.17-rc0(26-07-04),双周 release + 热修,Q2 仅 gfx950 就 41+ PR。
- **gfx950**:原生一等 + 真机 CI,但成熟度有坑 —— 旗舰 `blockscale_bpreshuffle` 在 gfx950/ROCm7.2 数值错(sglang # 28685)。
- **DSv4 专属**:roadmap # 3442 有 fused `qk_norm_rope_quant`、CSA/HCA、MoE A8W4;ATOM recipe:V4-Pro 路由专家=FP4 e2m1 per-1×32、V4-Flash=FP8 128×128 blockscale。
- **归属**:AMD 控盘,收外部 PR(CLA),MIT,~478⭐。
- **SOTA 判定**:**AMD 侧的前沿**(CDNA4 上无更强 ROCm 选项),但比 NV/社区前沿落后 ~6-12 月且更窄(无 NVFP4、无训练反向、新硅正确性还在稳定)。

#### ② Primus-Turbo(AMD-AGI)—— AMD 侧训练量化的 SOTA

- **scheme**:TENSORWISE / ROWWISE / **BLOCKWISE(128)** / **MX_BLOCKWISE(32,E8M0)**,act+weight 都量化。
- **精度**:FP8 **E4M3/E5M2/HYBRID**、FP4 **E2M1**、**MXFP8 + MXFP4**(gfx950 原生 ASM,CDNA4 scaled-MFMA);roadmap 还有 FP6。
- **训练**:✅ **一等 fwd+bwd** —— `torch.autograd.Function`(`FP8GemmBlockFunction`/`FP8GemmMXFunction`);反向把梯度量到 FP8(dgrad row-wise / wgrad col-wise),HYBRID = **E4M3 fwd / E5M2 bwd**;MoE 有 `grouped_gemm_fp8/fp4`。
- **时效**:v0.1.0(25-09)→ v0.3.1(26-06),提交到 26-07-01(flydsl mxfp8 gemm、mxfp4 grouped)。66⭐ 是新/小众非不成熟;进 AMD MLPerf Training v6.0。
- **gfx950**:原生一等,MXFP8/MXFP4 gate 到 gfx950+,手调 ASM;FP8/FlashAttn 依赖 aiter。归属:AMD 内部(AMD-AGI),MIT。
- **SOTA 判定**:**AMD 低精度*训练* 的 SOTA** —— 正是要 beat/复用的对象。短板:纯 AMD 内部、社区杠杆最低、无 NVFP4。

#### ③ Liger-Kernel(LinkedIn)—— 不是量化库,别拿它比 quant

- **量化 GEMM**:**基本为零**(仅实验性 `matmul int2xint8`);无 FP8/FP4/MX GEMM,无 scale 基建。
- **它其实是**:BF16/FP16 的**融合训练 kernel**(纯 Triton,都有 fwd+bwd)—— RMSNorm、RoPE、SwiGLU、**Fused Linear CrossEntropy**(招牌省显存)、MoE、**mHC**。
- **mHC 澄清**:mHC = **Hyper-Connections**(残差连接的学习化泛化,**不是 multi-head attention**)。`LigerMHC`,fwd+bwd,但是融合 elementwise/reduction kernel,**不是量化 GEMM**。
- **gfx950**:**纯通用 Triton,零 arch 调优**(号称支持 AMD,但无 gfx950 实例化 / CDNA4 MFMA / shape-tuned config)。时效:v0.8.0(26-04-30),~6.5k⭐,BSD-2,真社区(80+ 贡献者)。
- **SOTA 判定**:量化上缺席,但与量化正交;价值 = norm/激活/CE/mHC 等非量化融合算子(BF16、通用 Triton、无 AMD 调优)。

### 结论 + proposal

- **量化 GEMM 之争其实是 aiter vs Primus-Turbo**(Liger 那栏全空)。
- 推理侧 quant → **aiter**(生产在吃,但要绕 gfx950 blockscale 正确性坑);训练侧 quant fwd+bwd → **只有 Primus-Turbo 真有**(aiter 无反向);mHC/融合非量化算子 → **只有 Liger 现成有**(但要补 gfx950 调优)。
- 对 DSv4 混精,"谁 FP8 全"是错问题;真问题 = 热点 block-scale FP8 GEMM 在 gfx950 上快且正确、rollout↔train 边界一致。

**一句话定调(供 proposal)**:量化 GEMM 落 **aiter**(AMD 战略 + DSv4 现成,PR 可见度最大),mHC/融合训练算子落 **Liger**(社区杠杆 + 现成 mHC;招牌 = 补 gfx950/ROCm 支持,`mhc.py` 现为 CUDA-tuned、`num_warps` 未按 wavefront=64 调、无 `_amd` backend,对标已 merged 的 NPU mHC PR),**Primus-Turbo** 作 AMD 内部训练栈备选,自写仅原型(HipKittens 先例:独立小库被 AMD 反吸收进 TE)。AMD 落后前沿 ~6-12 月的缺口(可靠 gfx950 block-scale FP8、TE-fork 的 MXFP4/NVFP4、scaled-MFMA tuned GEMM)**正是上游 PR 刷影响力的落点**。

**实测旁证:aiter 的 gluon blockscale FP8 GEMM 目前不 win**(2026-06-28,gfx950)。真实 DSv4-4layer GEMM 形状上,gluon `gemm_a8w8_blockscale`(a8w8,act 1×128 / weight 128×128)数值正确(cos=1.0、relerr_vs_ck=0),但比 aiter production CK/asm kernel **慢 2-8×**(大-M geomean gl/ck **0.28x**、K=4096 最差 ~5×),对纯 triton 也基本不赢、无一形状胜。根因:这版 gluon kernel 没用 CDNA4 关键技法——非 hardware scaled-MFMA(手动 post-MFMA 乘 block scale)、无 async_copy、非 persistent,是早期未优化 kernel(TokenSpeed blog 的 1.1-1.6× 胜绩在 attention/MoE,是别的 gluon kernel)。而且**就算它快也落不到本训练/服务**:训练侧 blockwise-fp8 GEMM 走 TE 自己 vendored 的 triton kernel(`blockwise_fp8_gemm.py`,不碰 aiter);DSv4 稠密形状不在 sglang 的 14 个 tuned 白名单(走 CK)+ 服务容器 triton 3.4(<3.6 跑不了 gluon)+ ROCm 7.0 hipcc 在 gfx95 误编 bpreshuffle 路(sglang guard hip≥7.2,# 23319)。→ 印证「自写仅原型」:价值在生态/未来,不在本训练。

### on-hardware 验证清单(有卡后做 = leadership 要的"逐项跑测精度")

对每个候选算子做 **forward + backward** unit test:输出和梯度(分别对 activation + weight 两方向)跟 FP32/torch numeric reference 比(gradcheck 式)在 **gfx950** 上跑。重点:① aiter/Primus 的 **FP8 blockscale GEMM 反向**(dgrad/wgrad);② Liger **mHC 在 gfx950 的数值 parity**(fwd+bwd);③ **rollout(sglang/aiter forward)↔ 训练(Primus/TE backward)量化边界一致性**(on-policy 关键)。

**Sources**:[aiter](https://github.com/ROCm/aiter)(roadmap # 3442)· [sglang # 28685](https://github.com/sgl-project/sglang/issues/28685)· [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)· [ROCm MLPerf v6.0](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.0/README.html)· [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)· [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)· CUTLASS 3.8/4.2 · TE(Float8BlockScaling v2.3 / MXFP8 v2.0 / NVFP4 v2.8)· torchao(# 3293)· [NVFP4 预训练 arXiv 2509.25149](https://arxiv.org/abs/2509.25149)· OCP MX v1.0 · [HipKittens](https://github.com/HazyResearch/HipKittens)· [LMSYS DSv4 Day0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)

### 调研执行计划(Step 0 → 4)

老板要的是"系统比较 + 论据 + proposal",拆成两半:**A = landscape(纸面)** + **B = on-hardware 精度实测(要卡)**。

- **Step 0 ✅ landscape 对比**:三库 + 前沿参照系 + 带日期证据 + proposal —— 就是本节上面。A 的主体已交付。
- **Step 1 ✅ 盘清"我们实际用了哪些 Liger 算子"**:4-layer 里从 Liger 走的是 **q-RMSNorm**(`deepseek_v4.py:258`)、**mHC/Hyper-Connections**(`ops/hyper_connection.py`)、**CrossEntropy**(`loss_hub/math_utils.py:956`),全部 `torch.version.hip` 门控(只 ROCm 用)。**三个都是非量化融合算子,没有量化 GEMM**(compressor norm 有意不走 Liger;SwiGLU 走 Megatron jit_fuser)。→ 关键结论:Liger 与 aiter/Primus 大体**正交**,"Primus vs Liger 按 quant 比"里 Liger 那栏本就空。
- **Step 2 ⬜(要卡)unit-test harness**:每个候选算子做 forward(固定输入 → 输出 vs FP32 reference)+ backward(对 activation 的 grad + 对 weight 的 grad,分别 vs FP32 reference,gradcheck 式),gfx950 上跑。重点:aiter/Primus 的 **FP8 blockscale GEMM 反向**、Liger **mHC 数值 parity**、**rollout↔train 量化边界一致性**。
- **Step 3 ⬜(要卡)按 quant scheme 分测**:FP8 per-token、FP8 128×128 block、Primus 的 group/MX、FP4/MXFP4 —— 记误差 + 速度。
- **Step 4 ⬜ 产出**:A(landscape)+ B(实测精度/速度)合成 slide/report + proposal(哪个算子落哪个库)。
- **卡点**:Step 2-4 要 GPU,机器被 dn 抢占;GPU 一有即上。harness 可先离线写好等卡直接跑。

## AMD kernel 隔离(`miles_plugins/amd/` 文件夹)

为了让 AMD 改动可维护、可上游,曾把 DSv4 的 AMD-specific kernel 从 NV 主线**物理隔离**到 `miles_plugins/amd/models/deepseek_v4/`。**2026-07-08 后这个文件夹已清空**(只剩 `__init__.py`)—— 三个 AMD kernel 全部消解:

| 曾经的 AMD 文件 | 现在怎么处理 | 结果 |
|---|---|---|
| `mhc.py`(7 个 mHC op 的 torch drop-in)| 真 `tile_kernels.modeling.mhc.ops`(Dockerfile sed 让 gfx950 能编,见下节)| `hyper_connection.py` 两平台同一 import,**0 分叉** |
| `cast_back.py`(FP8 反量化 Triton 重写)| 真 `tile_kernels.quant.per_token_cast_back`(靠 `hip_fp8.h` sed 解锁 fp8→float,见下节)| `qat.py` 两平台同一 import,**0 分叉** |
| `precision_aligned_ops.py`(纯 fp32 mirror)| **合进** `ops/kernel/precision_aligned_ops.py`:forward 里一行 `torch.version.hip` 分支 | 单文件、backward 共享(消除手动同步隐患),**0 import 分叉** |

**precision 为什么必须分支(已实测坐实)**:`torch.mm(bf16, bf16, out_dtype=torch.float32)` 在 ROCm 上被 **PyTorch dispatch 硬拒绝**(`RuntimeError: gemm input type at::BFloat16 and output type float is not supported for ROCm`,torch 2.9.0a0 / ROCm 7.0 实测,3 种形状全挂)—— 不是硬件不行,是 torch ROCm 后端没接 `out_dtype`。所以 ROCm 分支先 `.float()` 再纯 fp32 matmul:**和 NV cublas 路径数学等价**(同为 bf16 舍入输入 + fp32 累加 + fp32 输出),仅少了 cublas 的融合 kernel。**NV 运行时路径逐字节不变**(走 else 分支 = 原来那行)。

**还原参考**:`mhc.py` 见 `git show ba1de29^:...mhc.py`;`cast_back.py` / `precision_aligned_ops.py`(独立 mirror 版)见对应删除 commit 的父提交。

### perf-only Liger 已回退(为 V4 PR 保持最小 NV diff)

**策略**:先提一个**对 NV 改动最小**的干净版 V4 PR(能不 diff NV 就不 diff)、跑通、提上去;**加速(Liger)后面单独做**。所以把"ROCm 不需要、只为提速"的 Liger 全部回退成 NV 一致的 torch —— 这样对应的 NV 主线文件直接变回 byte-for-byte 上游。

**回退时的 commit(要还原 Liger 加速版,直接 `git show` 这些 commit 里对应文件即可,不用重写):**
- `874ad32` `[AMD] DSv4: isolate AMD kernel impls into miles_plugins/amd/...` —— **Liger 版 `ce.py` / `rmsnorm.py` 全文 + mhc 的 `_USE_LIGER_MHC` 层**都在这个 commit 里(还原参考它)。

| 回退项 | NV 现在走 | 回退后 NV 文件 diff | 曾经的 Liger 加速(还原参考) |
|---|---|---|---|
| **CE log-prob** | 内联 torch `log_softmax`+`gather` | `loss_hub/math_utils.py` → **0 diff(byte-for-byte 上游)** | `amd/.../ce.py` 用 `liger_cross_entropy`,省两次 `[R, V=129280]` 物化(no-entropy 快路);dispatch 曾在 `math_utils.py` 顶 |
| **q-RMSNorm** | 内联 torch fp32 rsqrt | `deepseek_v4.py` → 仅剩无关的 `V4Indexer(layer_id=…)` | `amd/.../rmsnorm.py` 用 `liger_rms_norm`(gemma casting);dispatch 曾在 `deepseek_v4.py:46` |
| **mHC 加速层** | 真 `tile_kernels.modeling.mhc.ops`(两平台同一 import) | `hyper_connection.py` → **0 分叉**(NV/ROCm 都 `import tile_kernels`) | `_USE_LIGER_MHC` + `liger_mhc_coeffs/pre/post_res`,在 `hc_pre_raw`/`hc_post_raw` 里(注意 `hc_post` 的 `comb.transpose` 修正,漏了 ~28% 端到端误差) |

> **注**:这里回退的只是 **perf-only 的 Liger 加速层**(CE / q-RMSNorm / mHC-Liger)。底座本身:`mhc` 和 `cast_back` 现在都走**真 tile_kernels**(靠 Dockerfile sed 在 gfx950 编译跑通,见下节),两平台同一 import、无分叉;`precision_aligned` 是唯一保留的 AMD 硬分叉(hipblas gemm 限制)。

**待办:**

0. **【提速阶段做】把回退掉的 Liger 加速改回来 + 给 NV 提 PR/issue。** (a) 本地还原:参照上表 commit `874ad32` 把 `ce.py`/`rmsnorm.py`/mhc-Liger 层放回 + 主线 import 分叉。(b) **给上游 NV 提 PR/issue**:CE log-prob 和 q-RMSNorm 用 Liger 能省显存/提速(CE 省两次 `[R,V]` 物化),NV 侧也受益 —— 这本就该是上游改进,不是 AMD 私有。提了 NV 收了,AMD 就不用维护这份分叉。

1. **【很大程度已闭合;剩硬件级 parity】mhc / cast_back 现在 gfx950 上跑的**就是**真 `tile_kernels`**(和 NV 同一份 TileLang 源码,靠 hip_fp8.h sed 让它在 gfx950 编译),不再是 AMD 重写 —— 所以"AMD-reimpl vs tile_kernels"这个缺口已消失。cast_back 已在 gfx950 多形状 bit-exact vs `float(fp8)*scale`。**剩下的**只是同一 kernel 在 NV vs AMD 硬件上的数值差(MFMA/rounding),要彻底坐实仍建议在 NV 上把 mhc 7 op 的 fwd 输出 + bwd 梯度(对 activation / 对 weight)与 gfx950 逐 op 比一遍(leadership 要的"逐项跑测精度")。`precision_aligned` 仍是纯 AMD 实现,需和 NV cublas bf16/fp32 gemm 对。
2. **用真 `tile_kernels` 的 mhc + cast_back(不重写 kernel)—— 2026-07-08 在 gfx950/ROCm 7.0 端到端验证跑通,已落 `Dockerfile.rocm`;`amd/mhc.py` / `amd/cast_back.py` 都已删。**

   base sgl-dev 镜像里**已经**有从源码为 ROCm 编好的 tilelang `a55a8230`(含 tvm-ffi,`import tilelang`/`tvm_ffi` 都干净)—— **不用自己编 tilelang**。只需在 miles 层加几条 sed(以 Dockerfile 为准):
   ```dockerfile
   RUN pip install --no-deps tile_kernels==1.0.0 && \
       TK=/opt/venv/lib/python3.10/site-packages/tile_kernels && \
       sed -i '/^import tilelang$/d; /^from \. import ($/,/^)$/d' "$TK/__init__.py" && \
       sed -i '/^from \. import engram$/d; /^from \. import mhc$/d' "$TK/modeling/__init__.py" && \
       sed -i '/T\.pdl_sync()/d' "$TK/mhc/post_kernel.py" && \
       TLH=/opt/tilelang/src/tl_templates/hip/hip_fp8.h && \
       sed -i 's|...static_cast<float>(static_cast<hip_fp8_e4_t>(*this))...|...__half2float(__half(__hip_cvt_fp8_to_halfraw(data, interp)))...|' "$TLH" && \
       sed -i 's|...hip_fp8_e5_t...|...E5M2...|' "$TLH"   # 两处 operator float,见 Dockerfile 全文
   ```
   - **`--no-deps`**:别让 tile_kernels 拉它自己那份 CUDA tilelang,复用镜像里 ROCm 编好的 a55a8230。
   - **lazy-import sed**(`__init__` / `modeling/__init__`):删急加载(`import tilelang` + 急 import engram),改懒加载 —— 否则 engram 用的 `TL_DISABLE_OUT_OF_BOUND_WARNING` enum 在 a55a8230 不存在,`import tile_kernels` 就崩。
   - **pdl_sync sed**(`mhc/post_kernel.py`):删 mhc_post 的 `T.pdl_sync()`(Hopper grid-sync 启动优化,no-op 数学,ROCm 无实现)。
   - **hip_fp8.h sed**(tilelang 源码,`/opt/tilelang/src/tl_templates/hip/hip_fp8.h`):把 `fp8_e{4,5}_t::operator float()` 的 `static_cast<float>` 换成 `__hip_cvt_fp8_to_halfraw` 的 on-device 转换 —— **这条解锁 cast_back**(fp8→float 不再撞 SDK host-only `operator float`)。上游修复在我们 tilelang fork 的 PR;base 镜像的 tilelang 收了这个修复后这两条 sed 可删。
   - 实测:`import tile_kernels` ✅;mhc 7 op `sinkhorn` fwd+bwd gfx950 JIT 编译+跑通 ✅;cast_back `per_token_cast_back` 在**全 bake 好的镜像**里多形状 bit-exact vs `float(fp8)*scale`(bf16/fp32、block 整除/不整除)✅。→ `hyper_connection.py`(mhc)、`qat.py`(cast_back)两平台同一 import,**AMD torch/triton 重写已删**。

   **现状(为什么要 sed)**:base tilelang = `a55a8230`(2026-01-29),比 tile_kernels 1.0.0 期望的 tilelang 早;tile_kernels 的 **engram 子模块**用了 `TL_DISABLE_OUT_OF_BOUND_WARNING`(tilelang commit `5951bce7`/2026-02-05 才加的 enum),a55a8230 没有 → 急加载 engram 时 `AttributeError` 崩。engram 我们不用,lazy-import 绕过即可(mhc 不依赖这个 enum)。

   **移除条件**:base sgl-dev 的 ROCm tilelang 升到 **≥0.1.8**(含该 enum,且 tvm-ffi/dlpack 已正确 ROCm 编)→ engram 急加载不再崩,这个 sed 可删。**盯 `rocm/sgl-dev` 新 tag 的 tilelang SHA**(现 0627/0708 都还是 a55a8230)。
3. **【未来时,等 AMD 真正重写 forward 级 kernel 再上】用 `--spec` + attention subclass 隔离 AMD 的 kernel 变体。** 机制:仿 `get_dsv4_spec`(deepseek_v4.py:353)做一个平行的 `get_dsv4_amd_spec`(放 `amd/models/deepseek_v4/deepseek_v4_amd.py`),返回 `ModuleSpec(module=DeepSeekV4AttentionAMD)`;`DeepSeekV4AttentionAMD(DeepSeekV4Attention)` **subclass 复用**共享模型逻辑,只 override 掉 kernel 调用;fork 发生在启动脚本选 `--spec`(NV 脚本用原 spec,AMD 脚本用 amd spec),**运行时零 if-hip**。共享类改动压到最小:把要换的 kernel 调用(如 forward:319 的 `sparse_attn_tilelang(...)`)抽成一个几行的 hook 方法(additive、可上游),AMD 子类只 override 这个 hook → 零重复、零散 if-hip。
   - **为什么现在不做**:mhc / cast_back 回归真 tile_kernels、precision 收成 forward 内联分支、Liger 回退后,**已经没有任何 import 分叉了**(mhc/cast_back 两平台无条件 import tile_kernels;precision 的 `torch.version.hip` 判断在 forward 运行时、不在 import)。也就没有"forward 级 AMD kernel 变体"需要 spec/subclass 去隔离 —— 现在建平行入口无对象、纯零收益。
   - **什么时候值得做**:当 AMD 开始有**整份不同的 forward 级大 kernel**要换(Triton / FlyDSL 重写 sparse_attn / indexer,即 `amd/.../kernel/sparse_attn_amd.py` 这类)。那些是 forward 里的 kernel 调用、不是 import gate,subclass+hook 是最干净的 swap 点,且能一次容纳多个大 kernel,平行入口的成本才摊得平。
4. **【✅ 已完成 2026-07-09】precision fallback 已实测坐实 + mirror 已消掉。** gfx950/ROCm 7.0(torch 2.9.0a0)上 `torch.mm(bf16, bf16, out_dtype=torch.float32)` 被 PyTorch dispatch 硬拒绝(`RuntimeError: gemm input type at::BFloat16 and output type float is not supported for ROCm`,3 种 compressor 形状全挂;不是硬件限制,是 torch ROCm 后端没接 `out_dtype`)。→ fallback **确实必须**。做法不是留独立 mirror,而是**合进** `ops/kernel/precision_aligned_ops.py`:forward 里一行 `torch.version.hip` 分支(ROCm 先 `.float()` 再 fp32 matmul,NV 走原 cublas 行不变),backward 共享。**独立 amd mirror 文件删除**,`amd/models/deepseek_v4/` 清空。数值:ROCm 分支与 fp32 参考 bit-exact、和 NV cublas bf16-in/fp32-accum 数学等价(同 bf16 舍入输入 + fp32 累加)。

## 优化路线图(已完成实测 + 待办)

> 稳定性验证完成后的下一阶段:**给 rollout / 训练提速**。方法论:**每个改进单独 A/B 测**(一次只动一项才知各自贡献),粗略记提升即可。基线取归档稳定 run 的 wandb 11 步均值:**step 20.9min / rollout(train_wait)7.4min / actor_train 9.5min**,reward 0.44–0.68。改动进 fork 分支 `wip/dsv4-flash-full-train-20260701-loop`(commit+push,不开 PR)。
>
> 每项:根因坐实(file:line)→ 备好确切改动 → skip-save A/B 测吞吐/正确性 → 记结论。✅ 已完成实测、⬜ 待做、⛔ 受阻。

### R1 ✅ 升 triton 3.4 → 3.7,解锁 DSA paged-MQA 快路 — ~7% faster

详见正文「五 · 后续 R1」。结论:**AMD ROCm-7.0.0 原生配对** `triton==3.7.0+amd.rocm7.0.0` + 同 hash 的 `triton-kernels==1.0.0+amd.rocm7.0.0`(源 `pypi.amd.com/.../rocm-7.0.0`)——直接 pip 装、**不碰 torch**(torch≥2.9.1 门是保守假门);单升 triton-3.6 会挂在 triton-kernels 不配套,原生配对才干净。**必打** torch inductor `cluster_dims` patch(`triton_heuristics.py:1680` 改 getattr 默认,triton≥3.6 删了 `cluster_dims`/`num_ctas`)。paged-MQA gate `dsa/utils.py:46-56` 升级后 True。**结果:稳态 rollout 6.9min vs 7.4min ~7% faster,数值正确(raw_reward 0.5547)**。固化镜像 `miles-rocm700-mi35x-triton37-20260704`。(踩坑教训:期间一次 NCCL ALLGATHER 600s hang 其实是 node-6 磁盘 100% 憋的,非 triton3.7。)

### R2 ✅ 打开 fused_qk_norm_rope — ~5% faster,零成本

日志显示 `aiter fused_qk_norm_mrope_3d kernel available` 但 flag 默认 False。gate 在 `deepseek_v4.py:499`(`_is_hip and envs.SGLANG_OPT_USE_FUSED_QK_NORM_ROPE.get()`,DSv4 确有此路径,非 Qwen3 专属)。**开法 = launch env 加 `SGLANG_OPT_USE_FUSED_QK_NORM_ROPE=1`**。结果:step1 稳态 396.8s vs triton3.7 基线 416.1s **~5% faster**,reward 0.5625(数值等价)。单点含噪但方向对 + 零成本 → **保留此 env**。

### R3 ⬜ 补 gfx950 tuned GEMM config CSV(待,需 GPU window)

`bf16_tuned_gemm.csv`(`/sgl-workspace/aiter/aiter/configs/`)存在但 **gfx950 条目 = 0** → 915 次投影 GEMM 全退回 torch;`tuned_fmoe.csv` 让 fused_moe 走 2stage default 没调优。动作 = 跑 GEMM tuner(`gradlib/gemm_tuner.py` / `aiter/tuned_gemm.py`)生成 gfx950 条目(纯 tuning、低风险)。前置:收集缺的 shape(日志有 M:15/22,N:1024,K:4096 等)+ 一个 GPU window。payoff 待测(小-M decode GEMM 走 torch 未必很慢)。

### R4 ⬜ sparse MLA attention 后端 A/B(triton → tilelang / FlashMLA)

现走 triton;代码里 tilelang / 原生 FlashMLA 两条路都在但都没 tune gfx950。值得 A/B(和记忆里"FlashMLA gfx950 tune 5-8%"对得上)。待:定位三条后端的选择开关、确认 gfx950 上哪条可用。

### R5 ⬜ 训练侧 DSv4 自定义 RoPE 融合核(ROI 低)

`view_as_complex` fp32 复数乘,每层多次、无融合。**无现成开关**,得自写 triton/aiter 融合核(有开发成本);先评估是否有 aiter 现成 rope 核可借。优先级低。

### R6 ⬜ 去掉 NO-OP / 死开关(下次重发脚本时,零风险清理)

- 去掉 `--sglang-dsa-topk-backend torch`(对 DSv4 是 NO-OP,topk 走原生 HIP kernel)。
- 删 env `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`(死开关 footgun)。

### R7 ⛔ 开 cuda graph — 在 ROCm colocate 下会 wedge,需 R9 才能真用

miles `arguments.py:2310` 在 ROCm+colocate **强制**关 cuda graph(注释:full capture HANG)。实测:patch 掉强制(env gate `MILES_ALLOW_CUDA_GRAPH_COLOCATE=1`,默认 off、无害)后,即便升到 triton3.7,capture **仍冻死**(冻在 `Capture begin bs=256`、GPU 全 idle,进程杀不掉、~197GB 显存释放不掉、**只能 reboot 节点**)。

**根因坐实 = 已知 bug [aiter # 2061](https://github.com/ROCm/aiter/issues/2061)**:aiter custom all-reduce 的 IPC buffer 用固定 VA,和 `torch_memory_saver`(colocate 的显存 pause/resume)冲突 → HIP graph capture 把 stale 指针 baked-in → GPU 硬件级 deadlock → KFD hold 进程(dmesg 卡在 `kfd_ioctl_free_memory_of_gpu → amdgpu_bo_unpin`,逐条吻合)。**fix 其实已在栈里**(aiter # 2075 + sglang # 19162/# 20155)**但被一个默认关的 env 门控没走到安全路径**:`custom_all_reduce.py:396 enable_register_for_capturing = not SGLANG_MEMORY_SAVER_CUDA_GRAPH`,该 env 默认 False、我们没设 → aiter AR 走 register 路径(= # 2061 会 wedge 那条),unreg 安全分支没走到。

**解法 = 开 cuda graph 时加 env `SGLANG_MEMORY_SAVER_CUDA_GRAPH=1`**(走 fix 的 unreg 安全路径,不 wedge),或 `SGLANG_USE_AITER_AR=false`(回退 sglang 自带 AR)。**caveat**:堵住 aiter-AR 这条后,capture 可能仍因别的 DSv4 特有原因卡 —— 上游 DSv4 piecewise cuda graph 仍 WIP([# 29657](https://github.com/sgl-project/sglang/pull/29657))、c4-indexer capture-OOM([# 29985](https://github.com/sgl-project/sglang/pull/29985))、ROCm ≤7.2.0 的 `hipEventQuery` capture bug([# 24011](https://github.com/sgl-project/sglang/issues/24011),需升 7.2.2)。所以 cuda graph 想真用于 colocate DSv4 目前不通,**留给 R9(非-colocate)**;env-gate patch 留在树里备用。
> **教训**:cuda-graph capture-hang 会 wedge GPU 驱动、需 reboot 才能清(软件层 kill/docker/gpureset 对 amdgpu D-state 全无效;`reboot -f` → SysRq-B → IPMI power-cycle 逐级兜底;若节点是别人的登录/submit 机,reboot 前先知会)。**测这类实验前必设 `SGLANG_MEMORY_SAVER_CUDA_GRAPH=1` 防 wedge,并能一冻(GPU idle + log 停 `Capture begin`)就立刻止损清进程。**

### R8 ⬜ MTP 投机解码(可行、无硬 blocker,潜在 2.55×)

DSv4-Flash **自带 MTP 头**(config `num_nextn_predict_layers=1`,`deepseek_v4_nextn.py`,mtp.0.* 1575 keys 自带 DSA+256 专家,不需外挂 draft)。配置 = `--sglang-speculative-algorithm NEXTN --num-steps 2 --eagle-topk 1 --num-draft-tokens 3`(自动复用主 ckpt 的 mtp.0 头,不设 draft-path、不开 mtp-training)。routing-replay 兼容;cuda graph 非硬依赖(有 eager 路径)→ 不像 R7 被卡。盯显存(MTP 头 + draft KV 可能要降 mem-fraction)+ reward 对齐。**⚠️ caveat** [# 29144](https://github.com/sgl-project/sglang/issues/29144):DSv4-Flash + spec decode 有 DP-attention NCCL deadlock 风险(CUDA 也复现)。

### R9 ⬜ async / 训推分离(顺带解锁 cuda graph)

现在 colocate 串行;miles 有 `train_async.py`。非-colocate 后 R7 的 memory_saver 冲突消失、cuda graph 可用。待:评估 4 节点上训推分离的资源切分(几卡 rollout / 几卡 train)+ 和现 colocate recipe 的差距。

### R10+ ⬜ 永续找新点

R1–R9 做完后 profile 当前 run(steady-decode profiler、wandb `perf/*`、rocm-smi)找新热点,对着 NV 官配 / sglang upstream / aiter 找没吃到的优化(新 kernel/flag/fusion),追加为 R11、R12… 逐项做。

### 扩展 / 根本解
- [ ] 上 **8 节点官方配置(PP8)**,一次拿回所有 headroom(4 节点显存双紧的根本解)。
- [ ] 用长 token 预算(如 32768)重测 eval,看 V4 真实 AIME 分数(现被 4096 上限压低)。
- [ ] 重开 indexer-replay(改 rollout 为 attn_tp=1 对齐 NV;需去 `attn_tp==1` assert + rank0 广播 top-k)。

### 其他 / 环境
- [ ] 在训练节点上正经复测跨节点 NCCL 带宽 + 查是否用满所有 rail(卡内已确认 392 GB/s;跨节点早先 46 GB/s 疑似偏低,空闲节点测不通)。
- [ ] 再试一次 NFS(当时 90MB/s 太慢,换了本地 NVMe)。

### 上游 PR 盘点(待逐个确认是否已在上游)
1. **triton 3.4 广播 fix**(`_c128_decode_kernel`,`dsv4/fused_compress_triton.py`)—— 通用兼容修复,最该提 → sglang `amd/dsv4-rebase`。
2. **fp8 row-linear tp_invariant gate**(`layers/linear.py`)—— 通用(部分可能已在上游 sglang-miles 线)。
3. **Megatron SwiGLU fused clamp**(`262b82419`)—— DSv4 激活融合,纯增量、NV 不受影响 → Megatron 上游。
4. **indexer-replay TP capture**(去 `attn_tp==1` assert + rank0 广播 top-k)—— 让 DSA R3 支持注意力 TP,通用价值 → sglang(diff 备份 `docs/sglang-indexer-topk-tp-capture.patch`)。
5. **E22 getattr 保护** —— 小修,可随第 1 项一起。
- 注:`offload-fraction 0.75` 那类是"配方调参"不是代码 PR,整理成一篇"4 节点显存平衡"经验更合适。

---

## 附:4-layer bring-up 报错台账(E2..E26)

> 全量 full-train 之前,先在**单机 8 卡用 4-layer 版**把「能加载 → FP8 转 torch_dist → 跑通一个 train step」整条链走通(正文「一」提过)。这一阶段的核心不是一堆独立 bug,而是**一套 build 配方 + 一套 rollout env**:把 DSv4 的 sparse-attn indexer / MHC / MoE 从 CUDA-only(`deep_gemm` / `<cuda/ptx>` / tilelang / `<cuda_fp8.h>`)整体切到 aiter/triton。验收判据(锚 AMD TE blockwise FP8 PR # 647,无公开 NV 逐张量数字):fp8-vs-bf16 **relerr ≤ 0.04**、on-policy train-vs-rollout logprob **abs-diff ≤ 0.04**(≥10 步不上升)。台账:

| 编号 | 症状(一句) | 根因 | 修法 |
|---|---|---|---|
| **E2** | actor init 撞 `No module tile_kernels` → spec=None | `qat.py`/`hyper_connection.py` 硬 import CUDA/SM90-only 的 tile_kernels,gfx950 装不了 | `per_token_cast_back` + 7 个 mhc 函数纯 torch autograd 重写(后换 Liger triton,详见正文「一」) |
| **E10** | sgl_kernel 缺 dsv4 op | uninstall+重编 base v0.5.14 会丢自带 DSv4 op | 不重编,用 base 自带 |
| **E11** | ray `not a valid Sentinel` | requirements 把 click 升到 8.4.2 破 ray 2.44 CLI | re-pin `click==8.2.1` |
| **E12** | indexer `No module deep_gemm`(PagedIndexerMetadata CUDA-only) | metadata 走 deep_gemm | `FP8_PAGED_MQA_LOGITS_TORCH=1`(metadata=None) |
| **E13** | topk_v2 `<cuda/ptx> not found` | topk JIT 走 cuda ptx | `USE_TOPK_V2=false` |
| **E14** | mhc `NameError deep_gemm` | MHC prenorm 走 deep_gemm | `DEEPGEMM_HC_PRENORM=false`+`TILELANG_MHC=false` → `aiter.ops.mhc` |
| **E15** | MoE down_proj 的 tp_invariant row-linear 在 fp8 tuple 上读 `.shape` 炸 | true_on_policy 的 `should_use_tp_invariant_row_linear` 不认 fp8 tuple | `Fp8LinearMethod` gate 短路(sglang fork,详见正文「上游 PR 盘点」第 2 项) |
| **E16** | shared_experts silu 掉到 CUDA `silu_and_mul_clamp`(`<cuda_fp8.h>` gfx950 编不了) | 旧 base aiter 缺 `fused_clamp_act_mul`(aiter PR # 3057 才引入) | 换 base aiter `7d604afe5`,`FUSED_CLAMP_ACT_MUL` 走默认 True(miles PR # 1506) |
| **E17** | cuda graph capture 在 ROCm colocate hang(GPU 0%) | aiter AR 的 IPC buffer 与 colocate memory_saver 冲突 | 关 cuda graph(colocate+ROCm gate);根因+解详见正文 R7 |
| **E18** | aiter config-merge FileBaton 死锁(colocate 8 engine 抢同一 baton) | `update_config_files` merge 多文件走 mp_lock | `AITER_CONFIG_GEMM_*`(6 个)+`_FMOE` 各指单文件(`len≤1` 短路 merge) |
| **E19** | rollout `Tensor match failed` @ `c4_v2.cuh`(kernel 期望 2D 收 3D) | compressor_v2 的 c4_v2 kernel | `USE_COMPRESSOR_V2=false` → v1 triton compress |
| **E20** | indexer paged-MQA-logits 在 ROCm 无可用 kernel | aiter gluon 需 triton≥3.6、aiter legacy 要 `block==1` 与 DSv4 page 不符 | 定案 torch fn `fp8_paged_mqa_logits_torch`(后 R1 升 triton 3.7 解锁 aiter 快路,详见正文 R1) |
| **E21** | aiter gluon 编译 `CDNA_VERSION is not in list` | gluon 需 triton≥3.6,容器 triton 3.4 | 归 E20 torch fn;R1 升 triton 3.7 才走 gluon |
| **E22** | `indexer.py` 无条件读 `c4_sparse_raw_indices` → AttributeError | 非 capture 路径本该 None | `getattr(...,None)`(commit `f46525c85a`,详见正文「五」第 2 坑) |
| **E23** | `_c128_decode_kernel` 的 `tl.load` 指针 (1,D) vs mask (BLOCK,D) | triton 3.4 不自动广播 | 补 `(slot_offs*0)[:,None]` 显式广播(commit `914dd3d93d`,详见正文「五」第 3 坑) |
| **E24** | train step 崩 `rollout_routed_experts is required` | Rust `sglang_router` v0.3.2 转发 /generate 时丢 `return_routed_experts` 字段(pin `openai-protocol` 1.0.0 无 flatten catch-all) | ROCm 走 miles python router `--use-miles-router`(is_hip gate,commit `b9047e3`) |
| **E25** | compute_log_prob 崩 `fuse_wgrad_accumulation not supported in ROCm blockwise grouped FP8` | TE MoE grouped linear 反向不支持 wgrad 融合累加 | `--no-gradient-accumulation-fusion`(is_hip gate,DDP hook 补 fp32 累加,详见正文「训练反向」第 2 点) |
| **E26** | compressor wkv `torch.mm(bf16,bf16,out=fp32)` ROCm hipblas 不支持 | hipblas 无 bf16-in/fp32-out 路径 | `_BFloat16LinearFP32Func` 加 hip guard 用 fp32 matmul(数值 ≥ bf16-in/fp32-accum) |

> E12/E13/E14/E18/E19/E20 那批 env 已固化进 `docker/Dockerfile.rocm`(rollout env 段)+ 训练脚本 `scripts/amd/run-deepseek-v4-flash-fp8-4layer-amd.sh`,不用手 export。
>
> ⚠️ **启动务必走 `scripts/amd/run-deepseek-v4-flash-fp8-4layer-amd.sh`,不能裸跑 `python scripts/run_deepseek_v4.py full-train …`**。ROCm 关键 env(`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`、`SGLANG_USE_AITER`、`SGLANG_HACK_FLASHMLA_BACKEND=triton`、`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`、`SGLANG_DSA_TOPK_BROADCAST=1`、AITER config pin 等)**只在这个 `.sh` 里 export、不在镜像**。裸跑漏掉 → sglang rollout 侧 `fp8_paged_mqa_logits_torch` 崩 `assert seq_lens.shape == (batch_size,)`(`sglang/srt/layers/attention/dsv4/indexer.py:69`);其中 `SGLANG_DSA_TOPK_BROADCAST=1` 见上文「attn_tp=4 下开 indexer」一节。(2026-07-08 实测:`docker exec` 里裸跑必踩,换回 `.sh` 即过。)

**bring-up 阶段显式不做**(防在 toy 上空转):不追 4-layer 收敛 / loss 下降 / eval 准确率(模型未训练,eval 无意义);ue8m0 vs fp32 scale 属 recipe 差异,先归因 recipe 再归因 kernel,别死磕修不存在的 bug。

---

## 附:多机 bring-up 操作手册(mgmt/RDMA/GID/rail/ray)

> 正文「三、集群与网络」讲了**为什么**要分两张网;这里是**照着能做**的操作详版:识别网卡 → 选 GID → GPU-NIC rail 映射 → 起容器/Ray/提交。一句话总纲:多节点卡脖子的常常不是模型,是底下那张网 —— **Ray 心跳(小而频)走管理网,NCCL/RCCL 大流量走 RDMA 网,搞混是多节点起不来最常见的原因**。

### 1. 分清管理网卡 vs RDMA 网卡

| 信号 | 管理网(L2 控制面) | RDMA(L3 数据面) |
|---|---|---|
| 网段 | `/24` 共享子网 | `/31` 点对点 |
| 路由 | `scope link`(直达) | `via <gateway>`(过路由) |
| 设备类别 | 普通 netdev | 列在 `/sys/class/infiniband/` 下 |
| 绑谁 | Ray gRPC 心跳 + job server | NCCL/RCCL all-reduce/all-gather |

管理网卡 = 带默认路由、你 SSH 用的那张(名字各样 `enx*`/`eth*`/`eno*`/`ens*`);RDMA 网卡 = `/sys/class/infiniband/` 下的设备(有 `ibdev2netdev` 可把 RDMA 设备对到 netdev)。配反的两种典型故障:Ray 误放 RDMA 网 → worker 几秒被判死(`raylet marked dead / missed heartbeats`,因 `/31` 无通用路由 + jumbo MTU 黑洞大 gRPC 包);NCCL 误放管理网 → 吞吐塌到 socket 以太网速度、远低于 RDMA 的 160–210 GB/s。

### 2. RDMA fabric 内部:厂商 / 传输 / GID

- **厂商看前缀**:`ionic_*`=AMD Pensando(我们的)、`bnxt_re*`=博通、`mlx5_*`=Mellanox/NVIDIA、`irdma*`=Intel、`erdma*`=阿里云。
- **传输**:端口 link layer 是 `Ethernet` = RoCE(RoCEv2 跑 UDP/IP、**可跨网段路由**,`/31` fabric 必须它,要显式选 GID);是 `InfiniBand` = 原生 IB(有子网管理器,不用选 GID、`NCCL_IB_GID_INDEX` 可不设)。我们是 RoCEv2。
- **GID 选择(RoCE only)**:`fe80::` 开头是 link-local、不可路由,**绝不能选**;要选**可路由的 RoCEv2** 条目(IPv4 fabric 上是 `::ffff:a.b.c.d`,内嵌点分十进制 = 网卡自己的 fabric IP)。口诀:**选「类型 RoCEv2、地址可路由(非 fe80::)」里号最小的 GID index** → 我们用 `NCCL_IB_GID_INDEX=1`。

### 3. GPU-NIC rail 映射:一 GPU 一 rail,排除多余网卡

原则:每张 GPU 配 **PCIe 上离它最近**的 RDMA 网卡(GPU Direct RDMA 走最短 PCIe 路)。两个坑:①PCIe 距离打平时自动配对会挑错;②机器 RDMA 网卡可能比 GPU 多,**多出来的那张(孤立在别的网段)不是数据 rail,必须排除**,否则会把一张 GPU 绑到管理/备用网。

典型实况(我们的机器):8 GPU、9 张 `ionic`,分三个网段 —— `ionic_0~3` 在 `172.33.2.x`、`ionic_5~8`(或对应 8 张)在 `172.65.2.x` 是 8 根 GPU rail;单独孤立在另一个 `10.x` 网段的那张(某机是 `ionic_4`、某机是 `ionic_6`,编号随机)是备用口 → **排除它**。GPU4 是易错点:PCIe 上离孤立网卡和相邻数据网卡等距,笨算法会默认挑到孤立那张,得手动纠回数据平面。最终:

```
# 排除孤立那张(下例排 ionic_6;按各机实际孤立编号调整)
NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_7,ionic_8
```

### 4. 起飞流程

**4.1 通道数 + 冒烟测试.** NCCL channel 卡在带宽曲线拐点:`ionic` 类在 **16–32 通道**稳,**64 通道易在建 queue-pair 时卡死**。`NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` 钉同值(16 保底、32 榨吞吐)。开大任务前先跨节点冒烟:所有 rank 跑小 `all_reduce` + `all_gather_object`,俩都过且带宽在预期范围 = RDMA 真通。

**4.2 代码/数据到位.** 每节点在**相同路径**看到相同镜像/代码/数据。共享 NFS(一份、读慢)或各节点本地一份(我们走本地 NVMe 存 fp8+torch_dist,因 NFS 只有 ~90MB/s)。

**4.3 起容器**(RDMA 三要点:相同容器名便于 `docker exec` 批量驱动、`--network host` 让 NCCL/Ray 看到物理网卡、逐个 `/dev/infiniband/uverbs*` 传进去 + `memlock=-1` 让 RDMA 能 pin 内存):

```bash
docker run -d --name <CONTAINER> \
  --network host --ipc host --shm-size 128g \
  --device /dev/kfd --device /dev/dri \
  $(for d in /dev/infiniband/*; do printf -- '--device=%s ' "$d"; done) \
  --group-add video --cap-add CAP_SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  <IMAGE> sleep infinity
```

> ⚠️ 高频初见坑:`docker run` 完 verbs 设备在、但 `ibv_devices` 列出**空**、报 `Driver ionic does not support the kernel ABI` = 容器内 RDMA 用户态 provider 跟宿主机内核 ABI 不匹配(**不是 NCCL 配错**)。解法:把容器内 provider 库换成宿主机那份(`rdma-core` 版本对得上)+ `ldconfig`。此修法**按容器**、重建即失效、每次 `docker run` 后重打。注:`--device /dev/infiniband`(传目录)不行,必须逐个 uverbs 设备。

**4.4 清 Ray 端口.** Ray 用 6379(GCS)、8265(job server);`--network host` 下这俩被本机所有容器共享,别的容器残留旧 Ray 会截胡。起 Ray 前先查空:

```bash
ss -ltnp | grep -E ':6379|:8265'
```

**4.5 在管理网上起 Ray.** 关键:所有 `--node-ip-address` 填**管理网 IP**(不是 fabric IP)—— 这是治 `raylet marked dead` 的决定性一招。env 在 `ray start` 时注入(actor 顺 raylet→worker→actor 链继承):`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`=管理网卡(握手/Gloo 走管理网,大流量走 `NCCL_IB_HCA`);`NCCL_IB_HCA`=第 3 节的数据 rail;`NCCL_IB_GID_INDEX`=第 2 节的 RoCEv2 号;`NCCL_MIN/MAX_NCHANNELS`=4.1 的通道数;`RAY_health_check_failure_threshold=30` / `period_ms=10000` / `timeout_ms=30000`(放宽心跳容忍,免 fabric 抖动误判死)。

```bash
# head 节点
docker exec -i \
  -e NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_7,ionic_8 \
  -e NCCL_IB_GID_INDEX=1 \
  -e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 \
  -e RAY_health_check_failure_threshold=30 -e RAY_health_check_period_ms=10000 -e RAY_health_check_timeout_ms=30000 \
  -e NCCL_SOCKET_IFNAME=<MGMT_IFACE> -e GLOO_SOCKET_IFNAME=<MGMT_IFACE> \
  <CONTAINER> bash -lc '
    ray stop --force; rm -rf /tmp/ray
    ray start --head --node-ip-address <HEAD_MGMT_IP> --port 6379 --num-gpus 8 \
      --dashboard-host 0.0.0.0 --dashboard-port 8265 --disable-usage-stats'

# worker 节点(同一组 -e env)
  <CONTAINER> bash -lc '
    ray stop --force; rm -rf /tmp/ray
    ray start --address=<HEAD_MGMT_IP>:6379 --node-ip-address <WORKER_MGMT_IP> --num-gpus 8 --disable-usage-stats'
```

起完 `ray status` 验:所有节点在、GPU 数对、无 dead。实践中已固化成 `dsv4_env.sh`(MASTER_ADDR=head 管理 IP、RoCE env、通道 16、Ray 心跳放宽)+ `dsv4_ray_head.sh` / `dsv4_ray_worker.sh`(worker 自动探测本机管理 IP)三个脚本,每台一句 `docker exec <container> bash .../dsv4_ray_{head,worker}.sh` 即可。

**4.6 提交训练**(只在 head 节点):设 `MILES_SCRIPT_EXTERNAL_RAY=1`(挂到已有 Ray 集群、别自己另起)+ `WANDB_API_KEY`,跑 `run_deepseek_v4.py`(或对应 `run_*.py`)。**健康序列**:引擎 ready + `/health` 200 → Megatron 加载 torch_dist ckpt + Gloo 各 rank 连上 → `update_weights` 成功 → rollout 出 reward → **第一个 step 的 `grad_norm` 出现(= 跨节点 all-reduce 真跑通)** → `Job succeeded`。

**踩坑速记**:①`pkill -f "docker save ..."` 会自匹配杀掉自己的 shell → 用数字 PID 或 `pkill -x`;②重建容器丢掉运行时的 wip 分支 + liger + 并行档(新容器 = image 的 main 分支)→ 容器内 setup 必须重做;③设备传进去 ≠ 能用,`ibv_devices` 空 = provider/ABI 不匹配(见 4.3)。

---

## Sources

[aiter](https://github.com/ROCm/aiter)(roadmap # 3442,wedge # 2061 / fix # 2075)· [sglang # 19162](https://github.com/sgl-project/sglang/pull/19162) / # 20155(memory-saver AR fix)· [sglang # 23581](https://github.com/sgl-project/sglang/issues/23581)(SGLANG_USE_AITER_AR)· [# 24011](https://github.com/sgl-project/sglang/issues/24011)(ROCm hipEventQuery capture bug)· [# 29657](https://github.com/sgl-project/sglang/pull/29657)(DSV4 piecewise cuda graph WIP)· [# 29985](https://github.com/sgl-project/sglang/pull/29985)(DSV4 capture-OOM)· [# 29144](https://github.com/sgl-project/sglang/issues/29144)(DSv4-Flash spec-decode DP-attn deadlock)· [# 28685](https://github.com/sgl-project/sglang/issues/28685)(gfx950 blockscale 数值坑)· [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)· [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)· [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)· [ROCm MLPerf v6.0](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.0/README.html)· [NVFP4 预训练 arXiv 2509.25149](https://arxiv.org/abs/2509.25149)· [HipKittens](https://github.com/HazyResearch/HipKittens)· [LMSYS DSv4 Day0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)

<!-- v4:真正合并去重版。narrative(一~七)+ 耗时 + 反向/优化器 + recipe + NV 对比 + kernel 对照 + kernel landscape + 优化路线图(R1-R10 实测/待办,含 cuda-graph wedge 根因)+ 4-layer bring-up 报错台账(E2..E26,从 dsv4-rocm-fp8-bringup.md 提炼去重合并)+ 多机 bring-up 操作手册 + Sources。删:ledger OPT-R* 逐轮流水账、loop 纪律/心跳、单次 job id、node-6 wedge 运维细节、PLAYBOOK 与正文三节重复的概念叙述、3node 一次性状态;bringup 的 overnight 任务/状态、逐轮 train{N} 过程、产物路径大小等 ephemeral。 -->
