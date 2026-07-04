# 在 AMD MI355X 上把 DeepSeek-V4-Flash 的 FP8 强化学习训练跑起来

一次工程记录:从"让模型在 AMD 卡上能加载",到 4 台机器 32 张卡把 291B 的 DeepSeek-V4-Flash 全量 FP8 RL 训练稳定跑起来。

**结果**:4 节点 × 8 卡 = 32 GPU(AMD MI355X / gfx950),全量 DeepSeek-V4-Flash(291B、43 层 MoE、DSA 稀疏注意力),FP8 训练 + FP8 rollout 的 GRPO 强化学习,**已连续稳定跑过 8 步,零崩溃,优化器保持 fp32**,wandb 全程追踪。

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
- **连续 8 步稳定,零崩溃**:grad_norm 一直健康(0.08~0.17);reward 围 0.5 波动、还看不出学习趋势(lr 只有 1e-6,是 bring-up 的保守设定,早期正常);eval 的 AIME ~0.44——这个数被 eval 的 4096 token 上限压低了(57% 的题没写完就被截断)。

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

## 待办 / 优化方向

### rollout 加速(rollout 是瓶颈:占一步约 60%、生成吞吐仅 ~250 token/s/引擎)

**前置**:升级容器 triton(现在 3.4,锁住了一批 aiter gluon kernel)。**动之前先把当前能跑的 image push 备份**,再尝试下面这些。

kernel 审计发现的 5 处慢路径:

1.【最高】DSA paged-MQA indexer 退回 legacy page_size=1(rollout decode 热点)
   - 根因:容器 triton 3.4.0 太老,aiter gluon 那套优化 kernel 全要 ≥3.5/3.6。日志原话:`aiter preshuffle paged-MQA path unavailable (needs Triton>=3.5.0); falling back to legacy page_size=1`。
   - 解锁:升级容器 triton,或一个 env `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`。这是"一开解锁一批 aiter kernel"的高杠杆项。
   - 注意:indexer 的 logits 计算本身已经走 tilelang(优化);慢的是 KV 分页取数那条独立路径。
2.【高·零成本】enable_fused_qk_norm_rope 被关着
   - 日志:`aiter fused_qk_norm_mrope_3d kernel available`,但 flag = False。kernel 就位、纯没开 = 白捡的融合加速。确认精度对齐就能开。
3.【高】gfx950 的 tuned config CSV 缺
   - bf16_tuned_gemm.csv 缺形状 → 915 次投影 GEMM 退回 torch;tuned_fmoe.csv 让 fused_moe 走 2stage default 没调优。补 CSV 就提速,纯 tuning、低风险。
4.【中】sparse MLA attention 后端:现在 triton,代码里 tilelang / 原生 FlashMLA 两条路都在但没 tune gfx950 → 值得 A/B(和记忆里"FlashMLA gfx950 tune 5-8%"对得上)。
5.【中】训练侧 DSv4 自定义 RoPE = naive torch(view_as_complex fp32 复数乘,每层多次、无融合)。没有现成开关,得自写 triton/aiter 融合核,有开发成本。

**顺手清理**(下次重发时,都是 NO-OP/死开关,不影响正确性):
- 去掉 `--sglang-dsa-topk-backend torch`(对 DSv4 是 NO-OP,topk 走原生 HIP kernel)
- 删环境变量 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`(死开关 footgun)

**其他 rollout 加速**:
- [ ] 开 cuda graph(现在为省显存关着)
- [ ] MTP 投机解码(现在关着;ROCm 上此前测过 ~2.55×)
- [ ] async / 训推分离,让 rollout 和训练重叠

### 扩展 / 根本解
- [ ] 上 8 节点官方配置(PP8),一次拿回所有 headroom
- [ ] 用长 token 预算(如 32768)重测 eval,看 V4 真实 AIME 分数
- [ ] 重开 indexer-replay(改 rollout 为 attn_tp=1 对齐 NV;见 task #16)

### 其他
- [ ] 在训练节点上正经复测跨节点 NCCL 带宽 + 查是否用满所有 rail(卡内已确认 392 GB/s;跨节点早先 46 GB/s 疑似偏低,空闲节点测不通)
- [ ] 再试一次 NFS(当时 90MB/s 太慢,换了本地 NVMe)
- [ ] 上游 PR 盘点(待逐个确认是否已在上游):
  1. **triton 3.4 广播 fix**(E23,`_c128_decode_kernel`)—— 通用兼容修复,最该提 → sglang `amd/dsv4-rebase`
  2. **fp8 row-linear tp_invariant gate**(E15)—— 通用,`layers/linear.py`(部分可能已在上游 sglang-miles 线)
  3. **Megatron SwiGLU fused clamp**(`262b82419`)—— DSv4 激活融合,纯增量、NV 不受影响 → Megatron 上游
  4. **indexer-replay TP capture**(去 `attn_tp==1` assert + rank0 广播 top-k)—— 让 DSA R3 支持注意力 TP,通用价值 → sglang(diff 已备份 `docs/sglang-indexer-topk-tp-capture.patch`)
  5. E22 getattr 保护 —— 小修,可随 E23 一起
  - 注:offload-fraction 0.75 那类是"配方调参"不是代码 PR,整理成一篇"4 节点显存平衡"经验更合适。

---
<!-- v3:blog 质量,吸收 v1 全部批注。仍可补:各阶段配图、更细的 4-layer 数据、每个 PR 的最终去向。 -->
