# DSv4-4layer FP8 训练 bring-up —— 报错落地

目标:在 AMD MI355X / ROCm 上把 DeepSeek-V4-Flash 4-layer 缩小版(pipeline-only sanity check)的 **FP8 训练路径**跑通,验证 `fp8_training=True` 端到端。属于「支持 fp8 train」任务,过程报错全部落地、进最终 PR。

## 环境

| 项 | 值 |
|---|---|
| 镜像(当前) | `xinyujiangcmu/miles:rocm700-mi35x-sglang0.5.14-20260627`(base `rocm/sgl-dev:v0.5.14`, ROCm 7.0, gfx950),容器 `dsv4-fp8-v14`。**早期 E2-E9 在旧 base** `...-dev-20260627`(v0.5.10) |
| miles | 容器内 `radixark/main`(Dockerfile clone);**实验改动记录在** `XinyuJiangCMU/miles@wip/dsv4-rocm-tile-kernels-import-guard`(Dockerfile ENV + run_deepseek_v4) |
| TE | `2.14.0.dev0+7924deca`(FP8 feature 分支) |
| torch | `2.9.0a0`,8× MI355X |
| 模型 | `Pinaster/DeepSeek-V4-Flash-FP8-4layer`(28.65 GB,6 个 safetensors) |

命令:
```
python scripts/run_deepseek_v4.py full-train \
  --model-name DeepSeek-V4-Flash-FP8-4layer \
  --hf-checkpoint /workspace/models/DeepSeek-V4-Flash-FP8-4layer \
  --model-dir /workspace/models --data-dir /workspace/datasets \
  --num-nodes 1 --num-gpus-per-node 8
```

## 核心结论(TL;DR,2026-06-28)

**DSv4-Flash-FP8 在 ROCm/gfx950 跑通 rollout,本质只两件事,不是十几个独立 bug:**

1. **build 配方**(`docker/Dockerfile.rocm`):`FROM rocm/sgl-dev:v0.5.14`(base 自带 dsv4 op,**关键是别 uninstall+重编 sgl_kernel**)+ 末尾 re-pin `click==8.2.1`(ray 2.44 CLI)+ TE blockwise-fp8 分支。
2. **一套 rollout env**(Dockerfile ENV,镜像级):把 DSv4 的 sparse-attn indexer / MHC / MoE 从 CUDA-only(deep_gemm / `<cuda/ptx>` / tilelang / `<cuda_fp8.h>`)全切到 **aiter/triton**。**这套 env 直接照搬 sglang AMD nightly CI 的 `test/registered/amd/test_deepseek_v4_flash_fp8.py` 的 `COMMON_ENV_VARS`**(权威全集)。

| env | 绕开的 CUDA-only | 早期逐个撞的 error |
|---|---|---|
| `SGLANG_OPT_USE_AITER_INDEXER=true` + `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` | indexer metadata+compute → aiter/torch | E3 / E12 |
| `SGLANG_OPT_USE_TOPK_V2=false` + `USE_JIT_INDEXER_METADATA=false` | 跳过 `<cuda/ptx>` topk_v2 JIT | E5 / E13 |
| `SGLANG_OPT_DEEPGEMM_HC_PRENORM=false` + `TILELANG_MHC_PRE/POST=false` | MHC → `aiter.ops.mhc` | E6 / E14 |
| `SGLANG_OPT_USE_FUSED_COMPRESS=true` + `_TRITON=true` | MoE 压缩/激活 → triton | (E15/E16 由此避开) |
| `HACK_FLASHMLA_BACKEND=triton` / `USE_TILELANG_INDEXER=false` / `MULTI_STREAM=false×2` / `AITER_BF16_FP8_MOE_BOUND=0` / `FP8_WO_A_GEMM=false` | 其余 ROCm 路径 | — |

**教训:** 早期 E2-E16 逐个撞,是因为没先看 AMD test 的 `COMMON_ENV_VARS` —— 照搬这套 env 即可,不必逐个试。我中途的 `SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL=0`(E15/E16)是**歧路**(官方靠 `FUSED_COMPRESS` 走 triton,不需要它),已从 Dockerfile 移除。

> 下面 E2-E16 详细探路记录**保留**(每个 error 的根因/调用栈/源码行),作 PR 过程证据;但**最终方案以本结论 + Dockerfile ENV(= AMD test COMMON_ENV_VARS)为准**。actor 侧(miles Megatron MHC/quant via TileKernels,E2)走自己路径、不经 sglang env,待 rollout 全绿后暴露。

## 流水线进展

**当前(新 base v0.5.14)** rollout 链:E10✅(dsv4 op,base 自带)→E11✅(click==8.2.1)→E12✅(aiter indexer)→E13✅(topk_v2=0)→E14✅(aiter mhc)→**E15 MoE down_proj 收 tuple ❌(诊断中)**→ actor mhc/quant(E2,未到达)。

> 早期(旧 base v0.5.10,已被新 base 取代)链:cast✅→torch_dist✅(E2 stub)→deep_gemm✅(E3 torch env)→topk_v2❌(E5)→mhc(E6)。下面 E2/E3/E5/E6 段保留作探路记录,**结论已被新 base + aiter 路线更正**(见各段「状态(更正)」与末尾「配置统一」)。

## E2 —— BF16→torch_dist 失败:缺 `deepseek-ai/TileKernels`(OPEN blocker,ROCm)

- **症状:** `tools/convert_hf_to_torch_dist.py`(`torchrun --nproc-per-node 4`)在所有 rank 上失败:`Exception: specialize for NoneType.`,位置 `megatron/core/transformer/transformer_block.py:326`。
- **前置信号:** `spec_utils.py:55 - couldn't import module due to No module named 'tile_kernels'` → DSv4 的 module spec 被静默置为 `None`。
- **根因:** DSv4 plugin **硬 import** `tile_kernels`(= `deepseek-ai/TileKernels`,见 `hyper_connection.py` / `qat.py` 的 docstring),无 try/except fallback,且 spec 加载必经此路:
  - `miles_plugins/models/deepseek_v4/ops/qat.py:3` → `from tile_kernels.quant import per_token_cast_back`
  - `miles_plugins/models/deepseek_v4/ops/hyper_connection.py:18` → `from tile_kernels.modeling.mhc.ops import ...`
  - `deepseek_v4.py:36` import `qat` → 触发失败的 import。
- **失败仅在 import 阶段:** `convert_hf_to_torch_dist.py` 只做 `get_model`(建结构) → `load_weights` → `save_checkpoint`,**不跑 forward**;mhc/quant kernel 运行时不被调用。失败 100% 来自建模型结构时 spec 加载的顶层 import。→ 让 import 通过即可过 torch_dist 这步;但**训练 step 会真调** mhc/quant,需 ROCm 上能跑的真实现。
- **ROCm/CUDA(关键,已查仓):** `deepseek-ai/TileKernels` 仓**明确 CUDA/NVIDIA-only** —— 要求 NVIDIA SM90/SM100(Hopper/Blackwell),pip 包 `tile-kernels`,建在 tilelang 上,零 AMD/ROCm/HIP 提及;镜像里 `tilelang` 也是 `+cuda` build。**在 gfx950 上装它是死路。**
- **两处用途(替代范围):** ① `qat.py` 仅借 `per_token_cast_back`(FP8 反量化;量化侧 `act_quant` 已是 in-tree)。② `hyper_connection.py` 借整套 `mhc_*`(Multi-Hyper-Connection 的 pre/post/head mix + sinkhorn)。
- **ROCm 生态调研(已查 aiter/sglang upstream,2026-06-27):** 真正的 ROCm mhc 在 **`aiter`** 而非 sglang 本体,但**全是推理 forward-only,无一带 backward**,训练用不了:
  - `ROCm/aiter` **main 已有** `aiter/ops/triton/fusions/mhc.py`(另有 `dev-mhc` 开发分支)。入口 `mhc()/mhc_post()/mhc_post_pre()`,**纯 forward**(无 `torch.autograd.Function`/`*_bwd`/`.backward`),推理 token 布局 `(M,n,C)`、`M≤64`。容器内 aiter(`@a6bb499`,4/29)**还没这文件** → sglang 优雅 fallback。
  - sglang 主 `srt/layers/mhc.py` 是 **tilelang** kernel(容器 tilelang=`0.1.7+cuda` build,gfx950 无效);AMD 路 `deepseek_common/amd/deepseek_v4_fused_mhc.py` 只是 try-import 上面的 aiter mhc,失败即 fallback。容器装的是 **`sglang-miles` 分支 @ `63e5bb2c`**(sgl-project 下有 `sglang-miles`、`sglang-miles-v0.5.8…v0.5.13` 一整排)。
  - **推论:** 即便把容器 aiter 升到 main,也只让 **sglang rollout(推理)** 用上 fused mhc,对 **miles 训练 backward 毫无帮助**。"等上游/升级 aiter"对训练这条路**判死**。
  - **数学参考已落地 `sglang-ref/`:** sglang tilelang `hc_split_sinkhorn_kernel`(pre/post/comb + sinkhorn 公式逐行清晰)+ aiter `mhc.py`,供纯 torch 重写时对齐数值语义。
- **方向(已查):** legacy in-tree fallback **已删** —— `_HYPER_CONNECTION_MIXER_NO_GRAD`、`hc_split_sinkhorn` 仅存于 docstring,无任何实现。唯一路 = 纯 torch 重写两处:① `per_token_cast_back`(FP8 反量化,简单几行);② 一整套 `mhc_*`(7 个函数:`mhc_pre_norm_fn`/`mhc_pre_split_mixes`/`sinkhorn_normalize`/`mhc_pre_apply_mix`/`mhc_post`/`mhc_head_compute_mix`/`mhc_pre_big_fuse`,较难,需准确复现 MHC 语义)。**分层:** torch_dist 只需 import 不炸(可先 stub 过);训练 step 才真调 mhc/quant,需真实现。
- **状态:** torch_dist 已用 stub 解除;训练 forward 的 mhc/quant 真实现仍 OPEN。
- **stub 处置(探路,A 方案):** `qat.py`/`hyper_connection.py` 的 `from tile_kernels...` 包成 `try/except ImportError` + stub(抛 `NotImplementedError`,见 `patches/`)。让 import 过 → **torch_dist ✅(52G 产物)**。训练真调 mhc/quant 时会抛 stub 错,届时暴露具体函数与调用路径。

## E3 —— 训练 rollout 失败:sglang DSv4 缺 `deep_gemm`(有 ROCm 开关,未启用)

- **症状:** torch_dist 后进训练,**SGLang rollout 引擎** capture cuda graph 时炸:`sglang/srt/layers/attention/dsv4/metadata.py:113 __post_init__` → `ModuleNotFoundError: No module named 'deep_gemm'`(经 `deepseek_v4_backend_hip_radix.py`)。
- **根因:** `deep_gemm`(= `deepseek-ai/DeepGEMM`,CUDA-only FP8 GEMM)是 DSv4 sparse-attention indexer 默认路径,镜像未装。
- **好消息(有现成 ROCm 开关):** `metadata.py:107` 分支 —— 设 `SGLANG_OPT_USE_AITER_INDEXER`(走 AMD aiter)或 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`(纯 torch)任一 → `deep_gemm_metadata=None`,不 import deep_gemm。两者默认 `EnvBool(False)`(`environ.py:761/765`)。
- **注入点:** `run_deepseek_v4.py:494` 的 `extra_env_vars` dict(已塞多个 `SGLANG_*`,line 585 传给 rollout)。加一行即可。
- **状态(更正,2026-06-28):** 旧 base 时用 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` 探路过,**现已被 aiter 取代**。原因:该 torch env 只 null metadata 路径(`metadata.py:108`),但 `indexer.py:529` 计算仍 `from deep_gemm import fp8_paged_mqa_logits`,**不完整**。当前统一 `SGLANG_OPT_USE_AITER_INDEXER=1`(metadata + compute 全走 aiter triton,见 E12 段 + 末尾「配置统一」)。这条复现在新 base = **E12**。

## E5 —— rollout topk_v2 JIT kernel 在 gfx950 编译失败(sglang HIP 适配不全)

- **症状:** 绕过 deep_gemm 后,rollout 引擎 capture cuda graph 时 JIT 编译 DSv4 topk_v2 kernel 失败:`ninja exited status 1`;`include/sgl_kernel/deepseek_v4/topk/ptx.cuh:4: fatal error: 'cuda/ptx' file not found`(`train4.log`)。
- **根因:** `ptx.cuh` include CUDA-only `<cuda/ptx>`,用 `cuda::ptx::mbarrier_*`(Hopper TMA/async-barrier PTX intrinsics),hipcc/gfx950 无对应。
- **sglang HIP 适配不全:** `jit_kernel/dsv4/topk.py` 的 `topk_transform_512` **有** HIP 分支(`if is_hip_runtime(): 走 topk_v1`),但 `plan_topk_v2`/`topk_transform_512_v2` **无条件走 v2**(ptx.cuh)→ HIP 下无 fallback;backend 调到 v2 路径 → 炸。
- **绕过(逐步实测):**
  - `--sglang-disable-cuda-graph` **无效**(train5 仍炸)。
  - `--sglang-dsa-topk-backend torch` **不够**(train6 仍炸):它只管 `topk_func`(decode 路径),但真正触发点是 **`metadata.py:138 __post_init__` 里 `plan_topk_v2(c4_seq_lens)`**(rollout prefill metadata 构造时无条件编 v2,经 `deepseek_v4_backend_hip_radix.py:398 init_forward_metadata_indexer`)。
  - **真正开关 = `SGLANG_OPT_USE_TOPK_V2=0`**:metadata.py:138 有门控 `if SGLANG_OPT_USE_TOPK_V2: plan_topk_v2 else: topk_metadata=torch.empty((0,))`。关掉 → 不编 v2,下游走 `topk_transform_512`(topk.py:52 `is_hip_runtime` 有 **HIP 预编译 op** `torch.ops.sgl_kernel.deepseek_v4_topk_transform_512`,不 JIT)。
- **状态:** 加 `SGLANG_OPT_USE_TOPK_V2=0` 重试(train7)。

## E6 —— sglang rollout MHC forward 用 deep_gemm(CUDA-only),无 ROCm 路径

- **症状:** topk_v2 绕过后进 model forward,`models/deepseek_v4.py:1386 forward → hc_pre(1244) → layers/mhc.py:686 mhc_pre → deep_gemm_wrapper/entrypoint.py:204 tf32_hc_prenorm_gemm → deep_gemm.tf32_hc_prenorm_gemm` → `NameError: deep_gemm not defined`(`train7.log`)。
- **根因:** `entrypoint.py:19 if ENABLE_JIT_DEEPGEMM: import deep_gemm`(ROCm 下 False);但 `tf32_hc_prenorm_gemm`/`bf16_gemm_nt` 等无条件用 `deep_gemm.*`,无 fallback。
- **两条路都 CUDA:** `mhc.py:674 mhc_pre` → `if SGLANG_OPT_DEEPGEMM_HC_PRENORM: deep_gemm`(炸) `else: tilelang kernel`(容器 tilelang=`+cuda`,gfx950 无效)。aiter 旧版无 mhc。**无 env 绕过。**
- **= actor E2 同一 MHC 数学的两个实现:** rollout(sglang tilelang/deep_gemm) + actor(miles TileKernels)。**需统一 torch MHC 重写**(pre/post/head + sinkhorn)。数学参考已落 `sglang-ref/`(`hc_split_sinkhorn` + `post_mix`/`comb_mix` 公式逐行)。
- **状态(更正,2026-06-28):** 旧 base 当时判"需 torch MHC 重写 / 等上游判死"是**错的** —— 那时容器 aiter 旧版没 mhc。**新 base v0.5.14 的 aiter 自带 mhc**(`aiter.ops.mhc`),rollout 侧 MHC 设 `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=0` 即走 aiter 路径(`deepseek_v4.py:1258`),**已验证解**(= 新 base 的 **E14**)。仅 **actor 侧** MHC(miles Megatron 经 TileKernels,= E2)走自己路径、不经 sglang/aiter,仍待训练真到 actor forward 时处理。

## E4(预告)—— blockwise FP8 weight-update + ROCm wgrad,DSv4 也吃(miles fp8 pin-set)

来自 qwen3 blockwise fp8 的在飞 miles PR(见 manifest fp8 pin-set 的 miles 条),两处对 DSv4 **也必需**,不止 qwen3:

- **`update_weight/update_weight_from_tensor.py`**:`update_weights()` 的 post_process 条件 `["compressed-tensors"]` → 加 `"fp8"`。DSv4-Flash-FP8 的 `config.json` 确认 `quant_method="fp8"`(blockwise [128,128]、e4m3/ue8m0)→ **命中**;否则 RL 训练 actor→rollout 的 fp8 权重同步跳过 reshuffle/cast,rollout 权重错。
- **`--no-gradient-accumulation-fusion`**(注释:ROCm 无 wgrad fusion)。`run_deepseek_v4.py` 当前**没有** → DSv4 训练 backward 大概率也要加。
- → 真跑 DSv4 fp8 train,这条 miles PR 的核心改动要 pin/移植进来(weight-update 那行是 model-agnostic)。

## base image 升级评估(v0.5.10 → v0.5.14, 2026-06-27)

当前 base `lmsysorg/sglang:v0.5.10-rocm720-mi35x`(5月)→ 拟升 `rocm/sgl-dev:v0.5.14-rocm700-mi35x-20260627`(6/27)。静态体检(docker run 探文件,未挂 GPU):

| 项 | v0.5.10(当前 hai-1) | v0.5.14 | 影响 |
|---|---|---|---|
| aiter | v0.1.12.post1(4/29) | **v0.1.14-rc0(6月)** | ⬆️ 唯一实质收益 |
| ├ fusions/mhc.py | ❌ 无 | ✅ 有 | 补 rollout 侧 mhc |
| └ pa_mqa_logits | ✅ | ✅ | E3 indexer 两版都行 |
| torch / ROCm | 2.9.0a0 / 7.0 | 同 / 7.0 | 无变化,驱动兼容、无升级风险 |
| tilelang | 0.1.7+cuda | 同 | 无变化(都 cuda,sglang 主 mhc 路无效) |
| deep_gemm | ❌ | ❌ | 都没有(本就靠 aiter indexer 绕) |
| sglang(base) | — | main(build 时仍切 sglang-miles) | 无影响 |

- **结论:值得升、低风险。** 实质变化就一条:aiter 4月→6月,多出 triton mhc kernel;其它(torch / ROCm 7.0 / tilelang / sglang-miles / deep_gemm)全不变或不受影响,无倒退面。
- **与 E6 直接相关:** E6 的 rollout MHC 无 ROCm 路,根因正是旧 aiter 没 mhc。新 aiter 的 mhc 就是 sglang `deepseek_v4_fused_mhc.py` try-import 的那个 → **升级 base 后 rollout 侧 MHC 可能自动有救**(actor 侧 E2/E6 仍需 torch 重写,因 actor 走 Megatron 不经 aiter)。
- **改动:** `docker/Dockerfile.rocm` 的 `ARG SGLANG_IMAGE_TAG` + `docker/build.py` rocm-mi350 的 `SGLANG_IMAGE_TAG` → `v0.5.14-rocm700-mi35x-20260627`(追加进 dockerfile-aiter 那个 PR)。
- **caveat:** ① `v0.1.14-rc0` 是 rc;② aiter mhc/indexer 的实际 import + 数值要 **build 后带 GPU** 才能验(docker run 无 GPU 验不了);③ `build.py` rocm-mi350 的 `tag_postfix` 仍是 `-rocm720-mi35x`,与新 base 的 rocm700 命名不一致,待定是否同步改。

## 已保留产物(在 mount 上,容器重启不丢)

- checkpoint:`/mnt/data/data/hai/models/DeepSeek-V4-Flash-FP8-4layer`(27 G)
- bf16 cast:`/mnt/data/data/hai/models/DeepSeek-V4-Flash-FP8-4layer-bf16`(52 G)
- torch_dist:`/mnt/data/data/hai/models/DeepSeek-V4-Flash-FP8-4layer_torch_dist`(52 G,E2 stub 后生成)
- 日志:`train.log`(E1)、`train2.log`(E2)、`train3.log`(torch_dist ✅→E3);stub patches:`patches/`;sglang/aiter 上游参考代码:`sglang-ref/`(sglang mhc tilelang + aiter mhc + amd wrapper)

## 待决策

- ~~**E3:** aiter 还是 torch~~ → **已决:aiter**(`SGLANG_OPT_USE_AITER_INDEXER=1`,见「配置统一」)。
- ~~**E6:** rollout MHC 需 torch 重写~~ → **已解:aiter mhc**(新 base,见 E14)。
- **E2(actor 侧 MHC/quant):** rollout 侧已用 aiter 解;**actor 侧**(miles Megatron 经 TileKernels)仍待 —— 需 rollout 全绿、训练真到 actor forward 才暴露具体调用与是否需 torch/aiter 实现。
- **E15:** MoE down_proj 收到 tuple(疑 fused clamp/act 的 ROCm 路径),诊断中。
- **整体:** rollout 侧 ROCm blocker 正逐个清(E10→E15,均 env/backend 绕,优先 aiter/triton);actor 侧(Megatron fp8 + MHC)待 rollout 通后暴露。所有验证过的改动 → `wip` 分支 Dockerfile ENV + run_deepseek_v4。

## 重大认知更新(2026-06-27 晚):E10 真相 + NV 对照 + 正确 build 配方

**E10 真相 = build 问题,非源码缺失。** main 与 sglang-miles 的 v4 算子源码一模一样(没文件可搬);dsv4 op 源码在、CMake 注册在,但 sglang-miles 重 build sgl_kernel 时**没编进 .so**(build 阶段问题,疑似类似 silu 的 ROCm 编译错被静默跳过)。**关键:base `rocm/sgl-dev:v0.5.14` 预装的 `sgl_kernel.so` 本就含 dsv4 op**(AMD CI 编好,实测 `dsv4_fused_q_indexer_rope_hadamard_quant`+`deepseek_v4_topk_transform_512` 都 True);是 Dockerfile `pip uninstall sgl_kernel` + 从 sglang-miles 重编**自废武功**。

**NV 怎么跑(`docker/Dockerfile`,对照 ROCm 缺什么)。** NV 上 DSv4 fp8 train 是原生主路径,E1-E10 几乎不存在:`pip install tile_kernels==1.0.0`(E2 是 pip 包);base NV sglang 自带 sgl_kernel dsv4 op(E5/E10);deep_gemm / silu(`cuda_fp8.h`+warp32) / mhc(tilelang) 全 NV 原生;**NV sglang 段只 fetch sglang-miles 装 python 层、不重编 sgl_kernel(用 base 的)、不打 megatron/miles patch**。→ ROCm 的 enablement = 把这列 CUDA 组件逐个换 ROCm 等价(aiter/triton/torch/补 op build);E1-E10 不是做错,是 DSv4 ROCm 移植本没完成。

**正确 build 配方(对齐 NV,增量、不重编 sgl_kernel)** — `miles-wip/docker/Dockerfile.rocm`:
- `FROM rocm/sgl-dev:v0.5.14-rocm700-mi35x-20260627`(base 自带 dsv4 op → 解 E5/E10)
- sglang 段:`fetch sglang-miles + pip install -e python[all_hip]`(只装 python=RL 改动),**删掉 `pip uninstall sgl_kernel` + `setup_rocm.py install` 重编**(那会丢 dsv4 op=E10)→ sgl_kernel=base, python=sglang-miles
- TE:`JessicaJiang-123/TransformerEngine@amd-qwen3-30b-a3b-fp8-dev`(编译)
- miles:暂 clone radixark/main(wip 的 DSv4 fix 未进 → 会撞 E2)
- **核心原则:base 已 ROCm 化好的部分(sgl_kernel dsv4 op)别被重编搞坏 = 增量 build**
- build(写死、无 --build-arg):`cd miles-wip && docker build -f docker/Dockerfile.rocm -t miles-rocm700-mi35x-sglang0.5.14-20260627 .`
- 预期:base v0.5.14 自带解 E5/E10 + E6(aiter triton mhc);仍撞 E2(miles radixark 硬 import tile_kernels,ROCm 无该 CUDA 包)+ E8/E9(sglang-miles silu JIT 无 ROCm fix)。

## 进展(2026-06-27 深夜):v0.5.14 镜像跑起来,E10/E11 解,推进到 E12(deep_gemm)

**镜像 build 成功 + push dockerhub**:`xinyujiangcmu/miles:rocm700-mi35x-sglang0.5.14-20260627`(67.9G)。容器 `dsv4-fp8-v14`(8 卡,新镜像)。配方见上「正确 build 配方」+ sglang 段 `git reset --hard`(清 base pyproject local 改动)。

**E10 解(运行时实锤)**:rollout indexer **跑过了 `dsv4_fused_q_indexer_rope_hadamard_quant`**(不再 `AttributeError: sgl_kernel has no attribute ...`),一路推进到 `init_forward_metadata_indexer` 才撞下一个。证明 base v0.5.14 的 sgl_kernel.so 自带 dsv4 op、我们不重编保住了它。静态也验过 `hasattr(torch.ops.sgl_kernel,'dsv4_fused_q_indexer_rope_hadamard_quant')`=True。

**E11 解** ray start `ValueError: not a valid Sentinel`:根因 click 被后续 pip 升到 8.4.2(破 ray 2.44.1 CLI 的 Sentinel API)。夹击:降 click 8.1.8 ray 好但 typer 0.25.1 炸(`click.Choice[...]` generic 需 ≥8.2)。**甜点 click==8.2.1**(ray+typer 都 OK)。修:Dockerfile 末尾(所有 pip 之后)re-pin `pip install "click==8.2.1"`(第 119 行虽 pin 了但被后续 pip 覆盖)。验证:ray 起、2 engine 起、模型加载。

**E12(新,绕法已知)** rollout 在 cuda graph capture 撞 `ModuleNotFoundError: No module named 'deep_gemm'`:位置 `dsv4/metadata.py:113 PagedIndexerMetadata.__post_init__`,deep_gemm 是 CUDA-only ROCm 没有。**该 __post_init__ 自带 env 开关**:设 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` 或 `SGLANG_OPT_USE_AITER_INDEXER=1` → `deep_gemm_metadata=None` 不 import deep_gemm。**这正是 wip 分支的 E3 fix**(SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 在 run_deepseek_v4 extra_env_vars);容器 miles=radixark/main 没带 → 复现。修:把该 env 传给 rollout SGLangEngine(run_deepseek_v4 extra_env_vars,或用 wip run_deepseek_v4)。

## 进展(2026-06-28 凌晨):E13 解,推进到 E14(mhc deep_gemm)

**E13 解(验证)** `cuda/ptx file not found`(ninja gfx950) JIT 编 topk_v2 kernel:位置 `dsv4/metadata.py:138 PagedIndexerMetadata.__post_init__` 的 `plan_topk_v2`(JIT `_jit_topk_v2_module`,头文件 include CUDA-only `cuda/ptx`)。`SGLANG_OPT_USE_TOPK_V2`(默认 True,environ:795)控制:`=0` 则 `topk_metadata=torch.empty(0)` 跳过 JIT。设 `SGLANG_OPT_USE_TOPK_V2=0` 后 train16 capture 推进过 indexer(cuda/ptx + deep_gemm 都不再撞)。

**E14(新,绕法已知)** cuda graph capture 撞 `NameError: name 'deep_gemm' is not defined`:`deepseek_v4.py:1244 hc_pre → mhc.py:686 mhc_pre`(走 `if SGLANG_OPT_DEEPGEMM_HC_PRENORM` 分支)→ `deep_gemm_wrapper/entrypoint.py:204 deep_gemm.tf32_hc_prenorm_gemm`(deep_gemm 未 import)。根因:`hc_pre`(deepseek_v4.py:1236)默认 `SGLANG_OPT_USE_TILELANG_MHC_PRE=True` 先走 sglang `mhc_pre`(tilelang/deep_gemm),aiter 分支(`_is_hip and SGLANG_OPT_USE_AITER_MHC_PRE`,1258)到不了。绕法=`SGLANG_OPT_USE_TILELANG_MHC_PRE=0` + `SGLANG_OPT_USE_TILELANG_MHC_POST=0` → 走 `aiter.ops.mhc`(AITER_MHC_PRE/POST 默认 True)。=wip 的 E6 fix。已加 Dockerfile ENV,train17 验证。

## E14 验证 + E15(新) + 配置统一(2026-06-28 凌晨)

**E14 解(验证)** 设 `SGLANG_OPT_USE_TILELANG_MHC_PRE=0`+`SGLANG_OPT_USE_TILELANG_MHC_POST=0` 后 train17 加载 aiter mhc(`[aiter] import module_mhc`),栈里不再有 mhc_pre/deep_gemm → mhc 走 `aiter.ops.mhc`(deepseek_v4.py:1258 `_is_hip and SGLANG_OPT_USE_AITER_MHC_PRE` 分支,AITER_MHC 默认 True)。

**E15(新,诊断中)** cuda graph capture forward 撞 `AttributeError: 'tuple' object has no attribute 'shape'`:`deepseek_v2.py:397 down_proj → linear.py:1575 should_use_tp_invariant_row_linear(input_parallel.shape[-1])`,input_parallel 是 tuple 而非 tensor。非 cuda/deep_gemm 类,疑 MoE activation(gate_up→act→down)某 ROCm 路径返回 tuple(疑似 fused clamp/act,对应 wip E7)。

**配置统一(单一真相源)**:DSv4 ROCm rollout 的一串 env(`SGLANG_OPT_USE_AITER_INDEXER=1` / `SGLANG_OPT_USE_TOPK_V2=0` / `SGLANG_OPT_USE_TILELANG_MHC_PRE+POST=0`)统一放 `docker/Dockerfile.rocm` ENV(镜像级,radixark+wip 都生效);`run_deepseek_v4.py` 不再重复设——删掉过时且不完整的 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`(只 null metadata 路径、compute 仍 import deep_gemm,且被 aiter indexer flag 取代,注释也误导)。

## E15 诊断+绕法(2026-06-28 凌晨)

**E15(绕法已知)** MoE down_proj 收到 tuple:`deepseek_v2.py:382` 的 fused-clamp fp8 路径 `x = (x_fp8, x_scale)`,传给 `down_proj`(397)→ `linear.py:1575 should_use_tp_invariant_row_linear(input_parallel.shape[-1])`,tuple 无 `.shape`。根因 `use_fused_clamp_act_mul = _is_hip and SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL`(deepseek_v2.py:262,默认 True,environ:743)→ ROCm 默认走 aiter fused clamp,其 fp8 分支返回 (fp8,scale) tuple 但 tp-invariant row-linear 未 unwrap。绕法 `SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL=0` → 走 `silu_and_mul_clamp`(deepseek_v2.py:390,返回 tensor)。= wip 的 E7。已加 Dockerfile ENV,train18 验证。
