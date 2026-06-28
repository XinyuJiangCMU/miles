# DSv4-4layer FP8 训练 bring-up（ROCm / MI355X / gfx950）

目标:把 DeepSeek-V4-Flash-FP8-4layer(pipeline-only sanity)的 **FP8 训练**在 AMD ROCm 上跑通。验收 = 训练 step 真迭代、不报错、不 NaN。

## 环境

| 项 | 值 |
|---|---|
| 镜像 | `xinyujiangcmu/miles:rocm700-mi35x-sglang0.5.14-20260627`(base `rocm/sgl-dev:v0.5.14`),容器 `dsv4-fp8-v14`(8×MI355X) |
| miles | 容器内 `radixark/main`;改动记录在 `XinyuJiangCMU/miles@wip/dsv4-rocm-tile-kernels-import-guard` |
| TE | `JessicaJiang-123/TransformerEngine@amd-qwen3-30b-a3b-fp8-dev`(blockwise fp8) |
| 模型 | `Pinaster/DeepSeek-V4-Flash-FP8-4layer` |

## 核心结论:本质两件事,不是一堆 bug

**① build 配方**(`docker/Dockerfile.rocm`):
- `FROM rocm/sgl-dev:v0.5.14`,**别 `uninstall`+重编 sgl_kernel** —— base 自带 DSv4 op(`dsv4_fused_q_indexer_rope_hadamard_quant` 等),重编反而丢。
- 末尾 re-pin `click==8.2.1`(中途被 requirements 升到 8.4.2 会破 ray 2.44 CLI;8.2.1 是 ray+typer 唯一兼容点)。
- TE 用 blockwise-fp8 分支;sglang 只装 sglang-miles 的 python 层(不重编 sgl-kernel)。

**② 一套 rollout env**(Dockerfile ENV,镜像级) —— **直接照搬 sglang AMD nightly CI 的 `test/registered/amd/test_deepseek_v4_flash_fp8.py` 的 `COMMON_ENV_VARS`**。作用:把 DSv4 的 sparse-attn indexer / MHC / MoE 从 CUDA-only(`deep_gemm` / `<cuda/ptx>` / tilelang / `<cuda_fp8.h>`)全部切到 **aiter/triton**:

| env | 作用 |
|---|---|
| `SGLANG_OPT_USE_AITER_INDEXER=true` + `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` | indexer metadata+compute → aiter |
| `SGLANG_OPT_USE_TOPK_V2=false` + `SGLANG_OPT_USE_JIT_INDEXER_METADATA=false` | 跳过 `<cuda/ptx>` topk JIT |
| `SGLANG_OPT_DEEPGEMM_HC_PRENORM=false` + `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=false` | MHC → `aiter.ops.mhc` |
| `SGLANG_OPT_USE_FUSED_COMPRESS=true` + `_TRITON=true` | MoE 激活/压缩 → triton |
| `HACK_FLASHMLA_BACKEND=triton` / `USE_TILELANG_INDEXER=false` / `MULTI_STREAM=false`×2 / `AITER_BF16_FP8_MOE_BOUND=0` / `FP8_WO_A_GEMM=false` / `DSV4_FP4_EXPERTS=false` | 其余 ROCm 路径 |

**教训:照搬这套 env 即可,不必逐个撞 error。** 这套是 AMD CI 验证过的权威全集。

## 已解 blocker(历史,各一行)

- **E10** sgl_kernel 缺 dsv4 op → base v0.5.14 自带、不重编。
- **E11** ray `not a valid Sentinel` → `click==8.2.1`。
- **E12** indexer `No module deep_gemm` → env 集(aiter indexer)。
- **E13** topk_v2 `<cuda/ptx> not found` → env 集(`TOPK_V2=false`)。
- **E14** mhc `NameError deep_gemm` → env 集(`DEEPGEMM_HC_PRENORM`+`TILELANG_MHC=false`)。
- **E15(未解,OPEN — sglang-miles bug,sglang main 无)** MoE down_proj 收 fused_clamp fp8 tuple `(x_fp8,x_scale)`;sglang-miles `linear.py:1575 should_use_tp_invariant_row_linear(input.shape[-1])`(true_on_policy RL 改动)读 tuple.shape 炸。**对比 main:无此行、无 true_on_policy 模块**,fp8 tuple 直接进 `quant_method.apply` 正常。禁 true_on_policy(train20 `--no-train-deterministic`)**实测仍撞** —— 那行无条件求值 `input.shape`,与 on-policy 开关无关。= sglang-miles 的 true_on_policy 改动没 handle fp8 pre-quant tuple,非 env、非 Xinyu/miles repo 的事。
- **E16(连带,OPEN)** 若用 `FUSED_CLAMP_ACT_MUL=0` 避 E15、改走 `silu_and_mul_clamp`,则 JIT 编 `<cuda_fp8.h>`(无 ROCm guard、无 triton 替代)又炸。故 E15/E16 两难。
- *(旧 base v0.5.10 时期 E2-E9:tile_kernels stub、silu warp patch 等 —— 已被「新 base + env 集」整体取代,不再相关。)*

## 当前进展 + 待办(重点)

- **rollout**:env 集解了 E12/E13/E14(train19 实测不再撞);capture 撞 **E15 = sglang-miles 的 true_on_policy bug**(main 无,见上)。这不是 env 能绕的,修在 sglang-miles 侧(报维护者 / build 时破例改那 1 行 / 等上游),待 Xinyu 定方向。
- **actor 侧(核心剩余工作)**:miles 训练走 **Megatron**,其 MHC/quant 经 `deepseek-ai/TileKernels`(CUDA-only,= 旧 E2),**不经 sglang env**,所以这套 rollout env 救不到它。rollout 全绿、训练真跑到 actor forward 时才会暴露;届时需要 ROCm 实现(优先找 aiter/triton 现成的,torch 兜底)。
- **fp8 训练 step 验收**:rollout + actor 都通后,确认 `fp8_training=True` 的训练 step 真迭代、loss 不 NaN(blockwise e4m3,`NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`)。

## 产物(host mount,容器重启不丢)

- checkpoint `models/DeepSeek-V4-Flash-FP8-4layer`(27G)、bf16 `...-bf16`(52G)、torch_dist `..._torch_dist`(52G)。
- 当前日志 `train19.log`(最新一轮)。

## 突破(2026-06-28):rollout engine 起来了 —— `fired up and ready to roll`

**E15 修法验证 ✅** 根因:true_on_policy(sgl-project/sglang#26359)给 RowParallelLinear 的 matmul 包了 `should_use_tp_invariant_row_linear(input_parallel.shape[-1])`;ROCm 上 DSv4 expert MLP 走 `_is_hip` 的 aiter `fused_clamp_act_mul` fp8 快路径,喂给 down_proj 的是预量化 `(x_fp8, x_scale)` tuple,`.shape[-1]` 在 tuple 上崩。**正确修法不是判断"输入是不是 tuple",而是"这层是不是 fp8"** —— `matmul_tp_inv` 仅 bf16/fp16/fp32、永远吃不了 fp8,fp8 row-linear 必走 `quant_method.apply`(`Fp8LinearMethod` 本就有 `isinstance(x, tuple)` 分支拆 fp8 tuple)。改成 `if not isinstance(self.quant_method, Fp8LinearMethod) and should_use_tp_invariant_row_linear(...)`(读 .shape 前短路)。capture 推进过 down_proj、不再撞 tuple.shape。**正式落地(fork+pin,非 patch):** fork branch [`XinyuJiangCMU/sglang @ sglang-miles-dsv4-rocm`](https://github.com/XinyuJiangCMU/sglang/tree/sglang-miles-dsv4-rocm)(= sglang-miles + 这 1 行修);`docker/Dockerfile.rocm` 已把 sglang 段改成 `git fetch ${SGLANG_REPO} ${SGLANG_BRANCH}` 拉该 fork branch。**已 fork+push + 提 PR：[XinyuJiangCMU/sglang#24](https://github.com/XinyuJiangCMU/sglang/pull/24)**(base `sglang-miles` ← head `sglang-miles-dsv4-rocm`)。后续可向 sgl-project 提 upstream PR。

**E17(新,已绕) cuda graph capture 在 ROCm colocate hang** —— GPU 0%、log 停在 `Capture cuda graph begin ... bs=256`。aiter/triton kernel 在 cuda graph capture 模式不兼容;miles colocate 已默认禁 piecewise cuda graph(NVLS OOM 注释),但**完整 cuda graph 还开着**。绕:rollout `disable_cuda_graph=True`。落在 `miles/utils/arguments.py` 的 colocate 块(照 piecewise 写法:`if is_hip() and not args.sglang_disable_cuda_graph: args.sglang_disable_cuda_graph = True`),经 miles 现成的 `--sglang-*` → ServerArgs 自动映射生效,**覆盖 normal/external 两条路径、限定 colocate+ROCm、NV 不影响**(经 agent review,优于最初在 sglang_engine._init_normal 写死 server_args_dict)。

**结果:E10-E17 全过,`The server is fired up and ready to roll!`** —— rollout engine 起来了。下一步:rollout forward 生成 → actor train step(预期撞 actor 侧 = 旧 E2,miles Megatron 经 TileKernels 的 mhc/quant,CUDA-only)。

## 进展(2026-06-28):E18 解(aiter config baton) → 撞 E19(c4_v2 kernel shape)

**E18 解 ✅** aiter `get_config_file`(jit/core.py) glob `model_configs/*.csv` 后 `update_config_files` merge 写 `/tmp/aiter_configs`,走 FileBaton mp_lock;colocate 8+ engine 进程抢同一 baton 死锁(持有者卡 do_wait 不 release、全员 wait;清 lock / 加大 watchdog 都治标)。**非 online tune**(AITER_ONLINE_TUNE 默认 0)。**根因短路**:`update_config_files` 开头 `if len(path_list)<=1: return file_path` —— 给每个 `AITER_CONFIG_GEMM_*` env 指 default 单文件(A8W8_BLOCKSCALE/BF16/A8W8/BPRESHUFFLE/BLOCKSCALE_BPRESHUFFLE/A4W4)→ 走单路径、跳过 merge、不抢 baton;缺的 shape gemm 自动用 default kernel。已落 Dockerfile ENV。train26 实测 config-merge baton 不再死锁、推进过。

**E19(未解,OPEN)** rollout forward 撞 `tvm.error.InternalError: Tensor match failed for Tensor<351195, 4, 512>` @ `c4_v2.cuh:364`(经 `indexer.py:482 forward_c4_indexer`)。c4_v2 kernel 的 `TensorMatcher({N, kElementSize=512})` 期望 kv_input **2D**,caller 传的是 **3D `<351195,4,512>`** → 维度不匹配。这是 sglang DSv4 c4 indexer v2 kernel 与 caller 的 shape 不一致(疑 ROCm 路径 / 4-layer prune 特有);`SGLANG_OPT_USE_AITER_INDEXER` 只换 paged-mqa-logits fn、**没覆盖这条 c4 kv-score 路径**(c4_v2.cuh 是 sgl jit kernel,总被调)。下一步:fork sglang 改(c4_v2 接受 3D 或 indexer 把 kv_input reshape 成 2D),或找 c4 v1/别的 indexer backend。
