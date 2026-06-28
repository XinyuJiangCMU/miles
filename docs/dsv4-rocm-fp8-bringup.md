# DSv4-4layer FP8 训练 bring-up（ROCm / MI355X / gfx950）

目标:把 DeepSeek-V4-Flash-FP8-4layer(pipeline-only sanity)的 **FP8 训练**在 AMD ROCm 上跑通。验收 = 训练 step 真迭代、不报错、不 NaN。

## 环境

| 项 | 值 |
|---|---|
| 镜像 | base `rocm/sgl-dev:v0.5.14`,容器 `dsv4-fp8-v14`(8×MI355X) |
| miles | 容器内 `radixark/main`;改动在 `XinyuJiangCMU/miles@wip/dsv4-rocm-tile-kernels-import-guard` |
| sglang | fork `XinyuJiangCMU/sglang@sglang-miles-dsv4-rocm`(PR #24);Dockerfile fetch 该 branch |
| TE | `JessicaJiang-123/TransformerEngine@amd-qwen3-30b-a3b-fp8-dev`(blockwise fp8) |
| 模型 | `Pinaster/DeepSeek-V4-Flash-FP8-4layer` |

## 核心结论:build 配方 + 一套 rollout env,不是一堆 bug

**① build 配方**(`docker/Dockerfile.rocm`):
- `FROM rocm/sgl-dev:v0.5.14`,**别 uninstall+重编 sgl_kernel** —— base 自带 DSv4 op(`dsv4_fused_q_indexer_rope_hadamard_quant` 等),重编反而丢。
- 末尾 re-pin `click==8.2.1`(中途被 requirements 升到 8.4.2 会破 ray 2.44 CLI;8.2.1 是 ray+typer 唯一兼容点)。
- sglang 段 `git fetch ${SGLANG_REPO} ${SGLANG_BRANCH}` 拉 fork branch `sglang-miles-dsv4-rocm`(= sglang-miles + E15 修),只装 python 层(不重编 sgl-kernel)。
- TE 用 blockwise-fp8 分支。

**② 一套 rollout env**(Dockerfile ENV,镜像级) —— 以 sglang AMD nightly CI `test/registered/amd/test_deepseek_v4_flash_fp8.py` 的 `COMMON_ENV_VARS` 为底,加上 DSv4 训练 rollout 实测需要的几个。作用:把 DSv4 的 sparse-attn indexer / MHC / MoE 从 CUDA-only(`deep_gemm` / `<cuda/ptx>` / tilelang / `<cuda_fp8.h>`)切到 aiter/triton:

| env | 作用 | 来源 |
|---|---|---|
| `AITER_INDEXER=true` + `FP8_PAGED_MQA_LOGITS_TORCH=1` + `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1` | indexer paged-MQA-logits → aiter AOT gluon kernel(支持 DSv4 page block,见 E20) | CI + E20 |
| `USE_TOPK_V2=false` + `USE_JIT_INDEXER_METADATA=false` | 跳过 `<cuda/ptx>` topk JIT | CI |
| `DEEPGEMM_HC_PRENORM=false` + `USE_TILELANG_MHC_PRE/POST=false` | MHC → `aiter.ops.mhc` | CI |
| `USE_FUSED_COMPRESS=true` + `_TRITON=true` | MoE 激活/压缩 → triton | CI |
| `USE_COMPRESSOR_V2=false` | c4 indexer compressor → v1 triton(避 c4_v2.cuh,见 E19) | E19 |
| `AITER_CONFIG_GEMM_*`(6 个) + `AITER_CONFIG_FMOE` 各指 default 单文件 | 避 aiter config-merge baton 死锁(见 E18) | E18 |
| `HACK_FLASHMLA_BACKEND=triton` / `USE_TILELANG_INDEXER=false` / `MULTI_STREAM=false`×2 / `AITER_BF16_FP8_MOE_BOUND=0` / `FP8_WO_A_GEMM=false` / `DSV4_FP4_EXPERTS=false` | 其余 ROCm 路径 | CI |

## blocker 台账(E10-E20 rollout + E2 actor)

完全解决/绕过的一行,关键的多写。

- **E10** ✅ sgl_kernel 缺 dsv4 op → base v0.5.14 自带、不重编。
- **E11** ✅ ray `not a valid Sentinel` → `click==8.2.1`。
- **E12** ✅ indexer `No module deep_gemm` → `AITER_INDEXER=true`(aiter indexer)。注意:aiter paged-MQA 默认 preshuffle 在 ROCm 需 AOT gluon(见 E20),否则 fallback legacy KVBlockSize=1。
- **E13** ✅ topk_v2 `<cuda/ptx> not found` → `USE_TOPK_V2=false`。
- **E14** ✅ mhc `NameError deep_gemm` → `DEEPGEMM_HC_PRENORM=false`+`TILELANG_MHC=false` → aiter.ops.mhc。
- **E15** ✅(已解,fork+PR) MoE down_proj 收 fused_clamp fp8 tuple `(x_fp8,x_scale)`;sglang-miles 的 true_on_policy(sgl#26359)给 RowParallelLinear 加了 `should_use_tp_invariant_row_linear(input_parallel.shape[-1])`,在 tuple 上读 `.shape` 炸(sglang main 无此模块)。根因:`matmul_tp_inv` 仅 bf16/fp16/fp32,fp8 row-linear 必走 `quant_method.apply`(`Fp8LinearMethod` 本就拆 tuple)。修:`if not isinstance(self.quant_method, Fp8LinearMethod) and should_use_tp_invariant_row_linear(...)`(读 .shape 前短路)。**fork+pin(非 patch):** [`XinyuJiangCMU/sglang@sglang-miles-dsv4-rocm`](https://github.com/XinyuJiangCMU/sglang/tree/sglang-miles-dsv4-rocm),PR [#24](https://github.com/XinyuJiangCMU/sglang/pull/24)(2-commit:tuple-skip → Fp8LinearMethod gate);Dockerfile fetch 该 branch。
- **E16** ✅(已绕) silu `<cuda_fp8.h>` JIT(无 ROCm guard) —— E15 的 Fp8 gate 路径走 `FUSED_COMPRESS`(clamp=1)、不再走 `silu_and_mul_clamp`,自然避开。
- **E17** ✅(已绕) cuda graph capture 在 ROCm colocate hang(GPU 0%、停在 `Capture cuda graph begin bs=256`);aiter/triton kernel 不兼容 cuda graph capture。绕:`miles/utils/arguments.py` colocate 块 `if is_hip() and not args.sglang_disable_cuda_graph: args.sglang_disable_cuda_graph = True`(经 `--sglang-*`→ServerArgs 自动映射,覆盖 normal/external、限 colocate+ROCm、NV 不影响)。
- **E18** ✅(已解) aiter `get_config_file`(jit/core.py) glob `model_configs/*.csv` 后 `update_config_files` merge 写 `/tmp/aiter_configs`,走 FileBaton mp_lock;colocate 8+ engine 进程抢同一 baton 死锁(持有者卡 do_wait 不 release、全员 wait;清 lock/加大 watchdog 都治标)。**非 online tune**(AITER_ONLINE_TUNE 默认 0)。根因短路:`update_config_files` 开头 `if len(path_list)<=1: return` —— `AITER_CONFIG_GEMM_*`(6 个)+`AITER_CONFIG_FMOE` 各指 default 单文件,走单路径跳过 merge;缺的 shape 自动用 default kernel。(train26 实测 gemm baton 绕过、train28 实测 fmoe baton 绕过。)
- **E19** ✅(已绕) rollout forward `tvm.error.InternalError: Tensor match failed Tensor<N,4,512>` @ `c4_v2.cuh`(`indexer.py forward_c4_indexer`)。compressor_v2 的 c4_v2 kernel `TensorMatcher` 期望 kv_input 2D,caller 传 3D。绕:`SGLANG_OPT_USE_COMPRESSOR_V2=false` 走 compressor v1 的 triton `_c128_compress_*` kernel。(train27/28 实测 c4_v2 不再撞。)
- **E20** (绕法 train30 验证中) aiter `deepgemm_fp8_paged_mqa_logits` 默认 preshuffle 路径在 ROCm 需 `Triton>=3.5.0` 或 `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`,否则 fallback legacy `page_size=1/KVBlockSize=1` 与 DSv4 `page_size=256` 不符 → `assert KVBlockSize==1`;torch fn(`fp8_paged_mqa_logits_torch`)也 `assert seq_lens.shape==(batch_size,)`。绕:`AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1` 启用 aiter AOT gluon paged-MQA(支持 page block,aiter/triton 非 torch;sglang `dsa/utils.py:49`+aiter `pa_mqa_logits.py` 认)。
- **E2** (actor 侧,OPEN) miles 训练走 Megatron,其 MHC/quant 经 `deepseek-ai/TileKernels`(CUDA/SM90-100-only),不经 sglang env、这套 rollout env 救不到。rollout 全通、训练真到 actor forward 才暴露;需 ROCm 实现(优先 aiter/triton,torch 兜底:`per_token_cast_back` 易、`mhc_*` 7 函数难)。

## 当前状态 + 待办

- **rollout**:E10-E19 已解/已绕,E20 绕法(`AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`)train30 验证中。E18-E20 都是 aiter/sglang 在 colocate+DSv4 page 的 ROCm 适配,逐个 env/backend 绕。
- **下一关**:rollout 全通后撞 actor 侧 E2(TileKernels)。
- **验收**:rollout+actor 都通后,确认 `fp8_training=True` 训练 step 真迭代、loss 不 NaN(blockwise e4m3,`NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`)。

## 产物(host mount,容器重启不丢)

- checkpoint `models/DeepSeek-V4-Flash-FP8-4layer`(27G)、bf16 `...-bf16`(52G)、torch_dist `..._torch_dist`(52G)。
- 运行日志 `train{N}.log`(最新 train30)。
