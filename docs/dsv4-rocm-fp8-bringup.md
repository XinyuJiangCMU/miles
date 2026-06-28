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
- **E15/E16** MoE down_proj tuple / silu `<cuda_fp8.h>` → env 集(`FUSED_COMPRESS` triton);中途的 `FUSED_CLAMP_ACT_MUL=0` 是歧路、已弃。
- *(旧 base v0.5.10 时期 E2-E9:tile_kernels stub、silu warp patch 等 —— 已被「新 base + env 集」整体取代,不再相关。)*

## 当前进展 + 待办(重点)

- **rollout**:套用 env 集后验证中(train19);预期一次过 capture + rollout forward。
- **actor 侧(核心剩余工作)**:miles 训练走 **Megatron**,其 MHC/quant 经 `deepseek-ai/TileKernels`(CUDA-only,= 旧 E2),**不经 sglang env**,所以这套 rollout env 救不到它。rollout 全绿、训练真跑到 actor forward 时才会暴露;届时需要 ROCm 实现(优先找 aiter/triton 现成的,torch 兜底)。
- **fp8 训练 step 验收**:rollout + actor 都通后,确认 `fp8_training=True` 的训练 step 真迭代、loss 不 NaN(blockwise e4m3,`NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`)。

## 产物(host mount,容器重启不丢)

- checkpoint `models/DeepSeek-V4-Flash-FP8-4layer`(27G)、bf16 `...-bf16`(52G)、torch_dist `..._torch_dist`(52G)。
- 当前日志 `train19.log`(最新一轮)。
