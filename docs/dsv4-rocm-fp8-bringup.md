# DSv4-4layer FP8 训练 bring-up（ROCm / MI355X / gfx950）

目标:把 DeepSeek-V4-Flash-FP8-4layer(pipeline-only sanity)的 **FP8 训练**在 AMD ROCm 上跑通。验收 = 训练 step 真迭代、不报错、不 NaN。

## 环境

| 项 | 值 |
|---|---|
| 镜像 | base `rocm/sgl-dev:v0.5.14`,容器 `dsv4-fp8-v14`(8×MI355X) |
| miles | 容器内 `radixark/main`;改动在 `XinyuJiangCMU/miles@wip/dsv4-rocm-tile-kernels-import-guard` |
| sglang | fork `XinyuJiangCMU/sglang@sglang-miles-dsv4-rocm`;Dockerfile fetch 该 branch |
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
| `AITER_INDEXER=false` + `FP8_PAGED_MQA_LOGITS_TORCH=1` | indexer paged-MQA-logits → torch fn(triton 3.4 下 aiter gluon/legacy 都不可用,见 E20;配 sglang fork 3 处修) | E20 |
| `USE_TOPK_V2=false` + `USE_JIT_INDEXER_METADATA=false` | 跳过 `<cuda/ptx>` topk JIT | CI |
| `DEEPGEMM_HC_PRENORM=false` + `USE_TILELANG_MHC_PRE/POST=false` | MHC → `aiter.ops.mhc` | CI |
| `USE_FUSED_COMPRESS=true` + `_TRITON=true` | MoE 激活/压缩 → triton | CI |
| `USE_COMPRESSOR_V2=false` | c4 indexer compressor → v1 triton(避 c4_v2.cuh,见 E19) | E19 |
| `AITER_CONFIG_GEMM_*`(6 个) + `AITER_CONFIG_FMOE` 各指 default 单文件 | 避 aiter config-merge baton 死锁(见 E18) | E18 |
| `HACK_FLASHMLA_BACKEND=triton` / `USE_TILELANG_INDEXER=false` / `MULTI_STREAM=false`×2 / `AITER_BF16_FP8_MOE_BOUND=0` / `FP8_WO_A_GEMM=false` / `DSV4_FP4_EXPERTS=false` | 其余 ROCm 路径 | CI |

## blocker 台账(E10-E23 rollout 全通 + E2/E24/E25/E26 actor)

完全解决/绕过的一行,关键的多写。

- **E10** ✅ sgl_kernel 缺 dsv4 op → base v0.5.14 自带、不重编。
- **E11** ✅ ray `not a valid Sentinel` → `click==8.2.1`。
- **E12** ✅ indexer `No module deep_gemm`(metadata.py PagedIndexerMetadata,CUDA-only)→ `FP8_PAGED_MQA_LOGITS_TORCH=1` 让 deep_gemm_metadata=None。(最初用 `AITER_INDEXER=true` 走 aiter,后因 E20 改 `AITER_INDEXER=false` 走 torch fn。)
- **E13** ✅ topk_v2 `<cuda/ptx> not found` → `USE_TOPK_V2=false`。
- **E14** ✅ mhc `NameError deep_gemm` → `DEEPGEMM_HC_PRENORM=false`+`TILELANG_MHC=false` → aiter.ops.mhc。
- **E15** ✅(已解,fork+PR) MoE down_proj 收 fused_clamp fp8 tuple `(x_fp8,x_scale)`;sglang-miles 的 true_on_policy(sglang upstream PR 26359)给 RowParallelLinear 加了 `should_use_tp_invariant_row_linear(input_parallel.shape[-1])`,在 tuple 上读 `.shape` 炸(sglang main 无此模块)。根因:`matmul_tp_inv` 仅 bf16/fp16/fp32,fp8 row-linear 必走 `quant_method.apply`(`Fp8LinearMethod` 本就拆 tuple)。修:`if not isinstance(self.quant_method, Fp8LinearMethod) and should_use_tp_invariant_row_linear(...)`(读 .shape 前短路)。**fork+pin(非 patch):** [`XinyuJiangCMU/sglang@sglang-miles-dsv4-rocm`](https://github.com/XinyuJiangCMU/sglang/tree/sglang-miles-dsv4-rocm),PR(branch sglang-miles-dsv4-rocm)(commit 链:tuple-skip → Fp8LinearMethod gate → torch fn path 3 修);Dockerfile fetch 该 branch。
- **E16** ✅(已绕) silu `<cuda_fp8.h>` JIT(无 ROCm guard) —— E15 的 Fp8 gate 路径走 `FUSED_COMPRESS`(clamp=1)、不再走 `silu_and_mul_clamp`,自然避开。
- **E17** ✅(已绕) cuda graph capture 在 ROCm colocate hang(GPU 0%、停在 `Capture cuda graph begin bs=256`);aiter/triton kernel 不兼容 cuda graph capture。绕:`miles/utils/arguments.py` colocate 块 `if is_hip() and not args.sglang_disable_cuda_graph: args.sglang_disable_cuda_graph = True`(经 `--sglang-*`→ServerArgs 自动映射,覆盖 normal/external、限 colocate+ROCm、NV 不影响)。
- **E18** ✅(已解) aiter `get_config_file`(jit/core.py) glob `model_configs/*.csv` 后 `update_config_files` merge 写 `/tmp/aiter_configs`,走 FileBaton mp_lock;colocate 8+ engine 进程抢同一 baton 死锁(持有者卡 do_wait 不 release、全员 wait;清 lock/加大 watchdog 都治标)。**非 online tune**(AITER_ONLINE_TUNE 默认 0)。根因短路:`update_config_files` 开头 `if len(path_list)<=1: return` —— `AITER_CONFIG_GEMM_*`(6 个)+`AITER_CONFIG_FMOE` 各指 default 单文件,走单路径跳过 merge;缺的 shape 自动用 default kernel。(train26 实测 gemm baton 绕过、train28 实测 fmoe baton 绕过。)
- **E19** ✅(已绕) rollout forward `tvm.error.InternalError: Tensor match failed Tensor<N,4,512>` @ `c4_v2.cuh`(`indexer.py forward_c4_indexer`)。compressor_v2 的 c4_v2 kernel `TensorMatcher` 期望 kv_input 2D,caller 传 3D。绕:`SGLANG_OPT_USE_COMPRESSOR_V2=false` 走 compressor v1 的 triton `_c128_compress_*` kernel。(train27/28 实测 c4_v2 不再撞。)
- **E20** ✅(已绕,torch fn 定案) DSv4 indexer paged-MQA-logits 在 ROCm 三条路:① aiter AOT gluon(`AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`)需 `triton>=3.6`,容器是 `pytorch-triton-rocm 3.4`(ROCm7.0 配套)→ 编译 `ValueError: CDNA_VERSION is not in list`(E21),死路;② aiter legacy → `assert KVBlockSize==1` 与 DSv4 page 不符;③ **torch fn `fp8_paged_mqa_logits_torch`**(`AITER_INDEXER=false`+`FP8_PAGED_MQA_LOGITS_TORCH=1`),纯 torch、不依赖 triton 版本。**定案走 ③**(agent 调研:升 triton≥3.6 要连 torch 升、炸整镜像;v2 c4_v2 是 tvm 语义 ABI 不匹配,均比 torch fn 重)。torch fn path 配 sglang fork 3 处修(都在 fork branch sglang-miles-dsv4-rocm):E20=torch fn 开头 squeeze seq_lens(caller forward_c4_indexer:551 给 deep_gemm path unsqueeze 成 2D);E22=indexer.py:586 `getattr(core_metadata,"c4_sparse_raw_indices",None)`(DSV4AttnMetadata 无此 field,非 capture/hisparse 本该 None);E23=`fused_compress_triton._c128_decode_kernel` 的 kv_input/score_input 两处 `tl.load` 指针补 `(slot_offs*0)[:,None]` 显式广播(triton 3.4 不自动把 (1,D) 广播到 mask 的 (BLOCK_S,D);与 page_size 无关,128 是 c128 压缩比)。train34 实测 **prefill+decode 全过、rollout forward 整链通**。
- **E2** ✅(已解,torch mhc/quant) rollout 全通后 Megatron actor init(`MegatronTrainRayActor.init`→`spec_utils.py:55 import_module`)撞 `No module named 'tile_kernels'`→spec=None→`specialize for NoneType`。DSv4 plugin(`miles_plugins/models/deepseek_v4/ops/qat.py`/`hyper_connection.py`)硬 import CUDA/SM90-100-only 的 `tile_kernels`(gfx950 装不了)。修:纯 torch autograd 重写 —— `qat.py` 的 `per_token_cast_back`(fp8 blockwise dequant)+ `hyper_connection.py` 的 7 个 mhc 函数(mhc_pre_norm_fn/split_mixes/apply_mix/head_compute_mix/post/pre_big_fuse/sinkhorn);数学从 sglang `srt/layers/mhc.py` 的 TileLang 参考反推。**全 torch** 因 `aiter.ops.mhc` 只 forward 无 backward、actor 训练需反传。自测 forward+backward finite、mhc_post vs 显式 loop diff 4.8e-7、sinkhorn doubly-stochastic。train35 实测 actor init 过、update_weights(11s)、rollout generate 都通。**提速待办(用户要求,torch mhc 慢)**:mhc forward 换 aiter(`mhc_pre`/`mhc_post`/`mhc_pre_big_fuse`)+ backward 手写 autograd.Function(aiter 无 bwd kernel);sanity 通过后做。
- **E24** ✅(已绕,关 R3) generate 完→train step 撞 `ValueError: rollout_routed_experts is required in rollout_data for replay`(`actor.py train_actor`→`replay_data.py:53`)。`--use-rollout-routing-replay`(R3,`run_deepseek_v4.py enable_r3`)需 rollout 回传 MoE routed_experts(`sglang_rollout.py:178` 请求 `return_routed_experts`→收 `meta_info["routed_experts"]`),但 ROCm sglang aiter fused MoE 不导出 routing→缺。R3 是 on-policy 优化、sanity 不必需。修:`run_deepseek_v4.py __post_init__` 加 `is_hip()` guard 关 `enable_r3`(NV 不影响)。要严格 on-policy 需让 aiter MoE 导出 routed_experts(单独 feature,后续)。
- **E25** ✅(已绕) compute_log_prob 撞 `NotImplementedError: fuse_wgrad_accumulation (gradient_accumulation_fusion) is not yet supported in the ROCm blockwise grouped FP8 path`(TE `grouped_linear_blockwise.py`,MoE grouped linear)。修(message 自带解):`run_deepseek_v4.py` fp8_training 段加 `is_hip()` guard `--no-gradient-accumulation-fusion`——wgrad 返回 plain gradient,Megatron DDP post-hook 累加进 fp32 main_grad(bring-up 数值等价;NV 保留 fusion)。train37 实测 compute_log_prob 过。
- **E26** ✅(已绕) train step compressor wkv `linear_bf16_fp32` 的 `torch.mm(bf16,bf16,out_dtype=fp32)` ROCm hipblas 不支持(`gemm input bf16 output float not supported for ROCm`)。修:`precision_aligned_ops.py _BFloat16LinearFP32Func.forward` 加 `torch.version.hip` guard 用 fp32 matmul(精度 ≥ cublas bf16-in/fp32-accumulate;backward 本就 fp32;NV 保留 bf16 快路径)。train38 实测 rollout/generate 整链过。

## 当前状态:✅ 端到端 fp8 训练 sanity 验收通过(train38)

train38 完整跑通一步 RL 训练并进入第 2 步循环,**训练 step 真迭代、不报错、不 NaN**:
- **generate** ✅ 256/256(rollout forward 全链,E10-E23)
- **compute_log_prob** ✅ `Timer log_probs end 47.5s`,`rollout/log_probs=-1.71`、`rollout_log_probs=-1.59`(有限、不 NaN;E25 wgrad-fusion 关生效)
- **train step** ✅ `Timer train end 163.7s`,`actor_train_tflops=15.47`、`tok_per_s=9359`(真在算;E2 torch mhc + E26 fp32 gemm 生效)
- **update_weights** ✅ 第 2 次 10.9s → **第 2 个 rollout step 已开始**(pipeline 真迭代起来)

全程 `check_for_nan_in_loss_and_grad=True` 没 raise。fp8 blockwise e4m3 + `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`。

**待办(性能/严格性,非 sanity 必需)**:
- ① **mhc 提速**:actor_train 一步 163.7s 大头在 torch mhc(用户已指出慢)。提速 = mhc forward 换 aiter(`mhc_pre`/`mhc_post`/`mhc_pre_big_fuse`)+ backward 手写 autograd.Function;prenorm-GEMM/norm 借 Primus-Turbo,mixing/residual/sinkhorn 主体手写(见全地图:MHC backward 是 actor 侧唯一要自己写的)。
- ② **R3 严格 on-policy**:ROCm 让 aiter fused MoE 导出 routed_experts 后可重开 routing replay。
- ③ **indexer 提速(torch fn 慢,用户要求)**:当前 indexer = torch fn `fp8_paged_mqa_logits_torch`(E20,正确但慢:全 KV upcast fp32 + padded dense bmm + 多趟 elementwise)。两条提速路:
  - **(a) tilelang stopgap** —— 可立即试,但 **off-blessed-path、仅作 gated 实验,勿直接信其训练 metric**。AMD 在 #24933(2026-05-18)曾以 tilelang 为 indexer 主路,11 天后 PR 26662(2026-05-29,Thomas Wang)主动切到 aiter;gfx950 零 CI 验证、kernel 无 gfx95 tuning(hardcoded NUM_CU=256)、唯一正确性对比 PR 24989 在 CUDA 机上;dtype 崩 issue 27124 限 gfx94x(gfx950 likely 不触发,`is_fp8_fnuz=False` 选对 e4m3fn)。
  - **(b) aiter proper(目标,AMD blessed)** —— aiter gluon paged-MQA(快)需 `triton>=3.5`,容器 triton 3.4 死(E20/E21);aiter non-gluon `pa_mqa_logits` 是 block==1(`assert KVBlockSize==1`,不吃 DSv4 page=64)→ 需补 per-c4-token index 展开 + packed [K|scale] 地址/scale 指针重建(medium,triton 3.4 可跑)。或等 ROCm 7.2.5+/7.3(aiter gluon 复活 + 修 7.2 IPC 泄漏)→ 直接用 blessed gluon。
  - **TODO**:
    - [x] 扒 PR 26662 → **结论:弃因是「性能」非「正确性」**。决策实在 PR 26208(`[AMD] Dsv4/pr2 compressor opt`,co-author 含 **`@HaiShaw`**):明写 "improves inference PERFORMANCE... high-performance fused paths... while maintaining numerical correctness checks" → 换 aiter 是为性能(gfx950 调优 fused path),tilelang 没被判错、只是比 aiter 慢。PR 26662 只是把 CI test 同步到该 main recipe(非决策点)。**含义**:tilelang 当 stopgap 更靠谱(仍比 torch fn 快、且大概率正确),但正确性只在 CUDA 验过(PR 24989)、gfx950 未验 + gfx94x 有 dtype 崩(issue 27124)→ 信训练 metric 前仍做 top-k parity(风险中等非高)。
    - [ ] (a) 试跑:`indexer.py` weights squeeze 后加 `if USE_TILELANG_INDEXER and not use_fp4: weights = weights.float()`(kernel 要 fp32,否则调用时大声崩)+ env `SGLANG_OPT_USE_TILELANG_INDEXER=1` 且**保留 `FP8_PAGED_MQA_LOGITS_TORCH=1`**(metadata=None,缺了回 E12;tilelang 在 fn dispatch 优先于 torch)+ cuda-graph 保持关。崩则 revert torch fn。
    - [ ] (a) 若跑通且要信训练结果:离线验 tilelang logits + **top-k 索引** parity vs 真 fp32 dequant ground truth(⚠️ 不是 vs torch fn —— torch fn 按 e4m3**fnuz** 解、本身偏~2x;拿它当基准会把对的修成错的)。
    - [ ] (b) aiter non-gluon + per-c4-token index 展开(medium effort,AMD-aligned 真·快修);或评估等 ROCm 7.2.5+/7.3 直接用 aiter gluon。

## 性能优化实测结论(2026-06-28):三面墙,torch-fn 是稳定绿配置

sanity 验收后转「性能优先」,把上面三个待办(mhc / indexer / rollout)逐个**实测**,三个角度全撞 ROCm 墙或 toy 不 transfer。**结论:torch-fn 全链(rollout 3:04 + train step 110.5s + on-policy abs_diff 0.325)是当前 ROCm/4-layer 下的稳定绿配置;perf 不再死磕,转 R3 严格 on-policy(确定价值、不依赖 toy transfer)。** 实测细节:

- **indexer→aiter(待办③):手写 aiter `_aiter_fp8_paged_mqa_logits` 真实引擎越界崩,已搁置。** offline parity 过(单序列 aiter-vs-GT rel 2.75e-6、top-k recall@512=1.0),但真实引擎**批量多序列 prefill(56 序列/16384 tok)GPU Memory access fault**——`kv_indices`(`b*max_blk+pos`,page=64)在批量 page-table 布局下越界,单序列 parity 没覆盖此 case。代码留在 sglang fork 但 gated 在 `SGLANG_OPT_USE_AITER_INDEXER=false`(默认关),后续可离线建批量 repro 调索引。另:indexer 只占 rollout 时间 <10%,即便修好提速上限也低。**注意 on-policy abs_diff=0.32 是 fp8 固有量化噪声(actor-fp8 vs sglang-fp8 logprob),非 indexer fnuz 偏 —— top-k 对全局缩放不敏感,换 aiter 不改 abs_diff(train39 实测仍 0.32);原「换 aiter 顺手修 0.32」假设已证伪。**
- **mhc→aiter(待办①):microbench 实测仅占 train step ~5%,非大头,已降优先级。** standalone microbench(`loop-dsv4-fp8-train/mhc_microbench.py`,真实 shape hidden=4096/hc_mult=4/sinkhorn=20/4层)测得 mhc fwd+bwd=**15-17ms/microbatch 且几乎不随 token 变(512→2048: 15.3→16.7ms)= kernel-launch 开销 bound,非 compute bound**(20 迭代 sinkhorn + 数百 fp32 小算子)。按 global_batch 256/mbs 1/DP 1→~256 microbatch、actor_train 81.7s→~319ms/microbatch,mhc 含 recompute~22ms → **mhc 仅 ~5-7%(任何合理 microbatch 数下最多 ~9%)**。原「mhc 是 163.7s 大头」是没实测的假设,被推翻。
- **rollout decode(最大块 184s):唯一大杠杆 cuda-graph 被 ROCm 硬墙挡死(E17 深化)。** 从 train42.log 拆 rollout:decode 是绝对大头(~4096 步自回归、~2975 tok/s/120-144 并发,prefill 很快过)。decode launch-bound,教科书级解药是 cuda-graph,但 **E17 根因 = ROCm colocate 下 cuda-graph capture 直接 hang(GPU 0%、卡 `Capture cuda graph begin bs=256`),aiter/triton kernel 不兼容 HIP graph capture** → 重开撞回 hang。这是 ROCm 硬墙,非本项目能短期解。
- **运维根因(供后人避坑)**:① GPU memory fault 会把 core dump(`gpucore.*`,8 engine 各 ~16-105G)吐到进程 cwd `/root/miles`,**几秒填满 `/`**(`No space left`),清 `gpucore.*` 即恢复;debug 越界类 bug 别跑全训练,改 standalone repro。② 运行镜像(`...20260627`)**早于 E18-E20 的 ROCm indexer env(`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` 等),没 bake 进镜像** → 必须在 launch shell export(`loop-dsv4-fp8-train/launch_dsv4.sh` 固化全套 knob + `AITER_INDEXER` 参数);否则 `metadata.py:113` 掉 else 分支 `import deep_gemm`(CUDA-only)崩。③ 停 colocate 训练要彻底杀 `sglang::router/scheduler`(`ray stop`/`pkill -f run_deepseek_v4` 不够,VRAM 不降就找活着的 sglang 进程杀,回 ~300MB 才算干净);dev 容器 pid1=`sleep` 不 reap → zombie 堆积但无害。

## 24h 目标 + TODO(先对再快)

sanity liveness 已过(train38),接下来按 **correctness before speed** 推进。**验收锚定 AMD TE blockwise FP8 PR(647)的事实标准**(无公开的真·NV DSv4 逐张量数字 → 要拿得有 NV 机):fp8-vs-bf16 **relerr ≤ 0.04**、on-policy train-vs-rollout logprob **abs-diff ≤ 0.04**(≥10 步不上升)。验收用 miles 现成工具:`tests/e2e` Comparator(`--diff-threshold 0.002 --logprob-threshold 0.06`)+ debug_dump 的 `train_rollout_abs_diff`。长期参考(非 4-layer gate):DeepSeek-V3 整轮 loss relerr <0.25%。

### Stage 1 — liveness(基本达成,补确认)
- [x] 清 actor ROCm 算子限制(E2 torch mhc/quant、E25 wgrad-fusion、E26 bf16→fp32 gemm),train step 真迭代不 NaN(train38)
- [ ] 连续 **≥10 步** loss/grad 全 finite、check_for_nan 不触发(现 1–2 步,再观察)

### Stage 2 — 数值对齐 bf16(真 bar,24h 主目标)
- [ ] **先跑 bf16 golden**:`fp8_training=False` 同 seed/同数据,dump 逐层 fwd / logprob / grad —— **不可跳的第一子任务**(无 golden 则 Stage 2 无从验)
- [ ] fp8 run dump → Comparator 对照 golden:端到端 **relerr ≤ 0.04**(0.04–0.05 报警并排查)
- [ ] **train_rollout_abs_diff ≤ 0.04** 且 ≥10 步不上升;`train_rollout_kl` ~1e-3 量级、无上趋势;importance ratio exp(Δlogprob) 围绕 1.0
- [ ] 比对前断言两边 quant 粒度一致(1×128 act / 128×128 weight)、fp8 侧 FP32 scale(`NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`,非 UE8M0)—— **recipe 差异 ≠ bug**
- [ ] (compressor 走 TP 时)backward FP32 all-reduce —— DeepSeek V4 的硬性正确性要求(= 我们的 E26 方向),bf16 跨 TP 累加会偏

### Stage 3 — 提速(⚠️ 已实测,见上「性能优化实测结论」:三个靶全撞 ROCm 墙/toy 不 transfer,torch-fn 为稳定绿配置,perf 不死磕)
- [ ] **mhc 提速**(见待办 ①):forward 换 aiter(`mhc_pre`/`mhc_post`/`mhc_pre_big_fuse`)+ backward 手写 `autograd.Function`。**gate:手写 backward 必须先过 `torch.autograd.gradcheck` / 对现有 pure-torch autograd 版达 bf16 容差(1e-2)才允许接入训练**;过不了回退 pure-torch
- [ ] **indexer 提速**(见待办 ③:tilelang stopgap 仅 gated 实验 / aiter non-gluon 才是 blessed 真修)
- [ ] 任何提速接入后,**重跑 Stage 2 全部数值 gate 仍须通过**,否则回退
- 数学参考:mHC = Manifold-Constrained Hyper-Connections(Sinkhorn-Knopp 投影到 Birkhoff polytope);TileKernels `mhc/`(sinkhorn / pre_split_mixes / pre_apply_mix / pre_big_fuse / head_compute_mix / post,fwd+bwd)= 唯一公开 reference 实现。MHC backward 是 actor 侧**唯一**全世界只有 CUDA 版(TileKernels)、ROCm 必须自己写的算子(aiter/TE/Primus-Turbo 全无)

### R3 严格 on-policy(见待办 ②,非 sanity 必需)
- [ ] ROCm 让 sglang aiter fused MoE 导出 `routed_experts` 后重开 routing replay(当前 `is_hip()` guard 关掉)

### 显式不做(防 24h 空转)
- 不追 4-layer 收敛 / loss 下降 / eval 准确率(模型未训练,eval 无意义)
- 不拿 DeepSeek-V3 的 <0.25% 当 4-layer gate(那是整轮收敛指标,几步 sanity 测不出)
- ue8m0 vs fp32 scale 的 recipe 差异先归因 recipe 再归因 kernel,别死磕 0.04 修不存在的 bug
- rollout generate 慢 → on-policy 趋势用 `debug_minimal`(短响应、关 eval)保证步数够,与输出质量无关

## 产物(host mount,容器重启不丢)

- checkpoint `models/DeepSeek-V4-Flash-FP8-4layer`(27G)、bf16 `...-bf16`(52G)、torch_dist `..._torch_dist`(52G)。
- 运行日志 `train{N}.log`(最新 train38 = 端到端验收通过那次;中间过程 log 已清)。
