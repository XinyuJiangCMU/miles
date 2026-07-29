"""GLM-4.7-Flash colocate GRPO on AMD MI355X (gfx950), with MTP *training*.

De-risk vehicle: GLM-4.7-Flash's native MTP head is the *standard* single-eh_proj
shape, so it exercises miles' generic MTP-training pipeline (mcore MTP block build +
mtp_loss + weight-sync) on gfx950 without any DSv4-specific custom layer. Validate the
AMD MTP pipeline here first, then port to DeepSeek-V4-Flash.

Derived from ../run_glm47_flash.py (NV/H200), with three deltas:
  1. AMD gfx950 rollout env (general SGLANG/AITER knobs; DSv4-DSA-specific ones dropped).
  2. MTP *training* enabled (NV run script only does rollout EAGLE; the MTP-training
     receipt lives in tests/e2e/megatron/test_glm47_flash/test_r3_mtp.py + _common.py:85).
  3. node-local model_dir (/workspace/models); NFS is too slow to read weights.

bf16 first (fp8-blockwise is a REQUIRED follow-up, not optional -- add
--fp8-format e4m3 --fp8-recipe blockwise + NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
+ --no-gradient-accumulation-fusion once bf16 is stable).
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-4.7-Flash"
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    enable_eval: bool = True
    enable_mtp_training: bool = True     # <-- the whole point of this de-risk run
    fp8_training: bool = False           # bf16 first; flip on for the REQUIRED fp8 follow-up
    skip_prepare: bool = True            # checkpoint/torch_dist already staged node-local
    skip_saving: bool = False            # skip --load/--save for smoke runs
    extra_args: str = ""
    # node-local NVMe (NFS reads weights at ~20-90MB/s -> GPU idles; JOURNEY-documented)
    data_dir: str = "/opt/shared/hai/datasets"
    model_dir: str = "/workspace/models"
    megatron_path: str = "/root/Megatron-LM"


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(
        f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def _train(args: ScriptArgs):
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {ref_load_path} "
    )
    # skip_saving is an outer/typer flag -- it just OMITs --load/--save (like run_deepseek_v4.py:327);
    # there is NO core `--skip-saving` arg, do not forward it to train.py.
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} --save {load_save_path} "
            f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type deepscaler --num-rollout 3000 --rollout-batch-size 32 --n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 --global-batch-size 256 "
    )

    eval_args = ""
    if args.mode != "debug_minimal" and args.enable_eval:
        eval_args += (
            f"--eval-prompt-data aime24 {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 --eval-max-response-len 16384 "
            "--eval-temperature 0.6 --eval-top-p 0.95 "
        )

    # TP4 (GLM-4.7-Flash has 20 attn heads; tp must divide 20), PP1, EP8. Single node = 8 GPU.
    perf_args = (
        "--tensor-model-parallel-size 4 --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        # MLA (--qkv-format bshd) is incompatible with --use-dynamic-batch-size; use fixed
        # --micro-batch-size like DSv4 (run_deepseek_v4.py). max-tokens-per-gpu still bounds the pack.
        "--micro-batch-size 1 --max-tokens-per-gpu 32768 "
    )

    grpo_args = (
        "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    # rollout engine: EAGLE speculative decoding (the frozen-draft side); tp must divide 20 -> engine tp4
    sglang_args = (
        "--rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.7 "
        "--sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 3 "
        "--use-rollout-routing-replay "
    )

    # MTP TRAINING: train the MTP head so the draft is synced to target each weight-update.
    # glm4.7-flash.sh MODEL_ARGS already carries --mtp-num-layers 1 (so torch_dist has MTP weights).
    mtp_args = ""
    if args.enable_mtp_training:
        mtp_args += "--enable-mtp-training --mtp-loss-scaling-factor 0.2 "

    precision_args = ""
    if args.fp8_training:
        precision_args += (
            "--transformer-impl transformer_engine --bf16 --fp8-format e4m3 --fp8-recipe blockwise "
            """--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"1"}' """
            "--no-gradient-accumulation-fusion "  # ROCm TE MoE FP8 lacks fused wgrad accumulation
        )
    else:
        precision_args += "--bf16 "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        # GLM-4.7-Flash is MLA (q-lora 768/kv-lora 512/qk 192/v 256), NOT plain GQA.
        # --attention-backend flash is for non-MLA models (see run-glm4.7-flash.sh comment);
        # MLA uses --qkv-format bshd like DSv4 (run_deepseek_v4.py:466).
        "--qkv-format bshd "
        "--moe-token-dispatcher-type alltoall "  # align with e2e receipt (_common.py); EP8 grouped-gemm
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate --use-fault-tolerance "
        "--sglang-watchdog-timeout 1800 "  # ROCm: slow aiter gemm tune under colocate
        "--rollout-health-check-interval 300 --rollout-health-check-timeout 300 "
    )

    # gfx950 rollout env -- only the *general* knobs. DSv4-DSA-specific ones
    # (SGLANG_DSV4_FP4_EXPERTS / FLASHMLA_BACKEND / TILELANG_INDEXER / COMPRESSOR_*) are
    # deliberately dropped: GLM has no sparse-MLA / indexer / compressor.
    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",   # tolerate MTP-head load-check noise
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",       # tolerate slow ROCm warmup / aiter tune
        "AITER_BF16_FP8_MOE_BOUND": "0",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "1",      # colocate cuda-graph safe capture (aiter #2061)
        # GLM-MLA rollout on gfx950: aiter's fused_qk_rmsnorm is a first-use JIT (compile_ops)
        # that dlopen's DURING cuda-graph capture (ROCm has no eager warmup) -> stray-ptr GPU fault.
        # Disabling aiter falls back to native QK-RMSNorm, no JIT-in-capture, cuda-graph stays safe.
        "SGLANG_USE_AITER": "0",
    }

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} {eval_args} {sglang_args} {mtp_args} {precision_args} {misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**extra_env_vars},
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training. Assumes checkpoint + torch_dist already staged node-local."""
    if not args.skip_prepare:
        prepare(args)
    _train(args)


app = typer.Typer(pretty_exceptions_show_locals=False)
app.command()(train)

if __name__ == "__main__":
    app()
