"""AMD/Triton variant of the true on-policy training script.

Uses SGLang's Triton attention backend (extend_attention_fwd_unified) on both
inference and training sides for numerical alignment on AMD/ROCm hardware.
"""

import os

import miles.utils.external_utils.command_utils as U

MODEL_NAME = os.environ.get("MILES_SCRIPT_MODEL_NAME", "Qwen3-0.6B")
assert MODEL_NAME in {"Qwen3-0.6B", "Qwen3-4B"}

MODE = os.environ.get("MILES_SCRIPT_MODE", "normal")
assert MODE in {"normal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "1"))


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {2 if MODE == 'debug_one_sample' else 3000} "
        f"--rollout-batch-size {1 if MODE == 'debug_one_sample' else 32} "
        f"--n-samples-per-prompt {1 if MODE == 'debug_one_sample' else 8} "
        f"--rollout-max-response-len {2 if MODE == 'debug_one_sample' else 1024} "
        "--rollout-temperature 1 "
        f"--global-batch-size {1 if MODE == 'debug_one_sample' else 256} "
    )

    eval_args = ""
    if MODE == "normal":
        eval_args = (
            "--eval-interval 20 "
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {0.2 if MODEL_NAME == 'Qwen3-4B' else 0.4} "
        # Disable cuda-graph and radix-cache for deterministic inference alignment
        "--sglang-disable-cuda-graph --sglang-disable-radix-cache "
        f"{'--sglang-warmups 0 ' if MODE == 'debug_one_sample' else ''}"
    )

    fsdp_args = (
        "--update-weight-buffer-size 536870912 "  # 512MB
    )

    ci_args = ""

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--train-backend fsdp "
    )

    if MODEL_NAME == "Qwen3-4B":
        misc_args += (
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 2048 "
        )

    # AMD/Triton: replace fa3 with triton on both inference and training sides.
    # --attn-implementation triton activates the SGLang Triton attention bridge
    # (apply_sglang_triton_attention_patch), which uses extend_attention_fwd_unified —
    # the same kernel as the SGLang inference-side Triton backend.
    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend triton "
        "--attn-implementation triton "
        "--deterministic-mode "
        "--true-on-policy-mode "
    )
    true_on_policy_envs = {
        "NCCL_ALGO": "allreduce:tree",
    }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars={
            **true_on_policy_envs,
            # Return pre-temperature logprobs from SGLang for correct logprob alignment
            "SGLANG_RETURN_ORIGINAL_LOGPROB": "1",
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
