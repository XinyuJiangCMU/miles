from dataclasses import replace

from scripts.run_glm5_2_744b_a40b import ScriptArgs
from scripts.run_glm5_2_744b_a40b import _execute_train as _base_execute_train


def _execute_train(args: ScriptArgs) -> None:
    _base_execute_train(
        replace(
            args,
            extra_args=(
                f"{args.extra_args} "
                "--sglang-nsa-decode-backend tilelang "
                "--sglang-nsa-prefill-backend tilelang "
                "--check-weight-update-skip-list rotary_emb.cos_cache rotary_emb.sin_cache"
            ),
            megatron_use_deepep=False,
        )
    )
