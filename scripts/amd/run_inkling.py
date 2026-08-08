import os
import sys
from dataclasses import replace


def _train(args) -> None:
    from scripts.run_inkling import _train as base_train

    base_train(replace(args, sglang_attention_backend="triton"))


if __name__ == "__main__":
    cli_args = sys.argv[1:]
    if cli_args and cli_args[0] in {"train", "full-train"} and "--help" not in cli_args:
        cli_args = [*cli_args, "--sglang-attention-backend", "triton"]
    os.execv(
        sys.executable,
        [sys.executable, "-m", "scripts.run_inkling", *cli_args],
    )
