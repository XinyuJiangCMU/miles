import os
import sys
from dataclasses import replace

_AMD_EXTRA_ARGS = (
    "--sglang-nsa-decode-backend tilelang "
    "--sglang-nsa-prefill-backend tilelang "
    "--check-weight-update-skip-list rotary_emb.cos_cache rotary_emb.sin_cache"
)


def _with_amd_extra_args(extra_args: str) -> str:
    return f"{extra_args} {_AMD_EXTRA_ARGS}".strip()


def _execute_train(args) -> None:
    from scripts.run_glm5_2_744b_a40b import _execute_train as base_execute_train

    base_execute_train(
        replace(
            args,
            extra_args=_with_amd_extra_args(args.extra_args),
            megatron_use_deepep=False,
        )
    )


def _configure_cli_args(cli_args: list[str]) -> list[str]:
    cli_args = list(cli_args)
    for index in range(len(cli_args) - 1, -1, -1):
        if cli_args[index] == "--extra-args":
            if index + 1 == len(cli_args):
                raise SystemExit("--extra-args requires a value")
            cli_args[index + 1] = _with_amd_extra_args(cli_args[index + 1])
            break
        if cli_args[index].startswith("--extra-args="):
            _, extra_args = cli_args[index].split("=", 1)
            cli_args[index] = f"--extra-args={_with_amd_extra_args(extra_args)}"
            break
    else:
        cli_args.extend(["--extra-args", _AMD_EXTRA_ARGS])
    cli_args.append("--no-megatron-use-deepep")
    return cli_args


if __name__ == "__main__":
    cli_args = sys.argv[1:]
    if cli_args and cli_args[0] in {"train", "full-train"} and "--help" not in cli_args:
        cli_args = _configure_cli_args(cli_args)
    os.execv(
        sys.executable,
        [sys.executable, "-m", "scripts.run_glm5_2_744b_a40b", *cli_args],
    )
