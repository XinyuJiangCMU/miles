from dataclasses import replace

from scripts.run_inkling import ScriptArgs
from scripts.run_inkling import _train as _base_train


def _train(args: ScriptArgs) -> None:
    _base_train(replace(args, sglang_attention_backend="triton"))
