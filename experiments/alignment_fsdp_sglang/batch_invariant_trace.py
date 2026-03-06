from __future__ import annotations


def enable_miles_batch_invariant(enable_bmm: bool = False) -> None:
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

    enable_batch_invariant_mode(enable_bmm=enable_bmm)
