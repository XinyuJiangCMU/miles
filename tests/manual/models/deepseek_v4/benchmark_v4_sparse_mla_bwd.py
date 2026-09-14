"""Compare complete sparse MLA backward calls against an exported baseline module.

Run as a module from the repository root; pass --baseline-file to a copy of the
base revision's tilelang_sparse_mla_bwd.py. Compilation is excluded from timing.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import tilelang
import torch
from tests.manual.models.deepseek_v4.test_v4_tilelang_sparse_mla import compute_diff, make_inputs

from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_bwd import sparse_mqa_bwd_interface
from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface

CONFIGS = [
    (1, 512, 16, 512, 512, 128),
    (1, 512, 16, 512, 515, 160),
    (1, 512, 16, 512, 640, 256),
    (1, 1024, 16, 512, 1032, 160),
    (1, 1024, 16, 512, 1280, 384),
    (1, 512, 64, 512, 640, 256),
    (1, 1024, 64, 512, 1280, 512),
    (1, 2048, 64, 512, 2560, 512),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("warmup must be positive and repeats must be at least two")
    spec = importlib.util.spec_from_file_location("mla_bwd_baseline", args.baseline_file)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    functions = [baseline.sparse_mqa_bwd_interface, sparse_mqa_bwd_interface]
    torch.manual_seed(args.seed)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "tilelang": tilelang.__version__,
                "seed": args.seed,
                "warmup": args.warmup,
                "repeats": args.repeats,
            }
        ),
        flush=True,
    )
    for config in CONFIGS:
        q, kv, sink, indices = make_inputs(*config)
        out, lse = sparse_mqa_fwd_interface(q, kv, sink, indices)
        inputs = (q, kv, sink, out, torch.randn_like(out), indices, lse)
        reference, actual = [fn(*inputs) for fn in functions]
        errors = {}
        for name, ref, result in zip(("dq", "dkv", "d_sink"), reference, actual, strict=True):
            assert torch.isfinite(ref).all() and torch.isfinite(result).all(), name
            diff = compute_diff(ref, result)
            assert diff.rel_diff < 1e-3, (name, diff)
            errors[name] = {"rel_diff": diff.rel_diff, "max_abs_diff": diff.max_abs_diff}
        for _ in range(args.warmup):
            for fn in functions:
                fn(*inputs)
        torch.cuda.synchronize()
        samples = [[], []]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for repeat in range(args.repeats):
            for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                start.record()
                functions[index](*inputs)
                end.record()
                end.synchronize()
                samples[index].append(start.elapsed_time(end))
        before, after = [statistics.median(values) for values in samples]
        print(
            json.dumps(
                {
                    "config": config,
                    "baseline_ms": before,
                    "candidate_ms": after,
                    "reduction_pct": 100 * (1 - after / before),
                    "errors": errors,
                    "baseline_samples_ms": samples[0],
                    "candidate_samples_ms": samples[1],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
