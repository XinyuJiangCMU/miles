from __future__ import annotations

import re
from pathlib import Path

import torch


FILE_RE = re.compile(
    r"forward_pass_id=(?P<forward_pass_id>\d+)___rank=(?P<rank>\d+)___name=(?P<name>.+)___dump_index=(?P<dump_index>\d+)\.pt$"
)


def parse_partial_name(log_path: Path) -> str | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Choose partial_name=([0-9.]+)", text)
    return matches[0] if matches else None


def dir_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            latest = max(latest, child.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def resolve_dump_dirs(dumper_root: Path, server_log: Path, capture_log: Path) -> tuple[Path, Path]:
    sg_partial = parse_partial_name(server_log)
    hf_partial = parse_partial_name(capture_log)

    sg_dir = dumper_root / f"sglang_dump_{sg_partial}" if sg_partial else None
    hf_dir = dumper_root / f"sglang_dump_{hf_partial}" if hf_partial else None

    if sg_dir and hf_dir and sg_dir.exists() and hf_dir.exists():
        return hf_dir, sg_dir

    candidates = sorted(
        [p for p in dumper_root.glob("sglang_dump_*") if p.is_dir()],
        key=dir_mtime,
    )
    if len(candidates) < 2:
        raise RuntimeError(f"Expected at least 2 dump dirs under {dumper_root}, found {len(candidates)}")

    if sg_dir is None or not sg_dir.exists():
        sg_dir = candidates[0]
    if hf_dir is None or not hf_dir.exists():
        hf_dir = candidates[-1]
        if hf_dir == sg_dir and len(candidates) >= 2:
            hf_dir = candidates[-2]

    if hf_dir == sg_dir:
        raise RuntimeError(f"Failed to resolve distinct HF/SG dirs under {dumper_root}")
    return hf_dir, sg_dir


def load_dump_value(path: Path):
    obj = torch.load(path, weights_only=False, map_location="cpu")
    return obj["value"] if isinstance(obj, dict) and "value" in obj else obj


def extract_dump_index(path: Path) -> int:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected dump file name: {path.name}")
    return int(match.group("dump_index"))


def choose_best_dump(paths: list[Path]) -> int:
    scored: list[tuple[int, int, int]] = []
    for path in paths:
        idx = extract_dump_index(path)
        try:
            value = load_dump_value(path)
            shape = tuple(getattr(value, "shape", ()))
            ndim = len(shape)
            score = 0
            if ndim >= 2:
                score += 100
            if ndim >= 1 and shape and shape[0] > 1:
                score += 10
            scored.append((score, idx, -idx))
        except Exception:
            scored.append((0, idx, -idx))
    scored.sort(reverse=True)
    return scored[0][1]


def resolve_indices(dump_dir: Path, compare_names: list[str], name_override: dict[str, str] | None = None) -> dict[str, int]:
    name_override = name_override or {}
    out: dict[str, int] = {}
    for name in compare_names:
        actual_name = name_override.get(name, name)
        paths = sorted(dump_dir.glob(f"forward_pass_id=0___rank=0___name={actual_name}___dump_index=*.pt"))
        out[name] = choose_best_dump(paths) if paths else -1
    return out
