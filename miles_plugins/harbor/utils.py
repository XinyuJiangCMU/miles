import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_result_paths(path: str | None) -> list[Path]:
    if path is None:
        raise ValueError("Harbor results path is required. Use --prompt-data to point to a manifest or directory.")

    p = Path(path)
    if p.is_dir():
        results = sorted(p.rglob("result.json"))
        if not results:
            raise FileNotFoundError(f"No result.json found under directory: {p}")
        return results

    if not p.exists():
        raise FileNotFoundError(f"Harbor results manifest not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".txt", ".list"}:
        return _load_paths_from_txt(p)
    if suffix == ".jsonl":
        return _load_paths_from_jsonl(p)
    if suffix == ".json":
        return _load_paths_from_json(p)

    # Fallback: treat as a single result.json path
    return [p]


def _load_paths_from_txt(path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    if not paths:
        raise ValueError(f"No valid paths found in {path}")
    return paths


def _load_paths_from_jsonl(path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if isinstance(data, str):
            paths.append(Path(data))
            continue
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSONL entry in {path}: {data}")
        value = data.get("result_path") or data.get("path")
        if value is None:
            raise ValueError(f"JSONL entry missing result_path/path in {path}: {data}")
        paths.append(Path(value))
    if not paths:
        raise ValueError(f"No valid paths found in {path}")
    return paths


def _load_paths_from_json(path: Path) -> list[Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [Path(x) for x in data]
    if isinstance(data, dict):
        value = data.get("result_paths") or data.get("paths")
        if isinstance(value, list):
            return [Path(x) for x in value]
    raise ValueError(f"Unsupported JSON format in {path}")


def read_trial_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Trial result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_tokens_from_rollout_detail(
    prompt_token_ids: list[list[int]],
    completion_token_ids: list[list[int]],
) -> tuple[list[int], int]:
    tokens: list[int] = []
    response_length = 0

    for idx, completion in enumerate(completion_token_ids):
        prompt = prompt_token_ids[idx] if idx < len(prompt_token_ids) else []

        if not tokens:
            tokens = list(prompt)
        else:
            tokens = _append_prompt_suffix(tokens, prompt)

        tokens.extend(completion)
        response_length += len(completion)

    return tokens, response_length


def _append_prompt_suffix(tokens: list[int], prompt: list[int]) -> list[int]:
    if not prompt:
        return tokens

    if len(prompt) >= len(tokens) and prompt[: len(tokens)] == tokens:
        return tokens + prompt[len(tokens) :]

    max_k = min(len(tokens), len(prompt))
    for k in range(max_k, 0, -1):
        if tokens[-k:] == prompt[:k]:
            return tokens + prompt[k:]

    return tokens + prompt


def flatten_logprobs(logprobs: list[list[float]] | None) -> list[float] | None:
    if logprobs is None:
        return None
    flat: list[float] = []
    for item in logprobs:
        flat.extend(item)
    return flat


def maybe_shuffle(paths: list[Path], seed: int, epoch: int) -> list[Path]:
    shuffled = list(paths)
    random.seed(seed + epoch)
    random.shuffle(shuffled)
    return shuffled

