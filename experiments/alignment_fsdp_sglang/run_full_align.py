#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

from utils.dump_resolver import resolve_dump_dirs, resolve_indices
from utils.run_manifest import now_stamp, write_json, write_text
from utils.tensor_specs import ALL_COMPARE_NAMES, HF_NAME_OVERRIDE, SG_NAME_OVERRIDE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Miles FSDP vs SGLang alignment runner.")
    parser.add_argument("--server-gpu", type=str, default="4")
    parser.add_argument("--fsdp-gpu", type=str, default="5")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--server-attention-backend", type=str, default="triton")
    parser.add_argument("--fsdp-attn-implementation", type=str, default="triton")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--true-on-policy", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("/app/results/miles_alignment_runs"))
    parser.add_argument("--server-start-timeout", type=int, default=180)
    parser.add_argument("--server-stop-timeout", type=int, default=20)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=Path("/app/true_on_policy/miles/experiments/alignment_fsdp_sglang/configs/qwen3_0p6b_debug.json"),
    )
    parser.add_argument("--focus", choices=["full", "layer0", "layer1", "last_layer", "rope"], default="layer0")
    return parser.parse_args()


def build_pythonpath() -> str:
    base = "/app/sglang/python:/app/true_on_policy/miles"
    current = os.environ.get("PYTHONPATH", "")
    return f"{base}:{current}" if current else base


def make_env(cuda_visible_devices: str, dumper_root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env["HIP_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env["PYTHONPATH"] = build_pythonpath()
    env["SGLANG_DUMPER_ENABLE"] = "1"
    env["SGLANG_DUMPER_DIR"] = str(dumper_root)
    env["SGLANG_DUMPER_WRITE_FILE"] = "1"
    if extra:
        env.update(extra)
    return env


def shell_join(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _stream_subprocess_output(proc: subprocess.Popen, log_path: Path, prefix: str) -> threading.Thread:
    def _reader() -> None:
        with log_path.open("w", encoding="utf-8") as log_f:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                sys.stdout.write(f"{prefix}{line}")
                sys.stdout.flush()
    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


def run_cmd_with_tee(cmd: list[str], env: dict[str, str], log_path: Path, prefix: str) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd="/app",
        env=env,
        text=True,
        bufsize=1,
    )
    reader = _stream_subprocess_output(proc, log_path, prefix)
    code = proc.wait()
    reader.join(timeout=1)
    return code


def wait_for_server(host: str, port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    base_url = f"http://{host}:{port}"
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code < 500:
                return
        except Exception as e:
            last_error = e
        try:
            with socket.create_connection((host, port), timeout=1):
                time.sleep(1)
                return
        except OSError as e:
            last_error = e
        time.sleep(1)
    raise RuntimeError(f"Server not ready within {timeout_s}s: {last_error}")


def stop_process(proc: subprocess.Popen, timeout_s: int) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / f"run_{now_stamp()}"
    dumper_root = run_dir / "dumper"
    run_dir.mkdir(parents=True, exist_ok=True)
    dumper_root.mkdir(parents=True, exist_ok=True)

    server_log = run_dir / "server_stdout.txt"
    fsdp_log = run_dir / "fsdp_stdout.txt"
    compare_log = run_dir / "compare_stdout.txt"
    commands_txt = run_dir / "commands.txt"
    compare_json = run_dir / "compare.json"
    manifest_json = run_dir / "manifest.json"
    compare_detail = run_dir / "compare_detail.txt"
    hf_dir_txt = run_dir / "hf_dir.txt"
    sg_dir_txt = run_dir / "sg_dir.txt"
    hf_index_json = run_dir / "hf_index.json"
    sg_index_json = run_dir / "sg_index.json"

    server_cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--attention-backend",
        args.server_attention_backend,
        "--mem-fraction-static",
        "0.7",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--enable-deterministic-inference",
        "--disable-radix-cache",
        "--rl-on-policy-target",
        "fsdp",
        "--skip-server-warmup",
        "--disable-cuda-graph",
    ]
    server_env = make_env(
        args.server_gpu,
        dumper_root,
        extra={"SGLANG_RETURN_ORIGINAL_LOGPROB": "1"},
    )

    capture_cmd = [
        sys.executable,
        "/app/true_on_policy/miles/experiments/alignment_fsdp_sglang/capture_hf_fsdp_align.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-path",
        args.model_path,
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.fsdp_attn_implementation,
        "--dumper-root",
        str(dumper_root),
        "--prompt-config",
        str(args.prompt_config),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.batch_invariant:
        capture_cmd.append("--batch-invariant")
    if args.true_on_policy:
        capture_cmd.append("--true-on-policy")
    if args.deterministic:
        capture_cmd.append("--deterministic")
    capture_env = make_env(args.fsdp_gpu, dumper_root)

    compare_cmd_base = [
        sys.executable,
        "/app/true_on_policy/miles/experiments/alignment_fsdp_sglang/compare_tensors.py",
    ]

    write_text(
        commands_txt,
        "[server]\n"
        + shell_join(server_cmd)
        + "\n\n[capture]\n"
        + shell_join(capture_cmd)
        + "\n\n[compare]\n"
        + "<filled after dump resolution>\n",
    )

    server_proc = None
    server_reader = None
    try:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd="/app",
            env=server_env,
            text=True,
            bufsize=1,
        )
        server_reader = _stream_subprocess_output(server_proc, server_log, prefix="[server] ")
        wait_for_server(args.host, args.port, args.server_start_timeout)

        capture_returncode = run_cmd_with_tee(capture_cmd, capture_env, fsdp_log, prefix="[fsdp] ")
        if capture_returncode != 0:
            raise RuntimeError(f"capture_hf_fsdp_align failed with code {capture_returncode}")

        hf_dir, sg_dir = resolve_dump_dirs(dumper_root, server_log, fsdp_log)
        write_text(hf_dir_txt, str(hf_dir) + "\n")
        write_text(sg_dir_txt, str(sg_dir) + "\n")

        hf_indices = resolve_indices(hf_dir, ALL_COMPARE_NAMES, HF_NAME_OVERRIDE)
        sg_indices = resolve_indices(sg_dir, ALL_COMPARE_NAMES, SG_NAME_OVERRIDE)
        write_json(hf_index_json, hf_indices)
        write_json(sg_index_json, sg_indices)

        compare_cmd = compare_cmd_base + [
            "--hf-dir",
            str(hf_dir),
            "--sg-dir",
            str(sg_dir),
            "--hf-index-json",
            str(hf_index_json),
            "--sg-index-json",
            str(sg_index_json),
            "--focus",
            args.focus,
            "--output-txt",
            str(compare_log),
            "--output-json",
            str(compare_json),
        ]
        write_text(
            commands_txt,
            "[server]\n"
            + shell_join(server_cmd)
            + "\n\n[capture]\n"
            + shell_join(capture_cmd)
            + "\n\n[compare]\n"
            + shell_join(compare_cmd)
            + "\n",
        )
        compare_returncode = run_cmd_with_tee(compare_cmd, capture_env, compare_log, prefix="[compare] ")
        if compare_returncode != 0:
            raise RuntimeError(f"compare_tensors failed with code {compare_returncode}")

        if compare_log.exists():
            write_text(compare_detail, compare_log.read_text(encoding="utf-8"))

        manifest = {
            "run_dir": str(run_dir),
            "server_log": str(server_log),
            "fsdp_log": str(fsdp_log),
            "compare_log": str(compare_log),
            "commands_txt": str(commands_txt),
            "hf_dir": str(hf_dir),
            "sg_dir": str(sg_dir),
            "hf_index_json": str(hf_index_json),
            "sg_index_json": str(sg_index_json),
            "compare_json": str(compare_json),
            "compare_detail": str(compare_detail),
            "focus": args.focus,
            "model_path": args.model_path,
        }
        write_json(manifest_json, manifest)
        print(f"run_dir: {run_dir}")
        print(f"server_log: {server_log}")
        print(f"fsdp_log: {fsdp_log}")
        print(f"compare_log: {compare_log}")
        print(f"commands_txt: {commands_txt}")
        print(f"hf_dir: {hf_dir}")
        print(f"sg_dir: {sg_dir}")
        print(f"hf_index_json: {hf_index_json}")
        print(f"sg_index_json: {sg_index_json}")
    finally:
        if server_proc is not None and not args.keep_server:
            stop_process(server_proc, args.server_stop_timeout)
        if server_reader is not None:
            server_reader.join(timeout=1)


if __name__ == "__main__":
    main()
