#!/usr/bin/env python3
"""Unprivileged self-spawning eight-rank candidate worker (one process per GPU)."""

from __future__ import annotations
import argparse, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
from typing import Any, Mapping

WORLD = 8
MANIFEST_KEYS = {"schema_version", "suite_id", "cases"}
SUITE_IDS = {"final-holdout-v2"}


def validate_manifest(value: Mapping[str, Any], *, expected_count: int):
    if (
        set(value) != MANIFEST_KEYS
        or value.get("schema_version") != 2
        or value.get("suite_id") not in SUITE_IDS
    ):
        raise RuntimeError("manifest identity differs")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise RuntimeError("manifest case count differs")
    if any(not isinstance(x, dict) for x in cases):
        raise RuntimeError("manifest case schema differs")
    if len({str(x.get("case_id")) for x in cases}) != len(cases):
        raise RuntimeError("manifest case ids repeat")
    return cases


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def cpu_slices(world: int = WORLD) -> list[list[int]]:
    """Split the CPUs this worker may use into one contiguous slice per rank."""
    cpus = sorted(os.sched_getaffinity(0))
    per = max(1, len(cpus) // world)
    return [cpus[r * per:(r + 1) * per] or cpus for r in range(world)]


def child_environment(rank: int, port: int, cache: Path, base: Mapping[str, str] | None = None):
    source = dict(os.environ if base is None else base)
    visible = source.get("CUDA_VISIBLE_DEVICES", ",".join(str(x) for x in range(WORLD)))
    allowed = (
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "HOME",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "TMPDIR",
        "FLUXMOE_DEBUG",
    )
    env = {key: source[key] for key in allowed if key in source}
    env.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        CUDA_VISIBLE_DEVICES=visible,
        TRITON_CACHE_DIR=str(cache / f"rank{rank}"),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        NCCL_IB_DISABLE="1",
        NCCL_SOCKET_IFNAME="lo",
    )
    return env


def kill_process_group(process: subprocess.Popen[Any]):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def atomic_json(path: Path, value: Mapping[str, Any]):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def child_main(args, manifest):
    if args.cpus:
        os.sched_setaffinity(0, {int(x) for x in args.cpus.split(",")})
    sys.path.insert(0, str(args.runtime_root))
    from moe_ag_runtime import run_suite

    cases = validate_manifest(manifest, expected_count=args.expected_count)
    progress = (lambda line: print(line, flush=True)) if args.progress else None
    result = run_suite(
        args.candidate,
        cases,
        trials=args.trials,
        warmups=args.warmups,
        fixture_dir=args.fixtures,
        progress=progress,
    )
    if int(args.rank) == 0:
        if not isinstance(result, dict):
            raise RuntimeError("rank zero produced no result")
        atomic_json(args.output, result)
    return 0


def parent_main(args, manifest):
    validate_manifest(manifest, expected_count=args.expected_count)
    port = free_loopback_port()
    payload = json.dumps(manifest).encode()
    args.cache.mkdir(parents=True, exist_ok=True)
    slices = cpu_slices()
    children = []
    deadline = time.monotonic() + args.timeout
    try:
        for rank in range(WORLD):
            (args.cache / f"rank{rank}").mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--rank", str(rank),
                "--candidate", str(args.candidate),
                "--output", str(args.output),
                "--runtime-root", str(args.runtime_root),
                "--cache", str(args.cache),
                "--trials", str(args.trials),
                "--warmups", str(args.warmups),
                "--timeout", str(args.timeout),
                "--expected-count", str(args.expected_count),
                "--cpus", ",".join(str(x) for x in slices[rank]),
            ]
            if args.fixtures is not None:
                cmd += ["--fixtures", str(args.fixtures)]
            if args.progress:
                cmd.append("--progress")
            children.append(
                subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    env=child_environment(rank, port, args.cache),
                    start_new_session=True,
                )
            )
        for child in children:
            assert child.stdin is not None
            child.stdin.write(payload)
            child.stdin.close()
        while children:
            failed = next((p for p in children if p.poll() not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError(f"rank child failed: {failed.returncode}")
            children = [p for p in children if p.poll() is None]
            if not children:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("rank children timed out")
            time.sleep(0.05)
    except BaseException:
        for child in children:
            kill_process_group(child)
        raise
    if not args.output.is_file():
        raise RuntimeError("rank zero result missing")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, default=Path("/opt/flux_moe_ag_worker"))
    p.add_argument("--cache", type=Path, default=Path("/tmp/moe-ag-cache"))
    p.add_argument("--fixtures", type=Path)
    p.add_argument("--trials", type=int, default=7)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--timeout", type=float, default=5400)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--rank", type=int, choices=tuple(range(WORLD)))
    p.add_argument("--cpus", default="")
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = json.load(sys.stdin)
    return child_main(args, manifest) if args.rank is not None else parent_main(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
