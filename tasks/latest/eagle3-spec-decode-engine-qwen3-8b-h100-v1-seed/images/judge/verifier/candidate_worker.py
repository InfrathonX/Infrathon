#!/usr/bin/env python3
"""Clean unprivileged process that imports and times one candidate."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pwd
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

SPEC_ROOT = Path(os.environ.get("EAGLE3_SPEC_ROOT", "/app/workdir/spec_lab"))
MAX_REQUEST_BYTES = 16 * 1024 * 1024
T = TypeVar("T")


def reject_constant(value: str) -> None:
    """Reject JSON NaN and infinity."""
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_stdin() -> dict[str, Any]:
    """Read one bounded worker request object."""
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("worker request is empty or oversized")
    value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("worker request must be an object")
    return value


def drop_privileges() -> None:
    """Drop supplementary groups, GID, then UID before candidate import."""
    if os.geteuid() != 0:
        return
    account = pwd.getpwnam("agent")
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write one finite worker response through an atomic rename."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def call_with_alarm(seconds: float, operation: Callable[[], T], label: str) -> T:
    """Apply a process-local wall alarm around build or generate."""
    if seconds <= 0:
        raise ValueError(f"{label} timeout must be positive")

    def expired(_signum: int, _frame: Any) -> None:
        """Turn a process alarm into a bounded candidate failure."""
        raise TimeoutError(f"{label} cap exceeded")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return operation()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the controller-owned worker command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load trusted inputs, drop privileges, then run the selected engine."""
    args = parse_args(argv)
    connection = socket.socket(fileno=args.control_fd)
    channel = connection.makefile("rwb")
    # Bind transport, serialization and clock before candidate module import.
    read_line, write, flush = channel.readline, channel.write, channel.flush
    loads, dumps = json.loads, json.dumps
    clock = time.perf_counter
    def receive():
        """Receive one bounded protocol frame."""
        raw = read_line(MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            raise ValueError("invalid bounded request frame")
        value = loads(raw, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("request frame must be an object")
        return value
    def send(value):
        """Flush one finite protocol frame."""
        write((dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode())
        flush()
    request = receive()
    if set(request) != {"policy"}:
        raise ValueError("build header may contain only policy")

    # Bind every trusted runtime operation before the candidate can import.
    sys.path.insert(0, str(SPEC_ROOT))
    from runtime.api import GenerationSpec, validate_prompts, validate_result
    from runtime.loader import (
        build_contract,
        default_head_dir,
        load_draft_head,
        load_target,
        load_tokenizer,
    )
    from runtime.prompts import render

    bound_validate_prompts = validate_prompts
    bound_validate_result = validate_result
    bound_render = render
    bound_synchronize = torch.cuda.synchronize
    bound_reset_peak = torch.cuda.reset_peak_memory_stats
    bound_peak = torch.cuda.max_memory_allocated
    bound_current_stream = torch.cuda.current_stream
    bound_thread_enumerate = threading.enumerate

    drop_privileges()

    head = default_head_dir()
    head_manifest = json.loads((head / "MANIFEST.json").read_text(encoding="utf-8"))
    contract = build_contract(
        head_manifest["revision"],
        draft_model_id=head_manifest["repo_id"],
    )
    target = load_target(contract)
    tokenizer = load_tokenizer(contract)
    draft = load_draft_head(head, contract)

    # Loading trusted Torch/Transformers inputs may lazily start Python helper
    # threads. Capture the baseline only after that work, at the exact boundary
    # before any candidate module is imported.
    trusted_thread_ids = {
        thread.ident
        for thread in bound_thread_enumerate()
        if thread.ident is not None
    }

    def candidate_threads_idle() -> bool:
        """Return whether no candidate-created Python thread is still alive."""
        return all(
            thread.ident in trusted_thread_ids
            for thread in bound_thread_enumerate()
            if thread.ident is not None
        )

    candidate_root = args.candidate_root.resolve(strict=True)
    sys.path.insert(0, str(candidate_root))
    module_name = f"solutions.engine_v{args.candidate_version:03d}.candidate"
    module = importlib.import_module(module_name)
    build_cap = float(request["policy"]["build_cap_s"])
    build_start = clock()
    engine = call_with_alarm(
        build_cap,
        lambda: module.build(target, draft, tokenizer, contract),
        "build",
    )
    build_seconds = clock() - build_start
    if build_seconds > build_cap:
        raise TimeoutError("build cap exceeded")

    send({"kind": "built", "build_s": build_seconds})
    while True:
        frame = receive()
        if frame.get("kind") == "done":
            break
        name = frame["cell"]
        generation_spec = GenerationSpec(int(frame["max_new_tokens"]))
        prompts = [
            bound_render(row["messages"], tokenizer, bool(frame["enable_thinking"])).to(contract.device)
            for row in frame["rows"]
        ]
        bound_validate_prompts(prompts, contract, generation_spec)
        prompt_lengths = [int(prompt.numel()) for prompt in prompts]
        bound_synchronize()
        bound_reset_peak()
        wall_cap = float(request["policy"]["wall_caps_s"][name])
        send({"kind": "ready"})
        if receive() != {"kind": "go"}:
            raise ValueError("expected generation start")

        def generate_materialized():
            """Join all queued work and copy every output inside the wall alarm."""
            start = clock()
            result = engine.generate(prompts, generation_spec)
            returned = clock()
            pending_current_stream = not bound_current_stream().query()
            threads_idle = candidate_threads_idle()
            # Query precedes the trusted join. All other CUDA streams are also
            # joined and charged in this same request, whether query sees them.
            bound_synchronize()
            joined = clock()
            lengths = bound_validate_result(prompts, result, generation_spec, contract)
            outputs = [output.detach().cpu().tolist() for output in result.token_ids]
            # Even deferred properties/trace serialization execute before the
            # supervisor receives completion, and hence before its timer ends.
            trace = loads(dumps(result.trace, allow_nan=False))
            bound_synchronize()
            return {"kind": "result", "outputs": outputs,
                    "prompt_lengths": prompt_lengths,
                    "measurement": {"wall_s": clock() - start,
                        "generated_lengths": lengths,
                        "device_idle": threads_idle,
                        "pending_current_stream_at_return": pending_current_stream,
                        "return_to_sync_s": joined - returned,
                        "peak_memory_bytes": bound_peak(), "trace": trace}}

        response = call_with_alarm(wall_cap, generate_materialized, f"{name} generate")
        send(response)
        del frame, prompts, response
    channel.close()
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
