"""FORCED-ORACLE engine: official EAGLE-3 in SGLang behind the task ABI.

NOT an admissible agent submission. It needs the LOCAL-ONLY forced-oracle Judge
image (oracle/paper_reference/forced_judge/), which adds an isolated SGLang
virtualenv at /opt/sglang-venv and bypasses source admission (this file uses
subprocess, environment routing and network sockets on loopback).

What it runs: SGLang's EAGLE-3 speculative decoding (the integration the EAGLE
authors point to for EAGLE-3 serving) for the frozen Qwen/Qwen3-8B target and
the frozen Tengyunw/qwen3_8b_eagle3 head, with the draft-tree configuration the
head's own model card publishes for SGLang (num_steps 6, eagle_topk 10,
num_draft_tokens 32), CUDA graphs and SGLang's default Hopper attention backend.

ABI glue only: build() starts one SGLang server process on the Judge's device
from the same read-only model mounts the Judge hands to the worker (SGLang
loads its own copy of the weights; the handed-in target/draft tensors are not
used or mutated), waits for readiness and warms it up. generate() submits all
prompts of a call as one batched /generate request (greedy, temperature 0) so
SGLang batches them, then returns prompt ++ generated tokens cut by the
canonical runtime.api.truncate_generation. Radix (prefix) caching is disabled
so no request reuses another request's KV, and SGLang's deterministic
(batch-invariant) inference mode is on so results do not depend on how many
requests share a batch.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from runtime.api import GenerationResult, truncate_generation

SGLANG_PYTHON = os.environ.get("EAGLE3_FORCED_SGLANG_PYTHON", "/opt/sglang-venv/bin/python")
TARGET_REPO = Path(os.environ.get("EAGLE3_TARGET_REPO", "/opt/models/qwen3-8b-hf"))
HEAD_DIR = Path(os.environ.get("EAGLE3_HEAD_DIR", "/opt/models/eagle3-head"))
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# Draft-tree policy published with the head for SGLang (model card launch command).
SPEC_NUM_STEPS = 6
SPEC_EAGLE_TOPK = 10
SPEC_NUM_DRAFT_TOKENS = 32
MEM_FRACTION_STATIC = 0.55
READY_TIMEOUT_S = 540.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _snapshot() -> Path:
    snapshot = TARGET_REPO / "snapshots" / TARGET_REVISION
    return snapshot if snapshot.is_dir() else TARGET_REPO


class _Server:
    def __init__(self, contract: Any) -> None:
        self.port = _free_port()
        self.log_path = Path("/tmp") / f"eagle3-forced-sglang-{os.getpid()}.log"
        env = {
            "PATH": "/opt/sglang-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRITON_CACHE_DIR": "/tmp/eagle3-forced-triton",
            "XDG_CACHE_HOME": "/tmp/eagle3-forced-cache",
            "SGLANG_DISABLE_REQUEST_LOGGING": "1",
            "PYTHONNOUSERSITE": "1",
        }
        command = [
            SGLANG_PYTHON, "-m", "sglang.launch_server",
            "--model-path", str(_snapshot()),
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", str(HEAD_DIR),
            "--speculative-num-steps", str(SPEC_NUM_STEPS),
            "--speculative-eagle-topk", str(SPEC_EAGLE_TOPK),
            "--speculative-num-draft-tokens", str(SPEC_NUM_DRAFT_TOKENS),
            "--dtype", "bfloat16",
            "--context-length", str(int(contract.max_sequence_length)),
            "--max-running-requests", str(int(contract.max_batch_size)),
            "--cuda-graph-max-bs-decode", str(int(contract.max_batch_size)),
            "--mem-fraction-static", str(MEM_FRACTION_STATIC),
            "--disable-radix-cache",
            # Batch-invariant kernels: a prompt decodes to the same tokens whether
            # it is batched alone, with 2 or with 8 requests (the Judge's O5
            # reverse-order rerun). Without it O5 fails (measured on public prompts).
            "--enable-deterministic-inference",
            "--skip-tokenizer-init",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--log-level", "warning",
        ]
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT, env=env,
        )
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("SGLang server exited during start-up:\n" + self.tail())
            try:
                status, _ = self._request("GET", "/health_generate", None, timeout=30.0)
                if status == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("SGLang server did not become ready:\n" + self.tail())
            time.sleep(1.0)

    def tail(self) -> str:
        try:
            self.log.flush()
            return self.log_path.read_bytes()[-6000:].decode(errors="replace")
        except OSError:
            return ""

    def _request(self, method: str, path: str, body: Any, *, timeout: float) -> tuple[int, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            payload = None if body is None else json.dumps(body).encode()
            headers = {} if body is None else {"Content-Type": "application/json"}
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def generate(self, input_ids: list[list[int]], max_new_tokens: int, stop_ids: list[int]) -> list[dict[str, Any]]:
        body = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": int(max_new_tokens),
                "stop_token_ids": list(stop_ids),
                "ignore_eos": False,
                "skip_special_tokens": False,
                "no_stop_trim": True,
            },
        }
        status, raw = self._request("POST", "/generate", body, timeout=3600.0)
        if status != 200:
            raise RuntimeError(f"SGLang /generate returned {status}: {raw[-2000:]!r}\n" + self.tail())
        value = json.loads(raw)
        return value if isinstance(value, list) else [value]


class Engine:
    def __init__(self, contract: Any) -> None:
        self.contract = contract
        self.stop_ids = [int(t) for t in contract.eos_token_ids]
        self.server = _Server(contract)
        # Untimed build warm-up: exercise batch sizes 1 and 8 once.
        self.server.generate([[151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]], 16, self.stop_ids)
        self.server.generate([[151644, 872, 198, 9707 + i, 151645, 198, 151644, 77091, 198] for i in range(8)], 16, self.stop_ids)

    def generate(self, prompt_ids: list[torch.Tensor], spec: Any) -> GenerationResult:
        prompts = [p.detach().to("cpu", torch.int64) for p in prompt_ids]
        responses = self.server.generate([p.tolist() for p in prompts], spec.max_new_tokens, self.stop_ids)
        if len(responses) != len(prompts):
            raise RuntimeError("SGLang returned a different number of outputs")
        outputs = []
        verify_steps = 0
        for prompt, response in zip(prompts, responses):
            generated = response.get("output_ids")
            if not isinstance(generated, list) or not generated:
                raise RuntimeError(f"SGLang response without output_ids: {str(response)[:500]}")
            meta = response.get("meta_info") or {}
            verify_steps += int(meta.get("spec_verify_ct") or 0)
            tokens = torch.cat([prompt, torch.tensor(generated, dtype=torch.int64)])
            tokens = truncate_generation(tokens, prompt.numel(), spec.max_new_tokens, self.stop_ids)
            outputs.append(tokens.to(self.contract.device))
        generated_total = sum(int(o.numel() - p.numel()) for o, p in zip(outputs, prompts))
        trace = {
            "engine": "sglang-eagle3-forced-oracle",
            "spec_verify_steps": verify_steps,
            "mean_accept_length": (generated_total / verify_steps) if verify_steps else None,
        }
        return GenerationResult(outputs, trace=trace)


def build(target: Any, draft: Any, tokenizer: Any, contract: Any) -> Engine:
    del target, draft, tokenizer  # SGLang loads its own copy from the same read-only mounts.
    return Engine(contract)
