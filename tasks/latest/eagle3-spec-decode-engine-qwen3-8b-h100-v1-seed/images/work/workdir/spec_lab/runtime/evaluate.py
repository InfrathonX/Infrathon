"""Public timing evaluator and in-process development oracle."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from .api import EngineContract, GenerationResult, GenerationSpec, validate_prompts, validate_result
from .loader import build_contract, default_head_dir, load_draft_head, load_target, load_tokenizer
from .oracle import judge_outputs, teacher_forced_check
from .prompts import load_pool, render

_WALL_CLOCK = time.perf_counter


@dataclass
class PromptMeasurement:
    prompt_ids: list[str]
    generated_lengths: list[int]
    wall_s: float
    peak_memory_bytes: int
    trace: dict[str, Any]


@dataclass
class CellResult:
    name: str
    measurements: list[PromptMeasurement]
    outputs: list[GenerationResult] = field(repr=False)
    prompt_batches: list[list[torch.Tensor]] = field(repr=False)

    @property
    def total_tokens(self) -> int:
        return sum(sum(m.generated_lengths) for m in self.measurements)

    @property
    def total_wall_s(self) -> float:
        return sum(m.wall_s for m in self.measurements)

    @property
    def tokens_per_s(self) -> float:
        return self.total_tokens / self.total_wall_s

    @property
    def peak_memory_bytes(self) -> int:
        return max(m.peak_memory_bytes for m in self.measurements)

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "measurements": [asdict(m) for m in self.measurements],
            "total_tokens": self.total_tokens,
            "total_wall_s": self.total_wall_s,
            "tokens_per_s": self.tokens_per_s,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


def _as_batch(item) -> list[torch.Tensor]:
    return [item] if isinstance(item, torch.Tensor) else list(item)


def timed_generate(engine, batch, spec, contract):
    """Time generation, deferred validation, output copies and all queued CUDA work."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = _WALL_CLOCK()
    result = engine.generate(batch, spec)
    torch.cuda.synchronize()
    lengths = validate_result(batch, result, spec, contract)
    result = GenerationResult([output.detach().clone() for output in result.token_ids], json.loads(json.dumps(result.trace, allow_nan=False)))
    torch.cuda.synchronize()
    return result, lengths, _WALL_CLOCK() - start


def run_cell(engine, prompts, spec, contract, warmup_prompt, name="cell") -> CellResult:
    warmup = _as_batch(warmup_prompt)
    validate_prompts(warmup, contract, spec)
    warmup_result = engine.generate(warmup, spec)
    validate_result(warmup, warmup_result, spec, contract)

    measurements, outputs, batches = [], [], []
    for index, item in enumerate(prompts):
        batch = _as_batch(item)
        validate_prompts(batch, contract, spec)
        result, lengths, wall = timed_generate(engine, batch, spec, contract)
        measurements.append(PromptMeasurement(
            prompt_ids=[str(index * contract.max_batch_size + j) for j in range(len(batch))],
            generated_lengths=lengths,
            wall_s=wall,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            trace=result.trace,
        ))
        outputs.append(result)
        batches.append(batch)
    return CellResult(name, measurements, outputs, batches)


def score(cells: Sequence[CellResult]) -> float:
    if not cells:
        raise ValueError("at least one cell is required")
    throughputs = [cell.tokens_per_s for cell in cells]
    if any(value <= 0 for value in throughputs):
        raise ValueError("all cells must have positive throughput")
    return math.exp(sum(math.log(value) for value in throughputs) / len(throughputs))


def _load_module(path_or_name: str):
    path = Path(path_or_name)
    if path.is_file():
        package_root = Path(__file__).resolve().parents[1]
        try:
            relative = path.resolve().relative_to(package_root)
            module_name = ".".join(relative.with_suffix("").parts)
            return importlib.import_module(module_name)
        except ValueError:
            spec = importlib.util.spec_from_file_location("mlsysforge_candidate", path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return importlib.import_module(path_or_name)


def _draft_identity(head: Path) -> tuple[str, str]:
    for name in ("MANIFEST.json", "sha256s.json"):
        manifest = head / name
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            model_id = data.get("repo_id")
            if not model_id:
                model_id = "AngelSlim/Qwen3-8B_eagle3" if "angel" in str(head).lower() else "Tengyunw/qwen3_8b_eagle3"
            return str(data["revision"]), str(model_id)
    raise ValueError(f"head path has no manifest with pinned revision: {head}")


def _default_head() -> Path:
    return default_head_dir()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="candidate module name or candidate.py path")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    head = args.head or _default_head()
    draft_revision, draft_model_id = _draft_identity(head)
    contract = build_contract(draft_revision, draft_model_id=draft_model_id)
    target = load_target(contract)
    tokenizer = load_tokenizer(contract)
    draft = load_draft_head(head, contract)
    module = _load_module(args.engine)
    engine = module.build(target, draft, tokenizer, contract)
    pool_rows = load_pool(args.pool)
    requested_rows = args.n + 1 if args.batch == 1 else (args.n + 1) * args.batch
    rows = pool_rows[:requested_rows]
    tensors = [render(row["messages"], tokenizer).to(contract.device) for row in rows]
    warmup = tensors[:args.batch]
    measured = tensors[args.batch:]
    batches = [measured[i:i + args.batch] for i in range(0, len(measured), args.batch)]
    if len(warmup) != args.batch or any(len(batch) != args.batch for batch in batches):
        raise ValueError("pool does not contain one full warm-up group plus the requested full batches")
    cell = run_cell(engine, batches, GenerationSpec(args.max_new), contract, warmup)

    policy = json.loads(Path(__file__).with_name("policy_public.json").read_text(encoding="utf-8"))
    oracle_records = []
    for batch, result in zip(cell.prompt_batches, cell.outputs):
        for prompt, output in zip(batch, result.token_ids):
            record = teacher_forced_check(target, output, prompt.numel(), policy["eps_hard"])
            record.prompt_ids = [prompt]
            record.result = GenerationResult([output])
            record.spec = GenerationSpec(args.max_new)
            record.contract = contract
            oracle_records.append(record)
    oracle = judge_outputs(oracle_records, policy["eps_hard"], policy["p_soft"], policy.get("tiers"))
    oracle_per_output = [
        {"output_index": index, **judge_outputs([record], policy["eps_hard"], policy["p_soft"], policy.get("tiers"))}
        for index, record in enumerate(oracle_records)
    ]

    determinism = []
    for prompt in list(reversed(tensors[1:3])):
        first = engine.generate([prompt], GenerationSpec(args.max_new))
        second = engine.generate([prompt], GenerationSpec(args.max_new))
        determinism.append(torch.equal(first.token_ids[0], second.token_ids[0]))
    payload = {
        "contract": {**asdict(contract), "dtype": str(contract.dtype)},
        "cell": cell.public_dict(),
        "score": score([cell]),
        "oracle": oracle,
        "oracle_per_output": oracle_per_output,
        "o5_determinism": {"passed": all(determinism), "checks": determinism},
        "policy": policy,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
