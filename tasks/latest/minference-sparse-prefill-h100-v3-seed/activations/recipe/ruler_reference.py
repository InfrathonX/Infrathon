#!/usr/bin/env python3
"""Digest-bound, four-task RULER-derived diagnostic prompt protocol.

Generation executes the pinned MInference vendored RULER generators, retaining
their templates, task parameters and labels. This is a diagnostic subset, not
a published RULER score. No GPU or model weights are needed to build prompts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import struct
import subprocess
import sys
import tempfile
import types

PROTOCOL = "ruler-derived-four-task-v1"
UPSTREAM_COMMIT = "a4eb395f949ea39e871f9bc586d683390692c6be"
ESSAY_COMMIT = "021385d68d3202e37893e9d3cd29011c569abe30"
TASK_NAMES = ("niah_single_2", "niah_multikey_1", "vt", "fwe")
SOURCE_FILES = (
    "synthetic.yaml", "data/prepare.py", "data/template.py", "data/tokenizer.py",
    "data/synthetic/constants.py", "data/synthetic/niah.py",
    "data/synthetic/variable_tracking.py", "data/synthetic/freq_words_extraction.py",
    "data/synthetic/json/PaulGrahamEssays_URLs.txt", "eval/synthetic/constants.py", "eval/evaluate.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_token_sha256(ids):
    """Hash the exact model input IDs as signed int32 little-endian bytes."""
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


def derive_seed(seed, task, length, partition):
    """Domain separation avoids identical samples across lengths/partitions."""
    domain = f"{PROTOCOL}\0{seed}\0{partition}\0{task}\0{length}"
    return int.from_bytes(hashlib.sha256(domain.encode()).digest()[:4], "little")


def score_prediction(task, prediction, outputs):
    """Pinned official string_match_all, one example, percentage [0, 100].

    This is case-insensitive substring recall, not exact match; additional
    incorrect words are not penalized. Aggregate over examples before rounding
    when reproducing the official dataset-level metric.
    """
    if task not in TASK_NAMES or not outputs:
        raise ValueError("unsupported task or empty reference labels")
    prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()
    return round(100 * sum(str(r).lower() in prediction.lower() for r in outputs) / len(outputs), 2)


def score_predictions(task, predictions, references):
    """Official dataset-level rounding, after averaging unrounded recall."""
    if task not in TASK_NAMES or not predictions or len(predictions) != len(references):
        raise ValueError("unsupported task or mismatched/empty evaluation")
    scores = []
    for prediction, outputs in zip(predictions, references):
        if not outputs:
            raise ValueError("empty reference labels")
        prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()
        scores.append(sum(str(r).lower() in prediction.lower() for r in outputs) / len(outputs))
    return round(100 * sum(scores) / len(scores), 2)


def load_verified_prompts(root):
    """Load prompts after verifying the manifest and every exact input digest."""
    root = Path(root)
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("unsupported RULER protocol")
    if manifest.get("source_commit") != UPSTREAM_COMMIT:
        raise ValueError("unexpected RULER source pin")
    path = root / "prompts.jsonl"
    if sha256_file(path) != manifest["prompts_sha256"]:
        raise ValueError("prompts.jsonl digest mismatch")
    prompts = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(prompts) != manifest["sample_count"]:
        raise ValueError("prompt count mismatch")
    seen = set()
    for prompt in prompts:
        if prompt["id"] in seen or prompt["task"] not in TASK_NAMES:
            raise ValueError("duplicate ID or unknown task")
        seen.add(prompt["id"])
        ids = prompt["input_ids"]
        if not ids or not all(type(t) is int and 0 <= t < 2**31 for t in ids):
            raise ValueError("invalid token IDs")
        if input_token_sha256(ids) != prompt["input_token_sha256"]:
            raise ValueError("input token digest mismatch")
        if hashlib.sha256(prompt["input"].encode()).hexdigest() != prompt["input_text_sha256"]:
            raise ValueError("input text digest mismatch")
        if len(ids) != prompt["actual_input_length"] or len(ids) + prompt["tokens_to_generate"] > prompt["length"]:
            raise ValueError("generation reservation violated")
        if not prompt["outputs"] or not all(isinstance(x, str) and x for x in prompt["outputs"]):
            raise ValueError("invalid official labels")
    return prompts


def verified_sources(upstream):
    """Reject a dirty/mismatched selected source, even at the correct HEAD."""
    upstream = Path(upstream)
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"expected MInference commit {UPSTREAM_COMMIT}, got {commit}")
    sources = {}
    for name in SOURCE_FILES:
        path = upstream / "experiments/ruler" / name
        blob = subprocess.check_output(["git", "-C", str(upstream), "show", f"{UPSTREAM_COMMIT}:experiments/ruler/{name}"])
        if path.read_bytes() != blob:
            raise ValueError(f"selected source differs from pinned commit: {name}")
        sources[name] = hashlib.sha256(blob).hexdigest()
    return sources


def stage_essay_corpus(upstream, essay_checkout, output):
    """Freeze only the GitHub essays listed by pinned official RULER.

    The live paulgraham.com HTML half of the upstream downloader is deliberately
    excluded and disclosed. Lexicographic path order replaces filesystem glob
    order; each immutable source blob is hashed. No silent download failures.
    """
    source_list = Path(upstream) / "experiments/ruler/data/synthetic/json/PaulGrahamEssays_URLs.txt"
    prefix = "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/"
    paths = sorted({url.strip()[len(prefix):] for url in source_list.read_text().splitlines() if url.startswith(prefix)})
    if not paths:
        raise ValueError("pinned official essay URL list has no GitHub sources")
    entries, texts = [], []
    for path in paths:
        blob = subprocess.check_output(["git", "-C", str(essay_checkout), "show", f"{ESSAY_COMMIT}:{path}"])
        texts.append(blob.decode("utf-8"))
        entries.append({"path": path, "sha256": hashlib.sha256(blob).hexdigest()})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"text": "".join(texts)}, ensure_ascii=False) + "\n")
    return {
        "repository": "https://github.com/gkamradt/LLMTest_NeedleInAHaystack",
        "commit": ESSAY_COMMIT, "selection": "GitHub entries of pinned RULER URL list only",
        "ordering": "lexicographic source path, direct concatenation",
        "excluded": "live paulgraham.com HTML URLs", "files": entries,
        "sha256": sha256_file(output), "word_count": len("".join(texts).split()),
    }


def _worker(job):
    """Execute the official generator with only audited transport/budget changes."""
    import yaml
    from transformers import AutoTokenizer

    upstream, tokenizer_path = Path(job["upstream"]), job["tokenizer"]
    ruler = upstream / "experiments/ruler"
    task_configs = yaml.safe_load((ruler / "synthetic.yaml").read_text())
    config = task_configs[job["task"]]
    base = runpy.run_path(str(ruler / "data/synthetic/constants.py"))["TASKS"][config["task"]]
    model_templates = runpy.run_path(str(ruler / "data/template.py"))["Templates"]
    template = model_templates[job["template"]].format(task_template=base["template"]) + base["answer_prefix"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 2**30

    # Original HFTokenizer counts tokenize() without BOS. The downstream model
    # receives encode(add_special_tokens=True), so reserve the BOS explicitly.
    special_overhead = len(tokenizer.encode("", add_special_tokens=True)) - len(tokenizer.encode("", add_special_tokens=False))
    generator_limit = job["length"] - special_overhead - job["budget_margin"]
    token_module = types.ModuleType("tokenizer")
    class HFTokenizer:
        def text_to_tokens(self, text):
            return tokenizer.tokenize(text)
        def tokens_to_text(self, tokens):
            return tokenizer.convert_tokens_to_string(tokens)
    def select_tokenizer(kind, path):
        if kind != "hf" or str(path) != str(tokenizer_path):
            raise ValueError("only the pinned local HF tokenizer is supported")
        return HFTokenizer()
    token_module.select_tokenizer = select_tokenizer
    sys.modules["tokenizer"] = token_module
    # NeMo is only a JSONL I/O dependency in these scripts, not a generator.
    for name in ("nemo", "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.parts", "nemo.collections.asr.parts.utils", "nemo.collections.asr.parts.utils.manifest_utils"):
        sys.modules[name] = types.ModuleType(name)
    io_module = sys.modules["nemo.collections.asr.parts.utils.manifest_utils"]
    io_module.read_manifest = lambda path: [json.loads(x) for x in Path(path).read_text().splitlines() if x]
    def write_manifest(path, rows):
        with Path(path).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    io_module.write_manifest = write_manifest

    with tempfile.TemporaryDirectory(prefix="ruler-pinned-") as directory:
        staged = Path(directory)
        (staged / "json").mkdir()
        (staged / "constants.py").write_bytes((ruler / "data/synthetic/constants.py").read_bytes())
        if config["task"] == "niah":
            (staged / "json/PaulGrahamEssays.json").write_bytes(Path(job["essay_corpus"]).read_bytes())
        source = (ruler / f"data/synthetic/{config['task']}.py").read_text()
        if config["task"] == "freq_words_extraction":
            preimage = "input_max_len = max_seq_length"
            if source.count(preimage) != 2:
                raise ValueError("unexpected pinned FWE budget preimage")
            source = source.replace(preimage, "input_max_len = max_seq_length - tokens_to_generate")
        script = staged / f"{config['task']}.py"
        script.write_text(source)
        argv = [str(script), "--save_dir", str(staged), "--save_name", job["task"],
                "--tokenizer_path", tokenizer_path, "--tokenizer_type", "hf",
                "--max_seq_length", str(generator_limit), "--tokens_to_generate", str(base["tokens_to_generate"]),
                "--num_samples", str(job["samples"]), "--random_seed", str(job["derived_seed"]), "--template", template]
        for key, value in config["args"].items():
            argv.extend(("--" + key, str(value)))
        old_argv = sys.argv
        sys.argv = argv
        sys.path.insert(0, str(staged))
        try:
            with open(job["log"], "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                runpy.run_path(str(script), run_name="__main__")
        finally:
            sys.argv = old_argv
            sys.path.remove(str(staged))
        rows = io_module.read_manifest(staged / job["task"] / "validation.jsonl")
    prompts = []
    for row in rows:
        ids = tokenizer.encode(row["input"], add_special_tokens=True)
        if len(ids) + base["tokens_to_generate"] > job["length"]:
            raise ValueError("exact token budget exceeded; refuse to truncate the query")
        prompts.append({
            "id": f"{job['partition']}-{job['task']}-{job['length']}-r{row['index']}",
            "task": job["task"], "length": job["length"], "sample_index": row["index"],
            "input_ids": ids, "input": row["input"], "outputs": row["outputs"],
            "tokens_to_generate": base["tokens_to_generate"], "actual_input_length": len(ids),
            "input_token_sha256": input_token_sha256(ids),
            "input_text_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
            "derived_seed": job["derived_seed"], "official_length": row["length"],
            "generator_context_limit": generator_limit,
        })
    return prompts, {"task": job["task"], "length": job["length"], "derived_seed": job["derived_seed"],
                     "template": template, "parameters": config["args"], "tokens_to_generate": base["tokens_to_generate"]}


def build_prompts(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "prompts.jsonl").exists():
        raise ValueError("output already contains prompts; choose a fresh output directory")
    sources = verified_sources(args.upstream)
    implementation = {"ruler_reference.py": sha256_file(Path(__file__)),
                      "build_ruler_prompts.py": sha256_file(Path(__file__).with_name("build_ruler_prompts.py"))}
    essay = output / "PaulGrahamEssays.json"
    corpus = stage_essay_corpus(args.upstream, args.essay_checkout, essay)
    tokenizer_files = {}
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
        path = Path(args.tokenizer) / filename
        if not path.is_file():
            raise ValueError(f"missing local tokenizer metadata: {filename}")
        tokenizer_files[filename] = sha256_file(path)
    jobs = []
    for task in args.tasks.split(","):
        if task not in TASK_NAMES:
            raise ValueError(f"unsupported task: {task}")
        for length in map(int, args.lengths.split(",")):
            if length < 4096:
                raise ValueError("use context lengths >=4096 for this long-context protocol")
            jobs.append({"upstream": str(Path(args.upstream).resolve()), "tokenizer": str(Path(args.tokenizer).resolve()),
                         "essay_corpus": str(essay.resolve()), "task": task, "length": length,
                         "samples": args.samples_per_task, "derived_seed": derive_seed(args.seed, task, length, args.partition),
                         "partition": args.partition, "template": args.template, "budget_margin": args.budget_margin,
                         "log": str((output / f"generation-{task}-{length}.log").resolve())})
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_worker, jobs))
    prompts = [row for rows, _ in results for row in rows]
    path = output / "prompts.jsonl"
    with path.open("w") as stream:
        for row in prompts:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    dependencies = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "numpy", "scipy", "nltk", "wonderwords", "PyYAML", "tqdm")}
    import nltk
    import wonderwords
    resource_hashes = {}
    for prefix, resource in (("nltk/english", Path(nltk.data.find("tokenizers/punkt_tab/english"))),
                             ("wonderwords", Path(wonderwords.__file__).parent)):
        for file in sorted(resource.rglob("*.txt")):
            resource_hashes[prefix + "/" + str(file.relative_to(resource))] = sha256_file(file)
    manifest = {
        "protocol": PROTOCOL, "description": "RULER-derived four-task diagnostic, not the complete benchmark",
        "source_repository": "https://github.com/microsoft/MInference", "source_commit": UPSTREAM_COMMIT,
        "source_sha256": sources, "generator_jobs": [metadata for _, metadata in results],
        "implementation_sha256": implementation,
        "essay_corpus": corpus, "tokenizer_sha256": tokenizer_files, "tokenizer_snapshot": Path(args.tokenizer).name,
        "token_convention": "HF encode(add_special_tokens=True), exact saved IDs; SHA256 signed int32 little-endian",
        "seed": args.seed, "partition": args.partition, "samples_per_task_length": args.samples_per_task,
        "sample_count": len(prompts), "prompts_sha256": sha256_file(path), "dependencies": dependencies,
        "resource_sha256": resource_hashes, "python_version": sys.version.split()[0],
        "adaptations": [
            "Bypass prepare.py, which ignores --random_seed and overwrites it with 42 + chunk_idx; pass domain-derived seed directly to generators.",
            "Replace NeMo JSONL transport with equivalent local JSONL I/O; local-only HF tokenizer with original tokenize() counting.",
            "FWE: subtract tokens_to_generate at both input_max_len assignments; preserve the complete question and answer prefix.",
            f"Reserve tokenizer-added special tokens plus {args.budget_margin} safety tokens for generator boundary counting; never truncate generated prompts.",
            "Use immutable GitHub essay subset and deterministic concatenation order; omit upstream downloader's mutable HTML sources.",
        ],
        "official_metric": "eval/synthetic/constants.py string_match_all: case-insensitive reference substring recall, percentage; no false-positive penalty",
        "scope": "one model/tokenizer, four task types; downstream seed diagnostic and real-QKV capture; no general accuracy guarantee",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    load_verified_prompts(output)
    print(json.dumps({"output": str(output), "sample_count": len(prompts), "prompts_sha256": manifest["prompts_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--essay-checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default=",".join(TASK_NAMES))
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--partition", default="diagnostic")
    parser.add_argument("--template", default="llama-3", choices=("llama-3", "base"))
    parser.add_argument("--budget-margin", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.samples_per_task < 1 or args.workers < 1 or args.budget_margin < 0:
        parser.error("samples/workers must be positive and budget margin nonnegative")
    build_prompts(args)


if __name__ == "__main__":
    main()
