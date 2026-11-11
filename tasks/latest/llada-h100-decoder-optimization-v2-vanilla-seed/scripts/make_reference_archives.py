#!/usr/bin/env python3
"""Create reproducible ordinary submission archives for host calibration."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def add_bytes(bundle: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    bundle.addfile(info, io.BytesIO(data))


def deterministic_gzip(data: bytes) -> bytes:
    """Return gzip bytes with a cross-CPython-stable header."""
    compressed = bytearray(gzip.compress(data, compresslevel=9, mtime=0))
    compressed[9] = 3
    return bytes(compressed)


def create(path: Path, candidate: bytes, note: str) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as bundle:
        add_bytes(bundle, "FINAL_VERSION", b"1\n")
        add_bytes(bundle, "notes.md", (note.rstrip() + "\n").encode())
        add_bytes(bundle, "solutions/decoder_v001/__init__.py", b"")
        add_bytes(bundle, "solutions/decoder_v001/candidate.py", candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(deterministic_gzip(raw.getvalue()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stress-extra-mib", type=int, default=8192)
    args = parser.parse_args()
    root = args.task_root.resolve()
    artifacts = root / "artifacts"
    vanilla = root / "images/work/workdir/decoder_lab/solutions/decoder_v001/candidate.py"
    corrected = root / "host_reference/corrected_decoder/candidate.py"
    if args.stress_extra_mib <= 0:
        raise SystemExit("--stress-extra-mib must be positive")
    corrected_bytes = corrected.read_bytes()
    if corrected_bytes.count(b"THRESHOLD = 0.9") != 1:
        raise SystemExit("corrected reference threshold marker changed")
    positive_bytes = corrected_bytes.replace(b"THRESHOLD = 0.9", b"THRESHOLD = 0.85")
    negative_suffix = b'''

# Task-owned negative quality control: the full decoder/model call above still
# executes and covers every layer, then only the returned generation is
# deliberately degraded to a valid non-special textual token.
class DeliberatelyDegradedQualityDecoder:
    def __init__(self, model, tokenizer) -> None:
        self.inner = PrefixThresholdDecoder(model, tokenizer)
        ids = tokenizer("x", add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError("degraded-control tokenization failed")
        self.degraded_token_id = int(ids[0])

    def generate(self, prompt_ids, spec):
        result = self.inner.generate(prompt_ids, spec)
        # The inner decoder produces its tokens under torch.inference_mode();
        # an in-place update of that inference tensor from ordinary mode is a
        # RuntimeError, which would fail the integrity gate instead of the
        # quality gate.  Degrade a clone and return a fresh result instead.
        degraded = result.token_ids.clone()
        degraded[:, prompt_ids.shape[1]:] = self.degraded_token_id
        trace = dict(result.trace)
        trace["task_quality_control"] = "deliberately_degraded_after_real_model_call"
        return GenerationResult(token_ids=degraded, trace=trace)


def build(model, tokenizer):
    return DeliberatelyDegradedQualityDecoder(model, tokenizer)
'''
    stress_suffix = f'''

# Task-owned near-cap stress fixture. The buffer is allocated and touched
# inside the measured generation window while the corrected model path still
# runs normally. OOM is intentionally not caught and therefore fails closed.
STRESS_EXTRA_MIB = {args.stress_extra_mib}


class NearCapStressDecoder:
    def __init__(self, model, tokenizer) -> None:
        self.inner = PrefixThresholdDecoder(model, tokenizer)

    def generate(self, prompt_ids, spec):
        stress = torch.empty(
            STRESS_EXTRA_MIB * 1024 * 1024,
            dtype=torch.uint8,
            device=prompt_ids.device,
        )
        stress.fill_(1)
        result = self.inner.generate(prompt_ids, spec)
        result.trace["task_memory_stress_extra_mib"] = STRESS_EXTRA_MIB
        del stress
        return result


def build(model, tokenizer):
    return NearCapStressDecoder(model, tokenizer)
'''.encode()
    for label, source in (
        ("positive control", positive_bytes),
        ("negative control", corrected_bytes + negative_suffix),
        ("near-cap stress", corrected_bytes + stress_suffix),
    ):
        compile(source, label, "exec")
    create(
        artifacts / "vanilla-reference.tar.gz",
        vanilla.read_bytes(),
        "Host calibration: visible vanilla v001 seed.",
    )
    create(
        artifacts / "corrected-decoder-reference.tar.gz",
        corrected_bytes,
        "Host calibration: optimized reference candidate.",
    )
    create(
        artifacts / "positive-near-threshold-control.tar.gz",
        positive_bytes,
        "Precommitted positive control: independent 0.85 threshold variant.",
    )
    create(
        artifacts / "negative-quality-control.tar.gz",
        corrected_bytes + negative_suffix,
        "Precommitted negative control: real model call followed by quality degradation.",
    )
    create(
        artifacts / "near-cap-stress-reference.tar.gz",
        corrected_bytes + stress_suffix,
        f"Memory qualification: corrected model plus {args.stress_extra_mib} MiB touched buffer.",
    )
    archive_names = (
        "vanilla-reference.tar.gz",
        "corrected-decoder-reference.tar.gz",
        "positive-near-threshold-control.tar.gz",
        "negative-quality-control.tar.gz",
        "near-cap-stress-reference.tar.gz",
    )
    manifest = {
        "schema_version": 1,
        "generator": "make_reference_archives.py",
        "stress_extra_mib": args.stress_extra_mib,
        "archives": {
            name: {
                "sha256": hashlib.sha256((artifacts / name).read_bytes()).hexdigest(),
                "size_bytes": (artifacts / name).stat().st_size,
            }
            for name in archive_names
        },
    }
    (artifacts / "reference-archives-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
