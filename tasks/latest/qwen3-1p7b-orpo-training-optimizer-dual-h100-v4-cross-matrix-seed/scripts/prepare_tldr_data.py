#!/usr/bin/env python3
"""Freeze a deterministic, pretokenized TL;DR preference corpus.

The canonical output is safe-to-load ``safetensors`` plus canonical JSON/JSONL.
No Python pickle or moving Hugging Face branch is used.  The script intentionally
does not use ``datasets.load_dataset``: it verifies and streams the two small,
pinned Parquet source shards directly.

Run this outside the timed benchmark.  See ``docs/DATASET_FREEZE_NOTES.md`` for
the exact environment and commands used for a release freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Mapping, Sequence


DATASET_REPO_ID = "trl-lib/tldr-preference"
DATASET_REVISION = "7ae482b332979f8ce40d2a6b5f78f9eddf137efc"
DATASET_CONFIG = "default"
SOURCE_FILES: dict[str, dict[str, Any]] = {
    "train": {
        "filename": "data/train-00000-of-00001.parquet",
        "sha256": "eb719012049c4792ecd8aa47cfcc62503aefa9eedfacc4ac118b6a1cd1e88136",
        "size_bytes": 20_405_047,
        "row_count": 92_858,
    },
    "validation": {
        "filename": "data/validation-00000-of-00001.parquet",
        "sha256": "09c94323c556f14752140278379b4b047c6e46b227828ddcb4fff7b0dc806a31",
        "size_bytes": 21_419_388,
        "row_count": 86_086,
    },
}

TOKENIZER_REPO_ID = "Qwen/Qwen3-1.7B"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TOKENIZER_FILES: dict[str, dict[str, Any]] = {
    "merges.txt": {
        "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "size_bytes": 1_671_853,
    },
    "tokenizer.json": {
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "size_bytes": 11_422_654,
    },
    "tokenizer_config.json": {
        "sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "size_bytes": 9_732,
    },
    "vocab.json": {
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "size_bytes": 2_776_833,
    },
}

# This is the canonical preparation environment. Exact checks prevent an
# unnoticed tokenizer/runtime upgrade from silently changing frozen tensors.
PINNED_PACKAGES = {
    "huggingface-hub": "0.30.2",
    "numpy": "2.2.4",
    "pyarrow": "19.0.1",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.1",
    "transformers": "4.51.3",
}

PUBLIC_SOURCE_SPLIT = "train"
PRIVATE_SOURCE_SPLIT = "validation"
DEFAULT_PUBLIC_COUNT = 128
DEFAULT_PRIVATE_COUNT = 512
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 384
IGNORE_INDEX = -100
PAD_TOKEN_ID = 151_643
EOS_TOKEN_ID = 151_645
SELECTION_NAMESPACE = "edgebench:qwen3-1p7b:orpo:tldr-preference:v1"
REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _artifact_descriptor(path: Path, *, logical_size_bytes: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": path.name,
        "sha256": _sha256_file(path),
        "file_size_bytes": path.stat().st_size,
    }
    if logical_size_bytes is not None:
        descriptor["logical_size_bytes"] = logical_size_bytes
    return descriptor


def _check_environment_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    actual: dict[str, str] = {}
    problems: list[str] = []
    for package, expected in PINNED_PACKAGES.items():
        try:
            found = version(package)
        except PackageNotFoundError:
            problems.append(f"{package} is not installed (expected {expected})")
            continue
        actual[package] = found
        if found != expected:
            problems.append(f"{package}=={found} (expected {package}=={expected})")
    if problems:
        joined = "\n  - ".join(problems)
        raise RuntimeError(
            "The canonical freeze requires the pinned preprocessing environment:\n"
            f"  - {joined}\n"
            "Use the uv command in docs/DATASET_FREEZE_NOTES.md."
        )
    return actual


def _verify_file(path: Path, expected: Mapping[str, Any], label: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != int(expected["size_bytes"]):
        raise RuntimeError(
            f"{label} has {actual_size} bytes; expected {expected['size_bytes']}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected["sha256"]:
        raise RuntimeError(
            f"{label} SHA-256 is {actual_sha256}; expected {expected['sha256']}"
        )


def _resolve_sources(cache_dir: str | None, local_files_only: bool) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    result: dict[str, Path] = {}
    for split, expected in SOURCE_FILES.items():
        path = Path(
            hf_hub_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                filename=expected["filename"],
                revision=DATASET_REVISION,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
        _verify_file(path, expected, f"{DATASET_REPO_ID}/{expected['filename']}")
        result[split] = path
    return result


def _resolve_tokenizer(cache_dir: str | None, local_files_only: bool) -> tuple[Any, dict[str, Any]]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot_path = Path(
        snapshot_download(
            repo_id=TOKENIZER_REPO_ID,
            revision=TOKENIZER_REVISION,
            allow_patterns=sorted(TOKENIZER_FILES),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
    resolved_files: dict[str, Any] = {}
    for filename, expected in TOKENIZER_FILES.items():
        path = snapshot_path / filename
        if not path.is_file():
            raise RuntimeError(f"Pinned tokenizer file is missing: {path}")
        _verify_file(path, expected, f"{TOKENIZER_REPO_ID}/{filename}")
        resolved_files[filename] = {
            "sha256": expected["sha256"],
            "size_bytes": expected["size_bytes"],
        }

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id != EOS_TOKEN_ID:
        raise RuntimeError(
            f"Unexpected eos_token_id={tokenizer.eos_token_id}; expected {EOS_TOKEN_ID}"
        )
    if tokenizer.pad_token_id != PAD_TOKEN_ID:
        raise RuntimeError(
            f"Unexpected pad_token_id={tokenizer.pad_token_id}; expected {PAD_TOKEN_ID}"
        )
    if not tokenizer.is_fast:
        raise RuntimeError("The canonical freeze requires the pinned fast tokenizer")
    return tokenizer, resolved_files


def _record_content_sha256(prompt: str, chosen: str, rejected: str) -> str:
    payload = json.dumps(
        {"chosen": chosen, "prompt": prompt, "rejected": rejected},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _selection_rank(split: str, row_index: int, content_sha256: str) -> str:
    identity = f"{split}:{row_index}:{content_sha256}"
    payload = (
        f"{SELECTION_NAMESPACE}\0{DATASET_REVISION}\0{identity}"
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _create_candidate_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE candidates (
            prompt_sha256 TEXT PRIMARY KEY,
            rank_sha256 TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            content_sha256 TEXT NOT NULL,
            prompt TEXT NOT NULL,
            chosen TEXT NOT NULL,
            rejected TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection


def _index_split(parquet_path: Path, split: str, db_path: Path) -> tuple[sqlite3.Connection, dict[str, int]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    expected_rows = int(SOURCE_FILES[split]["row_count"])
    if parquet.metadata.num_rows != expected_rows:
        raise RuntimeError(
            f"{split} Parquet has {parquet.metadata.num_rows} rows; expected {expected_rows}"
        )
    schema = parquet.schema_arrow
    if tuple(schema.names) != REQUIRED_COLUMNS:
        raise RuntimeError(
            f"{split} columns are {tuple(schema.names)!r}; expected {REQUIRED_COLUMNS!r}"
        )
    for column in REQUIRED_COLUMNS:
        if not pa.types.is_string(schema.field(column).type):
            raise RuntimeError(
                f"{split}.{column} type is {schema.field(column).type}; expected string"
            )

    connection = _create_candidate_db(db_path)
    insert = """
        INSERT INTO candidates (
            prompt_sha256, rank_sha256, row_index, content_sha256,
            prompt, chosen, rejected
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(prompt_sha256) DO UPDATE SET
            rank_sha256=excluded.rank_sha256,
            row_index=excluded.row_index,
            content_sha256=excluded.content_sha256,
            prompt=excluded.prompt,
            chosen=excluded.chosen,
            rejected=excluded.rejected
        WHERE excluded.rank_sha256 < candidates.rank_sha256
           OR (excluded.rank_sha256 = candidates.rank_sha256
               AND excluded.row_index < candidates.row_index)
    """
    stats = {
        "source_rows": 0,
        "invalid_empty_or_equal_rows": 0,
        "unique_valid_prompt_groups": 0,
    }
    row_index = 0
    for batch in parquet.iter_batches(batch_size=1_024, columns=list(REQUIRED_COLUMNS)):
        columns = batch.to_pydict()
        pending: list[tuple[Any, ...]] = []
        for prompt, chosen, rejected in zip(
            columns["prompt"], columns["chosen"], columns["rejected"]
        ):
            current_row = row_index
            row_index += 1
            if (
                not isinstance(prompt, str)
                or not isinstance(chosen, str)
                or not isinstance(rejected, str)
                or not prompt.strip()
                or not chosen.strip()
                or not rejected.strip()
                or chosen == rejected
            ):
                stats["invalid_empty_or_equal_rows"] += 1
                continue
            content_sha256 = _record_content_sha256(prompt, chosen, rejected)
            prompt_sha256 = _sha256_bytes(prompt.encode("utf-8"))
            rank_sha256 = _selection_rank(split, current_row, content_sha256)
            pending.append(
                (
                    prompt_sha256,
                    rank_sha256,
                    current_row,
                    content_sha256,
                    prompt,
                    chosen,
                    rejected,
                )
            )
        connection.executemany(insert, pending)
    connection.commit()
    stats["source_rows"] = row_index
    if row_index != expected_rows:
        raise RuntimeError(f"Streamed {row_index} {split} rows; expected {expected_rows}")
    stats["unique_valid_prompt_groups"] = int(
        connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    )
    return connection, stats


def _encode_sequence(tokenizer: Any, prompt: str, completion: str) -> dict[str, Any] | None:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not prompt_ids or not completion_ids:
        return None

    # Keep the tail of long Reddit posts because it contains the TL;DR cue.  The
    # prompt cap reserves at least 128 positions for completion + EOS.
    prompt_ids = prompt_ids[-MAX_PROMPT_LENGTH:]
    completion_budget = MAX_LENGTH - len(prompt_ids)
    if completion_budget < 2:
        raise AssertionError("MAX_PROMPT_LENGTH must reserve a content token and EOS")
    completion_ids = completion_ids[: completion_budget - 1] + [EOS_TOKEN_ID]

    sequence = prompt_ids + completion_ids
    padding = MAX_LENGTH - len(sequence)
    if padding < 0:
        raise AssertionError("Encoded sequence exceeds MAX_LENGTH")
    input_ids = sequence + [PAD_TOKEN_ID] * padding
    attention_mask = [1] * len(sequence) + [0] * padding
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids + [IGNORE_INDEX] * padding
    scored_tokens = sum(label != IGNORE_INDEX for label in labels)
    if scored_tokens < 2:  # at least one text completion token plus EOS
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "non_padding_tokens": len(sequence),
        "scored_tokens": scored_tokens,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens_including_eos": len(completion_ids),
    }


def _select_records(
    connection: sqlite3.Connection,
    split: str,
    count: int,
    tokenizer: Any,
    excluded_prompt_sha256: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    stats = {
        "excluded_cross_partition_prompt_groups": 0,
        "tokenization_rejections": 0,
    }
    rows = connection.execute(
        """
        SELECT prompt_sha256, rank_sha256, row_index, content_sha256,
               prompt, chosen, rejected
        FROM candidates
        ORDER BY rank_sha256 ASC, row_index ASC
        """
    )
    for (
        prompt_sha256,
        rank_sha256,
        row_index,
        content_sha256,
        prompt,
        chosen,
        rejected,
    ) in rows:
        if prompt_sha256 in excluded_prompt_sha256:
            stats["excluded_cross_partition_prompt_groups"] += 1
            continue
        chosen_encoded = _encode_sequence(tokenizer, prompt, chosen)
        rejected_encoded = _encode_sequence(tokenizer, prompt, rejected)
        if chosen_encoded is None or rejected_encoded is None:
            stats["tokenization_rejections"] += 1
            continue
        record_id = f"{split}:{row_index}:{content_sha256}"
        selected.append(
            {
                "record_id": record_id,
                "source_split": split,
                "source_row_index": row_index,
                "content_sha256": content_sha256,
                "prompt_sha256": prompt_sha256,
                "chosen_sha256": _sha256_bytes(chosen.encode("utf-8")),
                "rejected_sha256": _sha256_bytes(rejected.encode("utf-8")),
                "selection_rank_sha256": rank_sha256,
                "chosen": chosen_encoded,
                "rejected": rejected_encoded,
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"Only {len(selected)} usable, prompt-unique {split} records were found; "
            f"{count} are required"
        )
    return selected, stats


def _write_records_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        count = len(records)
        for index, record in enumerate(records):
            metadata = {
                "chosen_sequence_row": index,
                "chosen_sha256": record["chosen_sha256"],
                "content_sha256": record["content_sha256"],
                "prompt_sha256": record["prompt_sha256"],
                "record_id": record["record_id"],
                "rejected_sequence_row": count + index,
                "rejected_sha256": record["rejected_sha256"],
                "selection_rank_sha256": record["selection_rank_sha256"],
                "source_row_index": record["source_row_index"],
                "source_split": record["source_split"],
            }
            handle.write(
                (
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )


def _write_partition(
    root: Path,
    partition_name: str,
    records: Sequence[dict[str, Any]],
    source_stats: Mapping[str, int],
    selection_stats: Mapping[str, int],
    common_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from safetensors.numpy import load_file, save_file

    partition_dir = root / partition_name
    partition_dir.mkdir()
    count = len(records)

    # Liger/TRL preference objectives split the leading dimension in half, so
    # every chosen row precedes every rejected row.
    ordered_sequences = [record["chosen"] for record in records] + [
        record["rejected"] for record in records
    ]
    tensors = {
        "input_ids": np.asarray(
            [sequence["input_ids"] for sequence in ordered_sequences], dtype=np.int32
        ),
        "labels": np.asarray(
            [sequence["labels"] for sequence in ordered_sequences], dtype=np.int32
        ),
        "attention_mask": np.asarray(
            [sequence["attention_mask"] for sequence in ordered_sequences], dtype=np.uint8
        ),
    }
    tensors_path = partition_dir / "tensors.safetensors"
    save_file(tensors, tensors_path)

    # Reload with safetensors (never pickle) and assert the full token contract.
    loaded = load_file(tensors_path)
    expected_shape = (2 * count, MAX_LENGTH)
    for name in ("input_ids", "labels", "attention_mask"):
        if loaded[name].shape != expected_shape:
            raise RuntimeError(
                f"{partition_name}/{name} shape is {loaded[name].shape}; "
                f"expected {expected_shape}"
            )
    if not np.all((loaded["attention_mask"] == 0) | (loaded["attention_mask"] == 1)):
        raise RuntimeError(f"{partition_name} attention_mask is not binary")
    if not np.all(loaded["labels"][loaded["attention_mask"] == 0] == IGNORE_INDEX):
        raise RuntimeError(f"{partition_name} padding labels are not masked")
    scored_per_sequence = np.sum(loaded["labels"] != IGNORE_INDEX, axis=1)
    if int(scored_per_sequence.min()) < 2:
        raise RuntimeError(f"{partition_name} contains a sequence without a completion token")

    records_path = partition_dir / "records.jsonl"
    _write_records_jsonl(records_path, records)
    logical_size = sum(int(tensor.nbytes) for tensor in tensors.values())
    tensor_artifact = _artifact_descriptor(
        tensors_path, logical_size_bytes=logical_size
    )
    records_artifact = _artifact_descriptor(records_path)
    record_ids = [record["record_id"] for record in records]
    prompt_hashes = [record["prompt_sha256"] for record in records]
    manifest = {
        "schema_version": 1,
        "partition": partition_name,
        "record_count": count,
        "sequence_count": 2 * count,
        "source_split": records[0]["source_split"],
        "source_stats": dict(source_stats),
        "selection_stats": dict(selection_stats),
        "record_ids": record_ids,
        "ordered_record_ids_sha256": _sha256_bytes(
            ("\n".join(record_ids) + "\n").encode("utf-8")
        ),
        "ordered_prompt_sha256s_sha256": _sha256_bytes(
            ("\n".join(prompt_hashes) + "\n").encode("utf-8")
        ),
        "non_padding_tokens": int(loaded["attention_mask"].sum()),
        "scored_completion_tokens_including_eos": int(
            np.sum(loaded["labels"] != IGNORE_INDEX)
        ),
        "artifacts": {
            "records": records_artifact,
            "tensors": tensor_artifact,
        },
        "provenance": dict(common_provenance),
    }
    manifest_path = partition_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "record_count": count,
        "sequence_count": 2 * count,
        "manifest": {
            "path": f"{partition_name}/manifest.json",
            "sha256": _sha256_file(manifest_path),
            "file_size_bytes": manifest_path.stat().st_size,
        },
        "records": {
            **records_artifact,
            "path": f"{partition_name}/{records_artifact['path']}",
        },
        "tensors": {
            **tensor_artifact,
            "path": f"{partition_name}/{tensor_artifact['path']}",
        },
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    task_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=task_root / "artifacts" / "tldr_preference_v1",
        help="new output directory; an existing path is never overwritten",
    )
    parser.add_argument("--public-count", type=int, default=DEFAULT_PUBLIC_COUNT)
    parser.add_argument("--private-count", type=int, default=DEFAULT_PRIVATE_COUNT)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="forbid network access and require all pinned inputs in the local cache",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.public_count <= 0 or args.private_count <= 0:
        raise SystemExit("--public-count and --private-count must both be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    package_versions = _check_environment_versions()
    source_paths = _resolve_sources(args.cache_dir, args.local_files_only)
    tokenizer, tokenizer_files = _resolve_tokenizer(
        args.cache_dir, args.local_files_only
    )
    script_path = Path(__file__).resolve()
    script_sha256 = _sha256_file(script_path)

    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    public_db: sqlite3.Connection | None = None
    private_db: sqlite3.Connection | None = None
    try:
        public_db, public_source_stats = _index_split(
            source_paths[PUBLIC_SOURCE_SPLIT],
            PUBLIC_SOURCE_SPLIT,
            stage / ".public-candidates.sqlite3",
        )
        public_records, public_selection_stats = _select_records(
            public_db,
            PUBLIC_SOURCE_SPLIT,
            args.public_count,
            tokenizer,
            excluded_prompt_sha256=set(),
        )
        public_prompt_hashes = {
            record["prompt_sha256"] for record in public_records
        }

        private_db, private_source_stats = _index_split(
            source_paths[PRIVATE_SOURCE_SPLIT],
            PRIVATE_SOURCE_SPLIT,
            stage / ".private-candidates.sqlite3",
        )
        private_records, private_selection_stats = _select_records(
            private_db,
            PRIVATE_SOURCE_SPLIT,
            args.private_count,
            tokenizer,
            excluded_prompt_sha256=public_prompt_hashes,
        )

        public_record_ids = {record["record_id"] for record in public_records}
        private_record_ids = {record["record_id"] for record in private_records}
        private_prompt_hashes = {
            record["prompt_sha256"] for record in private_records
        }
        if public_record_ids & private_record_ids:
            raise RuntimeError("Public/private source record IDs overlap")
        if public_prompt_hashes & private_prompt_hashes:
            raise RuntimeError("Public/private prompt groups overlap")

        common_provenance = {
            "dataset": {
                "repo_id": DATASET_REPO_ID,
                "revision": DATASET_REVISION,
                "config": DATASET_CONFIG,
                "license_declared_by_repository": None,
                "redistribution_status": "release_gate_pending",
                "source_files": SOURCE_FILES,
            },
            "tokenizer": {
                "repo_id": TOKENIZER_REPO_ID,
                "revision": TOKENIZER_REVISION,
                "files": tokenizer_files,
                "fast_tokenizer": True,
                "chat_template_applied": False,
            },
            "preprocessing": {
                "script": script_path.name,
                "script_sha256": script_sha256,
                "package_versions": package_versions,
                "selection_namespace": SELECTION_NAMESPACE,
                "selection": (
                    "lowest SHA-256 rank after retaining the lowest-ranked valid "
                    "record per exact prompt SHA-256 group"
                ),
                "max_length": MAX_LENGTH,
                "max_prompt_length": MAX_PROMPT_LENGTH,
                "prompt_truncation": "keep_end",
                "completion_truncation": "keep_start_then_append_eos",
                "padding_side": "right",
                "prompt_tokens_masked": True,
                "padding_tokens_masked": True,
                "ignore_index": IGNORE_INDEX,
                "pad_token_id": PAD_TOKEN_ID,
                "eos_token_id": EOS_TOKEN_ID,
                "row_order": "all_chosen_then_all_rejected",
            },
        }
        public_descriptor = _write_partition(
            stage,
            "public",
            public_records,
            public_source_stats,
            public_selection_stats,
            common_provenance,
        )
        private_descriptor = _write_partition(
            stage,
            "private",
            private_records,
            private_source_stats,
            private_selection_stats,
            common_provenance,
        )

        # Temporary selection databases contain source text and must never enter
        # either benchmark image or the released artifact directory.
        public_db.close()
        public_db = None
        private_db.close()
        private_db = None
        (stage / ".public-candidates.sqlite3").unlink()
        (stage / ".private-candidates.sqlite3").unlink()

        freeze_manifest = {
            "schema_version": 1,
            "status": "frozen",
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_license_declared_by_repository": None,
            "redistribution_status": "release_gate_pending",
            "tokenizer_repo_id": TOKENIZER_REPO_ID,
            "tokenizer_revision": TOKENIZER_REVISION,
            "preprocessing_script_sha256": script_sha256,
            "selection_namespace": SELECTION_NAMESPACE,
            "public_private_disjointness": {
                "record_id_intersection_count": 0,
                "prompt_sha256_intersection_count": 0,
                "source_splits_are_distinct": True,
            },
            "partitions": {
                "public": public_descriptor,
                "private": private_descriptor,
            },
        }
        manifest_path = stage / "freeze_manifest.json"
        _write_json(manifest_path, freeze_manifest)
        manifest_sha256 = _sha256_file(manifest_path)
        (stage / "freeze_manifest.sha256").write_text(
            f"{manifest_sha256}  freeze_manifest.json\n", encoding="ascii"
        )
        os.replace(stage, output_dir)
    except BaseException:
        if public_db is not None:
            public_db.close()
        if private_db is not None:
            private_db.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"Frozen artifacts: {output_dir}")
    print(f"freeze_manifest.json SHA-256: {manifest_sha256}")
    print(
        "Public/private records: "
        f"{args.public_count}/{args.private_count}; sequence length: {MAX_LENGTH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
