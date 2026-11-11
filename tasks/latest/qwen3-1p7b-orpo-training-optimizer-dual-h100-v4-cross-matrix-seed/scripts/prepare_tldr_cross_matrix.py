#!/usr/bin/env python3
"""Freeze the private TL;DR artifacts used by the V4 cross-matrix Judge.

The three sequence-length artifacts contain the same 240 source records in the
same order.  At length 512, the pair-batch rows consume nested prefixes of 60,
120, and 240 records.  TL;DR examples are short, so the longer rows are an
honest fixed-shape compute stress with right padding, not a claim that every
example contains 1,024 or 2,048 non-padding text tokens.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Sequence

import prepare_tldr_data as base


LENGTHS = (512, 1024, 2048)
RECORD_COUNT = 240
SELECTION_NAMESPACE = "edgebench:qwen3-1p7b:orpo:tldr-preference:v4-cross-matrix"


def _public_prompt_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            value = row.get("prompt_sha256")
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError("public records contain an invalid prompt hash")
            hashes.add(value)
    if not hashes:
        raise RuntimeError("public records are empty")
    return hashes


def _record(
    *,
    split: str,
    prompt_sha256: str,
    rank_sha256: str,
    row_index: int,
    content_sha256: str,
    prompt: str,
    chosen: str,
    rejected: str,
) -> dict[str, Any]:
    return {
        "record_id": f"{split}:{row_index}:{content_sha256}",
        "source_split": split,
        "source_row_index": row_index,
        "content_sha256": content_sha256,
        "prompt_sha256": prompt_sha256,
        "chosen_sha256": base._sha256_bytes(chosen.encode("utf-8")),
        "rejected_sha256": base._sha256_bytes(rejected.encode("utf-8")),
        "selection_rank_sha256": rank_sha256,
        "prompt": prompt,
        "chosen_text": chosen,
        "rejected_text": rejected,
    }


def _select_raw_records(
    connection: sqlite3.Connection,
    tokenizer: Any,
    excluded_prompt_sha256: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    excluded = 0
    rejected = 0
    rows = connection.execute(
        """
        SELECT prompt_sha256, rank_sha256, row_index, content_sha256,
               prompt, chosen, rejected
        FROM candidates
        ORDER BY rank_sha256 ASC, row_index ASC
        """
    )
    for values in rows:
        (
            prompt_sha256,
            rank_sha256,
            row_index,
            content_sha256,
            prompt,
            chosen,
            rejected_text,
        ) = values
        if prompt_sha256 in excluded_prompt_sha256:
            excluded += 1
            continue
        if not tokenizer.encode(prompt, add_special_tokens=False) or not tokenizer.encode(
            chosen, add_special_tokens=False
        ) or not tokenizer.encode(rejected_text, add_special_tokens=False):
            rejected += 1
            continue
        item = _record(
            split=base.PRIVATE_SOURCE_SPLIT,
            prompt_sha256=prompt_sha256,
            rank_sha256=rank_sha256,
            row_index=row_index,
            content_sha256=content_sha256,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected_text,
        )
        selected.append(item)
        if len(selected) == RECORD_COUNT:
            break
    if len(selected) != RECORD_COUNT:
        raise RuntimeError(f"only {len(selected)} cross-matrix records were selected")
    return selected, {
        "excluded_public_prompt_groups": excluded,
        "empty_tokenization_rejections": rejected,
        "selected_record_count": RECORD_COUNT,
    }


def _encode_records(
    tokenizer: Any, raw_records: Sequence[dict[str, Any]], length: int
) -> list[dict[str, Any]]:
    base.MAX_LENGTH = length
    base.MAX_PROMPT_LENGTH = length - 128
    encoded: list[dict[str, Any]] = []
    for raw in raw_records:
        chosen = base._encode_sequence(tokenizer, raw["prompt"], raw["chosen_text"])
        rejected = base._encode_sequence(tokenizer, raw["prompt"], raw["rejected_text"])
        if chosen is None or rejected is None:
            raise RuntimeError("a selected record failed deterministic encoding")
        encoded.append(
            {
                key: value
                for key, value in raw.items()
                if key not in {"prompt", "chosen_text", "rejected_text"}
            }
            | {"chosen": chosen, "rejected": rejected}
        )
    return encoded


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-records", type=Path, required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    public_records_path = args.public_records.expanduser().resolve(strict=True)
    package_versions = base._check_environment_versions()
    source_paths = base._resolve_sources(args.cache_dir, args.local_files_only)
    tokenizer, tokenizer_files = base._resolve_tokenizer(
        args.cache_dir, args.local_files_only
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    connection: sqlite3.Connection | None = None
    try:
        connection, source_stats = base._index_split(
            source_paths[base.PRIVATE_SOURCE_SPLIT],
            base.PRIVATE_SOURCE_SPLIT,
            stage / ".private-candidates.sqlite3",
        )
        raw_records, selection_stats = _select_raw_records(
            connection, tokenizer, _public_prompt_hashes(public_records_path)
        )
        common_record_ids = [row["record_id"] for row in raw_records]
        common_record_ids_sha256 = base._sha256_bytes(
            ("\n".join(common_record_ids) + "\n").encode("utf-8")
        )
        script_path = Path(__file__).resolve()
        partitions: dict[str, Any] = {}
        for length in LENGTHS:
            common_provenance = {
                "dataset": {
                    "repo_id": base.DATASET_REPO_ID,
                    "revision": base.DATASET_REVISION,
                    "config": base.DATASET_CONFIG,
                    "license_declared_by_repository": None,
                    "redistribution_status": "release_gate_pending",
                    "source_files": base.SOURCE_FILES,
                },
                "tokenizer": {
                    "repo_id": base.TOKENIZER_REPO_ID,
                    "revision": base.TOKENIZER_REVISION,
                    "files": tokenizer_files,
                    "fast_tokenizer": True,
                    "chat_template_applied": False,
                },
                "preprocessing": {
                    "script": script_path.name,
                    "script_sha256": base._sha256_file(script_path),
                    "base_script_sha256": base._sha256_file(Path(base.__file__).resolve()),
                    "package_versions": package_versions,
                    "selection_namespace": SELECTION_NAMESPACE,
                    "selection": "deterministic ranked prefix shared by all sequence lengths",
                    "common_ordered_record_ids_sha256": common_record_ids_sha256,
                    "max_length": length,
                    "max_prompt_length": length - 128,
                    "prompt_truncation": "keep_end",
                    "completion_truncation": "keep_start_then_append_eos",
                    "padding_side": "right",
                    "prompt_tokens_masked": True,
                    "padding_tokens_masked": True,
                    "ignore_index": base.IGNORE_INDEX,
                    "pad_token_id": base.PAD_TOKEN_ID,
                    "eos_token_id": base.EOS_TOKEN_ID,
                    "row_order": "all_chosen_then_all_rejected",
                },
            }
            encoded = _encode_records(tokenizer, raw_records, length)
            descriptor = base._write_partition(
                stage,
                "private",
                encoded,
                source_stats,
                selection_stats,
                common_provenance,
            )
            target_name = f"private_cross_{length}"
            (stage / "private").rename(stage / target_name)
            descriptor["manifest"]["path"] = f"{target_name}/manifest.json"
            descriptor["records"]["path"] = f"{target_name}/records.jsonl"
            descriptor["tensors"]["path"] = f"{target_name}/tensors.safetensors"
            partitions[target_name] = descriptor

        connection.close()
        connection = None
        (stage / ".private-candidates.sqlite3").unlink()
        root_manifest = {
            "schema_version": 1,
            "status": "frozen",
            "dataset_repo_id": base.DATASET_REPO_ID,
            "dataset_revision": base.DATASET_REVISION,
            "dataset_license_declared_by_repository": None,
            "redistribution_status": "release_gate_pending",
            "tokenizer_repo_id": base.TOKENIZER_REPO_ID,
            "tokenizer_revision": base.TOKENIZER_REVISION,
            "preprocessing_script_sha256": base._sha256_file(script_path),
            "selection_namespace": SELECTION_NAMESPACE,
            "record_count": RECORD_COUNT,
            "long_context_semantics": "fixed-shape right-padded compute stress",
            "common_ordered_record_ids_sha256": common_record_ids_sha256,
            "public_prompt_intersection_count": 0,
            "partitions": partitions,
        }
        base._write_json(stage / "freeze_manifest.json", root_manifest)
        root_sha = base._sha256_file(stage / "freeze_manifest.json")
        (stage / "freeze_manifest.sha256").write_text(
            f"{root_sha}  freeze_manifest.json\n", encoding="ascii"
        )
        os.replace(stage, output_dir)
    except BaseException:
        if connection is not None:
            connection.close()
        import shutil

        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(base._sha256_file(output_dir / "freeze_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
