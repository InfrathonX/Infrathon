"""Strict, retained-FD reader for real RULER-derived BF16 Q/K/V fixtures.

Only metadata and descriptors stay resident. Each file is authenticated again
when read, including after the worker has permanently dropped root privileges.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
from dataclasses import dataclass

PROTOCOL = "minference-real-ruler-prefill-v3"
TASKS = ("niah_single_2", "niah_multikey_1", "vt", "fwe")
BUDGETS = (32768, 65536, 131072)
LAYERS = (1, 4, 15, 27)
WARMUP_LAYER = 2
MAX_FILE_BYTES = 4 * 1024**3
MAX_MANIFEST_BYTES = 4 * 1024**2
SHA = re.compile(r"[0-9a-f]{64}\Z")


class ArchiveError(RuntimeError):
    pass


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArchiveError(f"duplicate JSON key {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ArchiveError(x)))


def read_json(path, limit=MAX_MANIFEST_BYTES):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
        raise ArchiveError("metadata must be a bounded regular file")
    raw = path.read_bytes()
    return strict_json(raw), hashlib.sha256(raw).hexdigest()


def validate_manifest(data, *, full=True):
    if data.get("schema_version") != 3 or data.get("protocol") != PROTOCOL:
        raise ArchiveError("unsupported fixture protocol")
    if data.get("partition") not in ("public", "final", "calibration"):
        raise ArchiveError("unsupported fixture partition")
    fixtures, groups = data.get("fixtures"), data.get("case_groups")
    if not isinstance(fixtures, dict) or not isinstance(groups, list) or not groups:
        raise ArchiveError("missing fixture/group map")
    seen_groups, grid, timed_ids = set(), set(), set()
    paths, digests, prompts = set(), set(), {}
    for fixture_id, f in fixtures.items():
        if not isinstance(fixture_id, str) or not isinstance(f, dict):
            raise ArchiveError("malformed fixture")
        file = f.get("file")
        if (not isinstance(file, str) or not file or Path(file).is_absolute()
                or ".." in Path(file).parts):
            raise ArchiveError("unsafe fixture path")
        for key in ("sha256", "input_token_sha256"):
            if not isinstance(f.get(key), str) or not SHA.fullmatch(f[key]):
                raise ArchiveError(f"missing fixture {key}")
        n, budget = f.get("actual_n"), f.get("context_budget")
        if type(n) is not int or type(budget) is not int or not 2048 <= n < budget or budget not in BUDGETS:
            raise ArchiveError("invalid actual token length/budget")
        if f.get("task") not in TASKS or type(f.get("layer")) is not int or f["layer"] not in (*LAYERS, WARMUP_LAYER):
            raise ArchiveError("invalid fixture task/layer")
        if type(f.get("size")) is not int or not 0 < f["size"] <= MAX_FILE_BYTES:
            raise ArchiveError("invalid fixture size")
        if f.get("trajectory") != "dense-sdpa" or not isinstance(f.get("prompt_id"), str):
            raise ArchiveError("fixture must bind a dense trajectory and prompt")
        if file in paths or f["sha256"] in digests:
            raise ArchiveError("fixture paths and tensor digests must be unique")
        paths.add(file)
        digests.add(f["sha256"])
        prompt_identity = (f["task"], budget, n, f["input_token_sha256"])
        if f["prompt_id"] in prompts and prompts[f["prompt_id"]] != prompt_identity:
            raise ArchiveError("prompt identity differs across captured layers")
        prompts[f["prompt_id"]] = prompt_identity
    for g in groups:
        if not isinstance(g, dict) or not isinstance(g.get("case_id"), str) or g["case_id"] in seen_groups:
            raise ArchiveError("duplicate/malformed group")
        seen_groups.add(g["case_id"])
        identity = (g.get("task"), g.get("context_budget"), g.get("layer"))
        if identity in grid or identity[0] not in TASKS or identity[1] not in BUDGETS or type(identity[2]) is not int or identity[2] not in LAYERS:
            raise ArchiveError("invalid/repeated group stratum")
        grid.add(identity)
        warm, trials = g.get("warmups"), g.get("trials")
        if not isinstance(warm, list) or not isinstance(trials, list) or len(warm) != 3 or len(trials) != 3:
            raise ArchiveError("each group needs three paired warm-ups and trials")
        token_ids = set()
        for wid, tid in zip(warm, trials):
            if wid not in fixtures or tid not in fixtures or tid in timed_ids:
                raise ArchiveError("missing/reused timed fixture")
            timed_ids.add(tid)
            w, t = fixtures[wid], fixtures[tid]
            if (t["task"], t["context_budget"], t["layer"]) != identity:
                raise ArchiveError("timed fixture/group identity differs")
            if w["task"] != t["task"] or w["context_budget"] != t["context_budget"] or w["layer"] != WARMUP_LAYER:
                raise ArchiveError("warm-up must use the unscored layer")
            if any(w[key] != t[key] for key in ("actual_n", "prompt_id", "input_token_sha256")):
                raise ArchiveError("warm-up must have the timed shape and prompt")
            if w["sha256"] == t["sha256"]:
                raise ArchiveError("warm-up and trial must have different real activations")
            token_ids.add(t["input_token_sha256"])
        if len(token_ids) != 3:
            raise ArchiveError("timed prompts must be distinct within each group")
    expected = {(t, b, l) for t in TASKS for b in BUDGETS for l in LAYERS}
    if full and (grid != expected or len(fixtures) != 180 or len(prompts) != 36
                 or len({p[3] for p in prompts.values()}) != 36):
        raise ArchiveError("formal suite must contain 48 groups, 36 distinct prompts and 180 unique fixtures")
    if not isinstance(data.get("bindings"), dict) or not data["bindings"]:
        raise ArchiveError("missing source/model/tokenizer/software bindings")
    return data


@dataclass
class ActivationHandle:
    fixture_id: str
    entry: dict
    fd: int
    header: dict
    payload_offset: int

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def digest_fd(fd, size):
    h, offset = hashlib.sha256(), 0
    while offset < size:
        chunk = os.pread(fd, min(8 * 1024**2, size - offset), offset)
        if not chunk:
            raise ArchiveError("truncated fixture")
        h.update(chunk)
        offset += len(chunk)
    return h.hexdigest()


def open_handle(root, fixture_id, entry):
    root = Path(root).resolve()
    path = root / entry["file"]
    # Refuse symlinks in every component, not just the final file.
    current = root
    for component in Path(entry["file"]).parts:
        current = current / component
        if current.is_symlink():
            raise ArchiveError("fixture symlinks are forbidden")
    if not path.resolve().is_relative_to(root):
        raise ArchiveError("fixture escaped archive root")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != entry["size"]:
            raise ArchiveError("fixture size/type differs")
        if digest_fd(fd, info.st_size) != entry["sha256"]:
            raise ArchiveError("fixture digest differs")
        first = os.pread(fd, 8, 0)
        if len(first) != 8:
            raise ArchiveError("missing safetensors header")
        length = struct.unpack("<Q", first)[0]
        if not 2 <= length <= 65536 or 8 + length >= info.st_size:
            raise ArchiveError("invalid safetensors header length")
        header = strict_json(os.pread(fd, length, 8))
        if set(header) - {"q", "k", "v", "__metadata__"} or not all(k in header for k in ("q", "k", "v")):
            raise ArchiveError("unexpected safetensors tensor names")
        metadata = header.get("__metadata__")
        expected_metadata = {
            "case_id": fixture_id, "prompt_id": entry["prompt_id"],
            "layer": str(entry["layer"]), "actual_n": str(entry["actual_n"]),
            "input_token_sha256": entry["input_token_sha256"], "task": entry["task"],
            "context_budget": str(entry["context_budget"]),
            "trajectory": "dense-sdpa", "protocol": PROTOCOL,
        }
        if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ArchiveError("safetensors input/layer/source identity differs")
        n, spans = entry["actual_n"], []
        for name, heads in (("q", 32), ("k", 8), ("v", 8)):
            t = header[name]
            offsets = t.get("data_offsets")
            if t.get("dtype") != "BF16" or t.get("shape") != [1, heads, n, 128]:
                raise ArchiveError("fixture dtype/shape differs")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ArchiveError("invalid tensor span")
            start, end = offsets
            if not 0 <= start < end <= info.st_size - 8 - length or end - start != heads * n * 128 * 2:
                raise ArchiveError("invalid tensor span size")
            spans.append((start, end))
        ordered = sorted(spans)
        if ordered[0][0] != 0 or any(a[1] != b[0] for a, b in zip(ordered, ordered[1:])) or ordered[-1][1] != info.st_size - 8 - length:
            raise ArchiveError("tensor spans overlap or leave gaps")
        return ActivationHandle(fixture_id, entry, fd, header, 8 + length)
    except BaseException:
        os.close(fd)
        raise


def read_tensors(torch, handle, device):
    if digest_fd(handle.fd, handle.entry["size"]) != handle.entry["sha256"]:
        raise ArchiveError("fixture changed after admission")
    result = {}
    for name in ("q", "k", "v"):
        tensor = handle.header[name]
        start, end = tensor["data_offsets"]
        data = bytearray(os.pread(handle.fd, end - start, handle.payload_offset + start))
        if len(data) != end - start:
            raise ArchiveError("short tensor read")
        cpu = torch.frombuffer(data, dtype=torch.bfloat16).reshape(tensor["shape"])
        result[name] = cpu.to(device=device).contiguous()
        if not bool(torch.isfinite(result[name]).all()):
            raise ArchiveError("fixture has nonfinite activations")
    return result
