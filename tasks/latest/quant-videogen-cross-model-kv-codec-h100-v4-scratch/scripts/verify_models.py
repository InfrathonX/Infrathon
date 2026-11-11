#!/usr/bin/env python3
"""Verify pinned, materialized Hugging Face model trees and write a lock file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED = {
    "self_forcing_base": (
        "staging/self-forcing/Wan2.1-T2V-1.3B",
        "37ec512624d61f7aa208f7ea8140a131f93afc9a",
        ["diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"],
    ),
    "self_forcing_dmd": (
        "staging/self-forcing/Self-Forcing",
        "2f8b779212da279d212c22a509b66ad6552f350e",
        ["checkpoints/self_forcing_dmd.pt"],
    ),
    "longcat": (
        "staging/longcat/LongCat-Video",
        "03b55529b1d1d4045f5fbe14d65c8c6e8116b278",
        ["dit/diffusion_pytorch_model.safetensors.index.json", "vae/diffusion_pytorch_model.safetensors"],
    ),
    "hy_worldplay_8b_action": (
        "staging/hy-worldplay-8b/HY-WorldPlay",
        "f4c29235647707b571479a69b569e4166f9f5bf8",
        ["ar_distilled_action_model/model.safetensors"],
    ),
    "hunyuanvideo_1_5": (
        "staging/hy-worldplay-8b/HunyuanVideo-1.5",
        "9b49404b3f5df2a8f0b31df27a0c7ab872e7b038",
        ["transformer/480p_i2v/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"],
    ),
    "qwen_vl_text_encoder": (
        "staging/hy-worldplay-8b/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        ["model.safetensors.index.json", "tokenizer.json"],
    ),
    "byt5_small": (
        "staging/hy-worldplay-8b/byt5-small",
        "68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
        ["pytorch_model.bin", "config.json"],
    ),
    "siglip_redux": (
        "staging/hy-worldplay-8b/FLUX.1-Redux-dev",
        "c95859fbf7703ca4d6824b4da4407d7cd0434f81",
        ["image_encoder/model.safetensors", "feature_extractor/preprocessor_config.json"],
    ),
}

GIT_EXPECTED = {
    "glyph_byt5": (
        "staging/hy-worldplay-8b/Glyph-SDXL-v2",
        "b51a8604c03c9a4e71a389f4c7deeb526320f2e6",
        [
            "assets/color_idx.json",
            "assets/multilingual_10-lang_idx.json",
            "checkpoints/byt5_model.pt",
        ],
    ),
    "hy_worldplay_official_assets": (
        "staging/hy-worldplay-official",
        "1588e1336e842b03b0a7860c654ebd7c46bb065e",
        ["assets/img/1.png", "assets/pose/test_forward_32_latents.json"],
    ),
    "vbench_source": (
        "staging/vbench",
        "45e79ec14e69a2187202c675d2dbce1a71843d53",
        ["vbench/background_consistency.py", "vbench/imaging_quality.py"],
    ),
    "vbench_dino_source": (
        "staging/vbench-cache/dino_model/facebookresearch_dino_main",
        "7c446df5b9f45747937fb0d72314eb9f7b66930a",
        ["hubconf.py", "vision_transformer.py"],
    ),
}

FILE_EXPECTED = {
    "vbench_clip_vit_b32": (
        "staging/vbench-cache/clip_model/ViT-B-32.pt",
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    ),
    "vbench_clip_vit_l14": (
        "staging/vbench-cache/clip_model/ViT-L-14.pt",
        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
    ),
    "vbench_dino_vitb16": (
        "staging/vbench-cache/dino_model/dino_vitbase16_pretrain.pth",
        "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    ),
    "vbench_dino_torch_hub_checkpoint": (
        "staging/vbench-cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth",
        "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    ),
    "vbench_aesthetic_linear": (
        "staging/vbench-cache/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
        "2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba",
    ),
    "vbench_musiq_spaq": (
        "staging/vbench-cache/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
        "358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50",
    ),
    "lpips_alexnet_imagenet": (
        "staging/vbench-cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    ),
    "qvg_lpips_vgg16_imagenet": (
        "staging/vbench-cache/torch/hub/checkpoints/vgg16-397923af.pth",
        "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
    ),
}

MOUNTED_TREE_EXPECTED = {
    "hidden": "hidden",
    "vbench_cache": "staging/vbench-cache",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _stable_stat_tuple(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_chain(
    path: Path,
) -> list[tuple[int, int | None, str | None, os.stat_result]]:
    """Open every absolute path component without following a symlink."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(absolute.anchor, _directory_flags())
    chain: list[tuple[int, int | None, str | None, os.stat_result]] = [
        (descriptor, None, None, os.fstat(descriptor))
    ]
    try:
        for component in absolute.parts[1:]:
            parent_fd = chain[-1][0]
            try:
                observed = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
            except OSError as exc:
                raise RuntimeError(f"unsafe directory component in {path}") from exc
            opened = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(observed.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or (observed.st_dev, observed.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                os.close(child_fd)
                raise RuntimeError(f"directory changed while opening {path}")
            chain.append((child_fd, parent_fd, component, opened))
        return chain
    except BaseException:
        for opened_fd, _parent_fd, _name, _initial in reversed(chain):
            os.close(opened_fd)
        raise


def _validate_and_close_directory_chain(
    chain: list[tuple[int, int | None, str | None, os.stat_result]],
) -> None:
    failure: BaseException | None = None
    try:
        for descriptor, parent_fd, name, initial in reversed(chain[1:]):
            assert parent_fd is not None and name is not None
            try:
                opened_after = os.fstat(descriptor)
                linked_after = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                failure = RuntimeError("directory path changed during attestation")
                failure.__cause__ = exc
                break
            if (
                not stat.S_ISDIR(opened_after.st_mode)
                or not stat.S_ISDIR(linked_after.st_mode)
                or (initial.st_dev, initial.st_ino)
                != (opened_after.st_dev, opened_after.st_ino)
                or (opened_after.st_dev, opened_after.st_ino)
                != (linked_after.st_dev, linked_after.st_ino)
            ):
                failure = RuntimeError("directory path changed during attestation")
                break
    finally:
        for descriptor, _parent_fd, _name, _initial in reversed(chain):
            os.close(descriptor)
    if failure is not None:
        raise failure


def _stable_file_identity(
    directory_fd: int,
    name: str,
    *,
    display_path: str,
    expected: os.stat_result,
) -> tuple[dict[str, Any], bytes]:
    """Hash a regular file through its parent dirfd and reject mutations."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeError(f"file is unavailable or unsafe: {display_path}") from exc
    digest = hashlib.sha256()
    prefix = bytearray()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (expected.st_dev, expected.st_ino)
        ):
            raise RuntimeError(f"file changed before hashing: {display_path}")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if len(prefix) < 4096:
                prefix.extend(chunk[: 4096 - len(prefix)])
        after = os.fstat(descriptor)
        try:
            linked_after = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeError(f"file disappeared while hashing: {display_path}") from exc
        if (
            _stable_stat_tuple(before) != _stable_stat_tuple(after)
            or (after.st_dev, after.st_ino)
            != (linked_after.st_dev, linked_after.st_ino)
            or not stat.S_ISREG(linked_after.st_mode)
        ):
            raise RuntimeError(f"file changed while hashing: {display_path}")
        return (
            {
                "path": display_path,
                "type": "file",
                "mode": stat.S_IMODE(after.st_mode),
                "size": after.st_size,
                "sha256": digest.hexdigest(),
            },
            bytes(prefix),
        )
    finally:
        os.close(descriptor)


def _stable_tree_manifest(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, bytes]]:
    """Attest exact topology, modes, and content via dirfd traversal."""

    chain = _open_directory_chain(root)
    root_fd = chain[-1][0]
    rows: list[dict[str, Any]] = []
    regular_by_path: dict[str, dict[str, Any]] = {}
    prefixes: dict[str, bytes] = {}

    def visit(directory_fd: int, prefix: str) -> None:
        before_directory = os.fstat(directory_fd)
        if not stat.S_ISDIR(before_directory.st_mode):
            raise RuntimeError(f"bundle directory became unsafe: {prefix or '.'}")
        try:
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise RuntimeError(f"cannot traverse bundle directory: {prefix or '.'}") from exc
        observed_entries: list[tuple[str, int, int, int]] = []
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                observed = os.stat(
                    entry.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise RuntimeError(f"bundle entry disappeared: {relative}") from exc
            observed_entries.append(
                (entry.name, observed.st_dev, observed.st_ino, observed.st_mode)
            )
            mode = stat.S_IMODE(observed.st_mode)
            if stat.S_ISDIR(observed.st_mode):
                try:
                    child_fd = os.open(
                        entry.name,
                        _directory_flags(),
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise RuntimeError(
                        f"bundle directory is unavailable or unsafe: {relative}"
                    ) from exc
                opened = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (observed.st_dev, observed.st_ino)
                ):
                    os.close(child_fd)
                    raise RuntimeError(
                        f"bundle directory changed during traversal: {relative}"
                    )
                rows.append({"path": relative, "type": "directory", "mode": mode})
                try:
                    visit(child_fd, relative)
                finally:
                    try:
                        opened_after = os.fstat(child_fd)
                        linked_after = os.stat(
                            entry.name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise RuntimeError(
                            f"bundle directory disappeared: {relative}"
                        ) from exc
                    finally:
                        os.close(child_fd)
                    if (
                        _stable_stat_tuple(opened) != _stable_stat_tuple(opened_after)
                        or (opened_after.st_dev, opened_after.st_ino)
                        != (linked_after.st_dev, linked_after.st_ino)
                        or not stat.S_ISDIR(linked_after.st_mode)
                    ):
                        raise RuntimeError(
                            f"bundle directory changed during traversal: {relative}"
                        )
            elif stat.S_ISREG(observed.st_mode):
                row, file_prefix = _stable_file_identity(
                    directory_fd,
                    entry.name,
                    display_path=relative,
                    expected=observed,
                )
                rows.append(row)
                regular_by_path[relative] = row
                prefixes[relative] = file_prefix
            elif stat.S_ISLNK(observed.st_mode):
                raise RuntimeError(f"bundle contains a symlink: {relative}")
            else:
                raise RuntimeError(f"bundle contains an unsupported entry: {relative}")
        try:
            with os.scandir(directory_fd) as iterator:
                linked_entries = []
                for entry in sorted(iterator, key=lambda item: item.name):
                    linked = os.stat(
                        entry.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    linked_entries.append(
                        (entry.name, linked.st_dev, linked.st_ino, linked.st_mode)
                    )
        except OSError as exc:
            raise RuntimeError(
                f"cannot revalidate bundle directory: {prefix or '.'}"
            ) from exc
        if linked_entries != observed_entries:
            raise RuntimeError(
                f"bundle directory changed during traversal: {prefix or '.'}"
            )
        after_directory = os.fstat(directory_fd)
        if _stable_stat_tuple(before_directory) != _stable_stat_tuple(after_directory):
            raise RuntimeError(
                f"bundle directory changed during traversal: {prefix or '.'}"
            )

    try:
        opened_root = os.fstat(root_fd)
        rows.append(
            {
                "path": ".",
                "type": "directory",
                "mode": stat.S_IMODE(opened_root.st_mode),
            }
        )
        visit(root_fd, "")
        opened_root_after = os.fstat(root_fd)
        if _stable_stat_tuple(opened_root) != _stable_stat_tuple(opened_root_after):
            raise RuntimeError("bundle root directory changed during traversal")
    finally:
        _validate_and_close_directory_chain(chain)
    rows.sort(key=lambda row: str(row["path"]))
    return rows, regular_by_path, prefixes


def _required_key(relative: str) -> str:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise RuntimeError(f"unsafe required bundle path: {relative}")
    return parsed.as_posix()


def verify_tree_files(root: Path, required: list[str]) -> dict[str, Any]:
    entries, regular_by_path, _prefixes = _stable_tree_manifest(root)
    for relative in required:
        if _required_key(relative) not in regular_by_path:
            raise RuntimeError(f"missing or unsafe required file: {root / relative}")
    files = [dict(row) for row in entries if row.get("type") == "file"]
    return {
        "root": str(root),
        "bytes": sum(int(row["size"]) for row in files),
        "tree_sha256": canonical_sha256(entries),
        "files": files,
        "entries": entries,
    }


def verify_tree(root: Path, revision: str, required: list[str]) -> dict[str, Any]:
    entries, regular_by_path, prefixes = _stable_tree_manifest(root)
    incomplete = [path for path in regular_by_path if path.endswith(".incomplete")]
    if incomplete:
        raise RuntimeError(f"incomplete downloads remain under {root}: {len(incomplete)}")
    tree_evidence = f".cache/huggingface/trees/{revision}.json"
    metadata_evidence = any(
        path.startswith(".cache/huggingface/download/")
        and path.endswith(".metadata")
        and prefix.splitlines()[:1] == [revision.encode("utf-8")]
        for path, prefix in prefixes.items()
    )
    if tree_evidence not in regular_by_path and not metadata_evidence:
        raise RuntimeError(f"no Hugging Face evidence for revision {revision} under {root}")
    for relative in required:
        if _required_key(relative) not in regular_by_path:
            raise RuntimeError(f"missing or unsafe required file: {root / relative}")
    files = [dict(row) for row in entries if row.get("type") == "file"]
    return {
        "root": str(root),
        "revision": revision,
        "bytes": sum(int(row["size"]) for row in files),
        "tree_sha256": canonical_sha256(entries),
        "files": files,
        "entries": entries,
    }


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot resolve git revision for {root}")
    return completed.stdout.strip()


def verify_git_tree(root: Path, revision: str, required: list[str]) -> dict[str, Any]:
    before_revision = _git_revision(root)
    if before_revision != revision:
        raise RuntimeError(
            f"git revision mismatch for {root}: {before_revision} != {revision}"
        )
    result = verify_tree_files(root, required)
    after_revision = _git_revision(root)
    if after_revision != revision:
        raise RuntimeError(f"git revision changed while attesting {root}")
    result["revision"] = revision
    return result


def verify_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    chain = _open_directory_chain(path.parent)
    parent_fd = chain[-1][0]
    try:
        try:
            observed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"missing or unsafe file: {path}") from exc
        row, _prefix = _stable_file_identity(
            parent_fd,
            path.name,
            display_path=str(path),
            expected=observed,
        )
    finally:
        _validate_and_close_directory_chain(chain)
    actual_sha256 = str(row["sha256"])
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: {actual_sha256} != {expected_sha256}"
        )
    return {
        "path": str(path),
        "bytes": int(row["size"]),
        "mode": int(row["mode"]),
        "sha256": actual_sha256,
    }


def verify_mounted_trees(
    artifact_root: Path,
    mounted_trees: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Attest complete score-affecting bind trees not covered as bundles."""

    return {
        name: verify_tree_files(artifact_root / relative, [])
        for name, relative in sorted(mounted_trees.items())
    }


def atomic_write_text(path: Path, value: str) -> None:
    """Publish a private lock file without following or truncating a symlink."""

    try:
        parent = path.parent.resolve(strict=True)
        parent_info = os.stat(parent, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("model lock output parent is unavailable") from exc
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) & 0o022
    ):
        raise RuntimeError("model lock output parent is unsafe")
    destination = parent / path.name
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("refusing to overwrite model lock output")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = value.encode("utf-8")
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if destination.exists() or destination.is_symlink():
            raise RuntimeError("refusing to overwrite model lock output")
        os.replace(temporary, destination)
        directory_fd = os.open(parent, _directory_flags())
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(os.environ["QVG_ARTIFACT_ROOT"])
        if os.environ.get("QVG_ARTIFACT_ROOT")
        else None,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.artifact_root is None:
        parser.error("set QVG_ARTIFACT_ROOT or pass --artifact-root")
    bundles = {
        name: verify_tree(args.artifact_root / relative, revision, required)
        for name, (relative, revision, required) in EXPECTED.items()
    }
    git_bundles = {
        name: verify_git_tree(args.artifact_root / relative, revision, required)
        for name, (relative, revision, required) in GIT_EXPECTED.items()
    }
    files = {}
    for name, (relative, expected_sha256) in FILE_EXPECTED.items():
        path = args.artifact_root / relative
        files[name] = verify_file(path, expected_sha256)
    mounted_trees = verify_mounted_trees(
        args.artifact_root,
        MOUNTED_TREE_EXPECTED,
    )
    payload = {
        "schema_version": 1,
        "status": "verified",
        "bundles": bundles,
        "git_bundles": git_bundles,
        "files": files,
        "mounted_trees": mounted_trees,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.output, encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
