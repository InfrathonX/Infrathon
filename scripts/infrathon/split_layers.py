#!/usr/bin/env python3
"""Re-layer a Docker image: split every layer above a size into parts.

    scripts/infrathon/split_layers.py SOURCE_TAG TARGET_TAG --workdir DIR [--max-layer-gib 4]
        [--keep-layers N --config-from-inspect FILE] [--listing FILE]

Some base images (the Miles bases) were flattened into one 50-65 GB layer, which
is fragile to push to or pull from a registry: a single failure restarts the
whole layer.  This rewrites the image with the same file system and the same
config, its large layers cut into consecutive parts of at most --max-layer-gib
(a single larger file gets a part of its own):

1. ``docker save SOURCE_TAG`` into DIR (the layers stay as saved, gzip or not).
2. Pass 1 reads each large layer's tar headers and assigns members to parts.  A
   hard link whose target falls in an earlier part becomes a regular file with
   the target's bytes (pass 2 spools those targets), since a layer cannot link
   into a lower one.
3. Pass 2 writes each part as an uncompressed tar, member headers unchanged
   (owner, mode, mtime, xattrs).  A part that adds entries under directories
   created in an earlier part ends with those directories' entries again, so they
   keep their original metadata after the part is applied.
4. A docker-archive (manifest.json, config with the new diff_ids and one history
   entry per part) is streamed into ``docker load`` and tagged TARGET_TAG.

Compare the result with the source (file listing inside both containers) before
using it; DIR can be removed afterwards.

``--keep-layers N`` keeps only the source's first N layers, with the config
(Env, Cmd, Entrypoint, WorkingDir, User, Labels...) taken from a ``docker image
inspect`` JSON of the image those layers came from: a base image can be rebuilt
from an image built on it, when the base itself is not on the host.
``--listing FILE`` writes the file listing of a single flattened source layer
(path, type, mode, owner, sha256 or link target, xattrs), in the format of a
``docker export`` listing of the result, to compare against.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

GIB = 1024 ** 3


def _layer_stream(save: tarfile.TarFile, name: str):
    """The uncompressed tar stream of one saved layer blob."""
    raw = save.extractfile(name)
    if raw is None:
        raise SystemExit(f"layer {name} is not a file in the saved archive")
    head = raw.read(2)
    raw.seek(0)
    return gzip.GzipFile(fileobj=raw) if head == b"\x1f\x8b" else raw


def _normal(name: str) -> str:
    return str(PurePosixPath(name.lstrip("./") or "."))


def plan_parts(save: tarfile.TarFile, layer: str, limit: int) -> tuple[list[int], dict[str, int], int]:
    """Pass 1: the part of every member (by index), cross-part link targets, count."""
    parts: list[int] = []
    part_of: dict[str, int] = {}
    materialize: dict[str, int] = {}
    part, used = 0, 0
    with tarfile.open(fileobj=_layer_stream(save, layer), mode="r|") as stream:
        for member in stream:
            if PurePosixPath(member.name).name.startswith(".wh."):
                raise SystemExit(f"{layer} has whiteouts ({member.name}); only flattened layers are split")
            size = member.size if member.isreg() else 0
            if used and used + size > limit:
                part, used = part + 1, 0
            used += size + 512
            parts.append(part)
            name = _normal(member.name)
            part_of[name] = part
            if member.islnk():
                target = _normal(member.linkname)
                if part_of.get(target, part) != part:
                    materialize[target] = part_of[target]
    return parts, materialize, part + 1


class _Hashed(io.RawIOBase):
    """Reads through, hashing what passes."""

    def __init__(self, source):
        self.source, self.hasher = source, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.source.read(len(buffer))
        self.hasher.update(data)
        buffer[:len(data)] = data
        return len(data)


def _listing_line(member: tarfile.TarInfo, name: str, extra: str, kind: str) -> str:
    xattrs = ",".join(sorted(k[len("SCHILY.xattr."):] for k in member.pax_headers if k.startswith("SCHILY.xattr.")))
    return f"{name}\t{kind}\t{member.mode:04o}\t{member.uid}\t{member.gid}\t{extra}\t{xattrs}"


def write_parts(save: tarfile.TarFile, layer: str, parts: list[int], materialize: dict[str, int],
                count: int, out: Path, spool: Path, listing: list[str] | None = None) -> list[tuple[Path, str]]:
    """Pass 2: write the parts; returns (path, diff_id) per part.  With ``listing``,
    appends one line per member (see --listing)."""
    contents: dict[str, str] = {}
    spool.mkdir(parents=True, exist_ok=True)
    spooled: dict[str, Path] = {}
    directories: dict[str, tuple[int, tarfile.TarInfo]] = {}
    written: list[tuple[Path, str]] = []
    current, writer, digest, handle = -1, None, None, None
    touched: set[str] = set()

    def close_part() -> None:
        nonlocal writer, handle
        if writer is None:
            return
        # Directories created in an earlier part that this part adds under:
        # restate them last so the applier leaves their original metadata.
        for name in sorted(touched, key=lambda n: n.count("/"), reverse=True):
            writer.addfile(directories[name][1])
        writer.close()
        handle.close()
        written.append((out / f"part-{current:03d}.tar", digest.hexdigest()))

    class Hashing(io.RawIOBase):
        def __init__(self, sink, hasher):
            self.sink, self.hasher = sink, hasher

        def writable(self):
            return True

        def write(self, data):
            self.hasher.update(data)
            return self.sink.write(data)

    with tarfile.open(fileobj=_layer_stream(save, layer), mode="r|") as stream:
        for index, member in enumerate(stream):
            part = parts[index]
            if part != current:
                close_part()
                current, touched = part, set()
                digest = hashlib.sha256()
                handle = (out / f"part-{part:03d}.tar").open("wb")
                writer = tarfile.open(fileobj=Hashing(handle, digest), mode="w|", format=tarfile.PAX_FORMAT)
            name = _normal(member.name)
            for ancestor in PurePosixPath(name).parents:
                ancestor = str(ancestor)
                if ancestor in directories and directories[ancestor][0] != part:
                    touched.add(ancestor)
            if member.isdir():
                directories[name] = (part, member)
            if listing is not None:
                if member.islnk():
                    listing.append(_listing_line(member, name, contents[_normal(member.linkname)], "f"))
                elif member.issym():
                    listing.append(_listing_line(member, name, member.linkname, "l"))
                elif member.isdir():
                    listing.append(_listing_line(member, name, "", "d"))
                elif not member.isreg():
                    listing.append(_listing_line(member, name, f"{member.type!r}:{member.devmajor}:{member.devminor}", "o"))
            if member.islnk() and _normal(member.linkname) in spooled:
                regular = member.__class__(member.name)
                for field in ("mode", "uid", "gid", "uname", "gname", "mtime", "pax_headers"):
                    setattr(regular, field, getattr(member, field))
                source = spooled[_normal(member.linkname)]
                regular.size = source.stat().st_size
                with source.open("rb") as data:
                    writer.addfile(regular, data)
                continue
            data = stream.extractfile(member) if member.isreg() else None
            if data is not None and name in materialize:
                path = spool / hashlib.sha256(name.encode()).hexdigest()
                with path.open("wb") as copy:
                    shutil.copyfileobj(data, copy)
                spooled[name] = path
                data = path.open("rb")
            hashed = _Hashed(data) if data is not None and listing is not None else None
            writer.addfile(member, io.BufferedReader(hashed) if hashed is not None else data)
            if hashed is not None:
                contents[name] = hashed.hasher.hexdigest()
                listing.append(_listing_line(member, name, contents[name], "f"))
    close_part()
    if len(written) != count:
        raise SystemExit(f"{layer}: wrote {len(written)} parts, planned {count}")
    return written


def _config_for_prefix(config: dict, keep: int, inspect: list | dict) -> dict:
    """The config of an image made of the first ``keep`` layers, from that image's inspect JSON."""
    row = inspect[0] if isinstance(inspect, list) else inspect
    if row["RootFS"]["Layers"] != config["rootfs"]["diff_ids"][:keep] or len(row["RootFS"]["Layers"]) != keep:
        raise SystemExit("the inspected image's layers are not the source's first --keep-layers layers")
    fields = ("User", "Env", "Cmd", "Entrypoint", "WorkingDir", "Labels", "ExposedPorts", "Volumes", "StopSignal",
              "Shell", "OnBuild", "Healthcheck")
    image_config = {k: row["Config"][k] for k in fields if row["Config"].get(k) not in (None, "", [], {})}
    return {"architecture": row.get("Architecture", "amd64"), "os": row.get("Os", "linux"),
            "created": row.get("Created"), "config": image_config,
            "rootfs": {"type": "layers", "diff_ids": row["RootFS"]["Layers"]},
            "history": [{"created": row.get("Created"), "created_by": f"infrathon: layer {i + 1} of {keep}"}
                        for i in range(keep)]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--max-layer-gib", type=float, default=4.0)
    parser.add_argument("--keep-layers", type=int, help="keep only the source's first N layers")
    parser.add_argument("--config-from-inspect", type=Path,
                        help="with --keep-layers: `docker image inspect` JSON of the image those layers belong to")
    parser.add_argument("--listing", type=Path, help="write the listing of the single flattened source layer")
    args = parser.parse_args()
    if (args.keep_layers is None) != (args.config_from_inspect is None):
        parser.error("--keep-layers and --config-from-inspect go together")
    limit = int(args.max_layer_gib * GIB)
    work = args.workdir.resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"--workdir must be new or empty: {work}")
    (work / "archive").mkdir(parents=True)
    saved = work / "source.tar"
    subprocess.run(["docker", "save", "-o", str(saved), args.source], check=True)
    with tarfile.open(saved, mode="r:") as save:
        manifest = json.load(save.extractfile("manifest.json"))
        if len(manifest) != 1:
            raise SystemExit("expected exactly one image in the saved archive")
        config = json.load(save.extractfile(manifest[0]["Config"]))
        layers = manifest[0]["Layers"]
        if args.keep_layers is not None:
            config = _config_for_prefix(config, args.keep_layers, json.loads(args.config_from_inspect.read_text()))
            layers = layers[:args.keep_layers]
        listing: list[str] | None = [] if args.listing else None
        if listing is not None and len(layers) != 1:
            raise SystemExit("--listing needs a single (flattened) source layer")
        diff_ids, layer_paths, history = [], [], []
        non_empty = [h for h in config["history"] if not h.get("empty_layer")]
        if len(non_empty) != len(layers):
            raise SystemExit("history does not have one non-empty entry per layer")
        layer_for_history = iter(zip(layers, config["rootfs"]["diff_ids"]))
        for entry in config["history"]:
            if entry.get("empty_layer"):
                history.append(entry)
                continue
            layer, diff_id = next(layer_for_history)
            parts, materialize, count = plan_parts(save, layer, limit)
            if count == 1:  # small enough: the saved blob as it is
                kept = work / "archive" / f"kept-{len(layer_paths):03d}.blob"
                with kept.open("wb") as copy:
                    shutil.copyfileobj(save.extractfile(layer), copy)
                diff_ids.append(diff_id)
                layer_paths.append(kept.name)
                history.append(entry)
                continue
            out = work / "archive" / f"layer-{len(layer_paths):03d}"
            out.mkdir()
            written = write_parts(save, layer, parts, materialize, count, out, work / "spool", listing)
            print(f"{layer}: {count} parts, {len(materialize)} hard-link target(s) copied", file=sys.stderr)
            for number, (path, part_digest) in enumerate(written, 1):
                diff_ids.append("sha256:" + part_digest)
                layer_paths.append(str(path.relative_to(work / "archive")))
                history.append({**entry, "created_by": f"{entry.get('created_by', '')} [infrathon split part {number}/{count}]"})
    if listing is not None:
        args.listing.write_text("\n".join(sorted(listing)) + "\n")
    config["rootfs"]["diff_ids"] = diff_ids
    config["history"] = history
    config_bytes = json.dumps(config, separators=(",", ":")).encode()
    (work / "archive" / "config.json").write_bytes(config_bytes)
    (work / "archive" / "manifest.json").write_text(json.dumps(
        [{"Config": "config.json", "RepoTags": [args.target], "Layers": layer_paths}]))
    saved.unlink()
    shutil.rmtree(work / "spool", ignore_errors=True)
    packer = subprocess.Popen(["tar", "-C", str(work / "archive"), "-cf", "-", "."], stdout=subprocess.PIPE)
    loaded = subprocess.run(["docker", "load"], stdin=packer.stdout, check=True)
    packer.stdout.close()
    if packer.wait() != 0 or loaded.returncode != 0:
        raise SystemExit("docker load failed")
    print(json.dumps({"target": args.target, "layers": len(diff_ids),
                      "config_sha256": hashlib.sha256(config_bytes).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
