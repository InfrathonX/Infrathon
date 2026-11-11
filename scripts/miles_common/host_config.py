#!/usr/bin/env python3
"""Host configuration of the Miles task family.

Every host-specific location a Miles task needs -- upstream checkouts, asset roots, the model
cache, scratch -- is resolved here, in this order:

  1. the process environment;
  2. the per-run launch config (launch/miles-*/<name>.env), when the caller passes one;
  3. the host config file: $MLSYSFORGE_CONFIG, else launch/mlsysforge.env (the file
     launch/common.sh reads; keys this module does not know are left to common.sh);
  4. a value derived from another setting (MILES_DATA_ROOT, the Hugging Face cache, this
     checkout);

and otherwise stops with NotConfigured, which names the variable and how to set it. No default
names a host path.

  host_config.py show                     # every variable, its value and where it came from
  host_config.py shell NAME...            # NAME=value lines for `eval` in shell scripts (exit 78 if unset)
  host_config.py scratch TASK_DIR MIN_GIB # create and print the task's scratch root after the disk checks
  host_config.py base-id KIND PINNED_ID   # the base image ID a task build must accept (see base_image_id)
  host_config.py upstream NAME COMMIT     # the MILES_UPSTREAM / MD_UPSTREAM checkout that contains COMMIT
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXIT_NOT_READY = 78

# name -> (derived from, relative part, what it is)
DERIVED: dict[str, tuple[str | None, str | None, str]] = {
    "HARNESS_ROOT": (None, None, "SForge checkout whose .venv/bin/python and sforge/ run the task (default: this checkout)"),
    "MILES_UPSTREAM": (None, None, "radixark/miles clone containing commit 78527b9102b3 (git archive source)"),
    "MD_UPSTREAM": (None, None, "radixark/miles_diffusion clone containing commit 29fde6ee (git archive source)"),
    "MILES_DATA_ROOT": (None, None, "root holding models/miles-*-assets and mlsysforge/* (the Miles /data layout)"),
    "MILES_SCRATCH_ROOT": (None, None, "large, non-root-disk directory for run and build scratch (hundreds of GB per run)"),
    "MILES_HF_CACHE": (None, None, "Hugging Face hub cache holding the pinned model repos"),
    "OPCD_ASSETS": ("MILES_DATA_ROOT", "models/miles-opcd-assets/task-v1", "OPCD task data (public/, private/)"),
    "OPCD_MEGATRON_INIT": ("MILES_DATA_ROOT", "models/miles-opcd-assets/megatron", "initial Megatron checkpoint (Qwen3-1.7B torch_dist)"),
    "OPCD_HF_CACHE": ("MILES_HF_CACHE", "", "model cache for OPCD"),
    "OPCD_ASSETS_V2": ("MILES_DATA_ROOT", "models/miles-opcd-assets/task-v2", "OPCD v2 qualification trees and archives (operator only)"),
    "OPCD_V3_ROOT": (None, None, "OPCD v3 root (assets/task-v3/{public,private,trees,qual}, MedMCQA sources); not under the read-only MILES_DATA_ROOT mirror"),
    "OPCD_ASSETS_V3": ("OPCD_V3_ROOT", "assets/task-v3", "OPCD v3 task data (public/, private/ rebuilt from MedMCQA), qualification trees and archives"),
    "OPD_ASSETS": ("MILES_DATA_ROOT", "models/miles-opd-assets/task-v1", "OPD task data (public/, private/)"),
    "OPD_ASSETS_V2": (None, None, "OPD v2 (math) task data (public/, private/); not under the read-only MILES_DATA_ROOT mirror"),
    "OPD_MEGATRON_INIT": ("MILES_DATA_ROOT", "models/miles-opcd-assets/megatron", "initial Megatron checkpoint (shared with OPCD)"),
    "OPD_HF_CACHE": ("MILES_HF_CACHE", "", "model cache for OPD"),
    "OPD_CALIBRATION_DATA": ("MILES_DATA_ROOT", "models/miles-opd-assets/data", "OPD stage-1/2 calibration datasets gsm8k/, medmcqa-opd/ (operator only)"),
    "OPCD_DATA": ("MILES_DATA_ROOT", "models/miles-opcd-assets/data", "OPCD source datasets medmcqa-dev/, medmcqa-v1/ (operator only; OPD stage 1 reads medmcqa-dev/)"),
    "ALFW_ASSETS": ("MILES_DATA_ROOT", "models/miles-alfworld-assets/task-v1", "ALFWorld task data (public/, private/)"),
    "ALFW_MEGATRON_INIT": ("MILES_DATA_ROOT", "models/miles-alfworld-assets/megatron", "initial Megatron checkpoint (Qwen3-4B torch_dist)"),
    "ALFW_HF_CACHE": ("MILES_HF_CACHE", "", "model cache for ALFWorld"),
    "MDOPD_ASSETS": ("MILES_DATA_ROOT", "models/miles-diffusion-assets", "SD3.5-M weights and teachers"),
    "MDOPD_PUBLIC_DATA": ("MDOPD_ASSETS", "public-01/root", "SD3.5-M public data root"),
    "MDOPD_PRIVATE_DATA": ("MDOPD_ASSETS", "private-01", "SD3.5-M private Judge root (owner not the container agent uid, mode 0700)"),
    "MDOPD_V2_ROOT": (None, None, "SD3.5-M v2 task data root (scorers/, public-02/, private-02/); not under the read-only MILES_DATA_ROOT mirror"),
    "MDOPD_V2_SCORERS": ("MDOPD_V2_ROOT", "scorers", "SD3.5-M v2 task-metric scorers: PickScore_v1/, CLIP-ViT-H-14-processor/"),
    "MDOPD_V2_PUBLIC_DATA": ("MDOPD_V2_ROOT", "public-02/root", "SD3.5-M v2 public data root"),
    "MDOPD_V2_PRIVATE_DATA": ("MDOPD_V2_ROOT", "private-02", "SD3.5-M v2 private Judge root (owner not the container agent uid, mode 0700)"),
    "MDOPD_CPU_CORES": (None, None, "number of physical cores local to the GPU pair that SD3.5-M Work and Judge are pinned to (legacy unpinned fallback: 24; current H100 uses the qualified CPU pinning policy)"),
    "MDWAN_ASSETS": ("MILES_DATA_ROOT", "models/miles-diffusion-wan", "Wan2.2-TI2V-5B, HPSv3, Qwen2-VL-7B"),
    "MDWAN_PUBLIC_DATA": ("MILES_DATA_ROOT", "mlsysforge/md-wandance/public/v1", "Wan2.2 public data root"),
    "MDWAN_PRIVATE_DATA": ("MILES_DATA_ROOT", "mlsysforge/md-wandance/private/v2", "Wan2.2 private Judge root (owner not the container agent uid, mode 0700)"),
    "MDWAN_CPU_CORES": (None, None, "number of physical cores local to the GPU pair that Wan2.2 Work and Judge are pinned to (legacy unpinned fallback: 24; current H100 uses the qualified CPU pinning policy)"),
    "MILES_OLMO3_PACKAGE": ("MILES_DATA_ROOT", "mlsysforge/miles-olmo3-task-package", "OLMo3 v2 task package (public golden, private, reference, negatives)"),
    "MILES_OLMO3_WORK_SCRATCH": ("MILES_SCRATCH_ROOT", "miles-olmo3-work-scratch", "OLMo3 v2 per-run Work scratch volumes and Judge session state (private, no symlinks; up to ~1.5 TiB per run)"),
    "MILES_BASE_IMAGE_RECORDS": (None, None, "build records of scripts/miles_base_images/build.sh (default: outputs/miles-base-images)"),
}
KNOWN = frozenset(DERIVED)

# With INFRATHON_ASSETS_ROOT set, the run-time inputs come from the tasks' Infrathon assets
# (tasks/latest/<task>/assets/manifest.json, fetched by assets/fetch.sh): name -> (task directory,
# path under <INFRATHON_ASSETS_ROOT>/<task directory>; {hardware} = this host's hardware type).  An
# explicit setting still wins; these replace only the values derived from MILES_DATA_ROOT.
_ALFW = "miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed"
_OPCD = "miles-qwen3-opcd-2xh100-v3-seed"
_OPD = "miles-qwen3-opd-scratch-2xh100-v2-seed"
_MDOPD = "miles-diffusion-sd35m-mopd-2xh100-v2-seed"
_MDWAN = "miles-diffusion-wan22-dancegrpo-2xh100-v2-seed"
INFRATHON_ASSET_PATHS: dict[str, tuple[str, str]] = {
    "ALFW_MEGATRON_INIT": (_ALFW, "public/megatron"),
    "OPCD_MEGATRON_INIT": (_OPCD, "public/megatron"),
    "OPD_MEGATRON_INIT": (_OPD, "public/megatron"),
    "MDOPD_ASSETS": (_MDOPD, "public/models"),
    "MDOPD_V2_SCORERS": (_MDOPD, "public/scorers"),
    "MDOPD_V2_PUBLIC_DATA": (_MDOPD, "public/data"),
    "MDOPD_V2_PRIVATE_DATA": (_MDOPD, "judge/private"),
    "MDWAN_ASSETS": (_MDWAN, "public/models"),
    "MDWAN_PUBLIC_DATA": (_MDWAN, "public/data"),
    "MDWAN_PRIVATE_DATA": (_MDWAN, "hw/{hardware}/judge/private"),
}


def _infrathon_hardware() -> str:
    """This host's hardware type, as scripts/infrathon/assets.py resolves it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("infrathon_assets", REPO / "scripts" / "infrathon" / "assets.py")
    assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assets)
    hardware = assets.detect_hardware()
    if hardware is None:
        raise NotConfigured("MDWAN_PRIVATE_DATA", "set INFRATHON_HARDWARE to pick the private root for this card")
    return hardware


class NotConfigured(RuntimeError):
    def __init__(self, name: str, reason: str = ""):
        what = DERIVED.get(name, (None, None, ""))[2]
        source = DERIVED.get(name, (None, None, ""))[0]
        how = f"set {name}"
        if source:
            how += f" or {source}"
        super().__init__(
            f"{name} is not configured{': ' + reason if reason else ''}. {how} in the environment, the launch "
            f"config or the host config file (launch/mlsysforge.env or $MLSYSFORGE_CONFIG). {name} = {what}.")
        self.name = name


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        values[key] = value.strip().strip('"').strip("'")
    return values


def host_config_file() -> Path | None:
    explicit = os.environ.get("MLSYSFORGE_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise NotConfigured("MLSYSFORGE_CONFIG", f"{path} does not exist")
        return path
    default = REPO / "launch" / "mlsysforge.env"
    return default if default.is_file() else None


class HostConfig:
    def __init__(self, run_config: dict[str, str] | None = None, environ: dict[str, str] | None = None):
        self.environ = dict(os.environ if environ is None else environ)
        self.run_config = {k: v for k, v in (run_config or {}).items() if k in KNOWN and v}
        path = host_config_file()
        self.host_file = path
        raw = _parse_env_file(path) if path else {}
        self.host_values = {k: v for k, v in raw.items() if k in KNOWN and v}
        # The asset root and the Hugging Face cache are general launch settings, not Miles keys: take
        # them from the environment, else from the same host config file the other launchers read.
        self.ambient = {k: self.environ.get(k) or raw.get(k, "") for k in ("INFRATHON_ASSETS_ROOT", "HF_HUB_CACHE", "HF_HOME")}

    def source(self, name: str) -> tuple[str, str] | None:
        """(value, where it came from), or None when nothing sets or derives it."""
        if name not in KNOWN:
            raise KeyError(f"unknown Miles host setting {name}")
        if self.environ.get(name):
            return self.environ[name], "environment"
        if name in self.run_config:
            return self.run_config[name], "launch config"
        if name in self.host_values:
            return self.host_values[name], f"host config {self.host_file}"
        if name == "HARNESS_ROOT":
            return str(REPO), "this checkout"
        if name == "MILES_BASE_IMAGE_RECORDS":
            return str(REPO / "outputs" / "miles-base-images"), "this checkout"
        if self.ambient["INFRATHON_ASSETS_ROOT"] and name in INFRATHON_ASSET_PATHS:
            task, relative = INFRATHON_ASSET_PATHS[name]
            if "{hardware}" in relative:
                relative = relative.format(hardware=_infrathon_hardware())
            return str(Path(self.ambient["INFRATHON_ASSETS_ROOT"]).expanduser() / task / relative), "INFRATHON_ASSETS_ROOT"
        if name == "MILES_HF_CACHE":
            if self.ambient["HF_HUB_CACHE"]:
                return self.ambient["HF_HUB_CACHE"], "HF_HUB_CACHE"
            if self.ambient["HF_HOME"]:
                return str(Path(self.ambient["HF_HOME"]) / "hub"), "HF_HOME"
            return str(Path.home() / ".cache" / "huggingface" / "hub"), "default Hugging Face cache"
        parent, relative, _ = DERIVED[name]
        if parent:
            base = self.source(parent)
            if base is not None:
                return str(Path(base[0]) / relative) if relative else base[0], f"derived from {parent}"
        return None

    def value(self, name: str) -> str:
        found = self.source(name)
        if found is None:
            raise NotConfigured(name)
        return found[0]

    def path(self, name: str, *, kind: str = "dir") -> Path:
        path = Path(self.value(name)).expanduser()
        if not path.is_absolute():
            raise NotConfigured(name, f"{path} is not an absolute path")
        ok = path.is_dir() if kind == "dir" else path.is_file() if kind == "file" else path.exists()
        if not ok:
            raise NotConfigured(name, f"{path} is not an existing {kind}")
        return path

    def upstream(self, name: str, commit: str) -> Path:
        """A git checkout that contains ``commit`` (MILES_UPSTREAM, MD_UPSTREAM)."""
        found = self.source(name)
        candidates = [Path(found[0]).expanduser()] if found else []
        if not found:  # a sibling clone next to this checkout, the layout the build scripts used to assume
            candidates.append(REPO.parent / {"MILES_UPSTREAM": "miles", "MD_UPSTREAM": "miles_diffusion"}[name])
        for path in candidates:
            if path.is_dir() and subprocess.run(["git", "-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"],
                                                capture_output=True).returncode == 0:
                return path
        raise NotConfigured(name, f"no checkout containing commit {commit}" + (f" at {candidates[0]}" if candidates else ""))

    def scratch(self, task_dir: str, min_free_gib: int) -> Path:
        """``$MILES_SCRATCH_ROOT/<task_dir>``, created, after the disk checks.

        Refuses a scratch root on the root filesystem (runs write hundreds of GB) unless MILES_SCRATCH_ALLOW_ROOT_FS=1, and refuses less than
        ``min_free_gib`` free there or less than 20 GiB free on /."""
        root = self.path("MILES_SCRATCH_ROOT")
        if os.stat(root).st_dev == os.stat("/").st_dev and self.environ.get("MILES_SCRATCH_ALLOW_ROOT_FS") != "1":
            raise NotConfigured("MILES_SCRATCH_ROOT", f"{root} is on the root filesystem (set MILES_SCRATCH_ALLOW_ROOT_FS=1 to accept)")
        free = _free_gib(root)
        if free < min_free_gib:
            raise NotConfigured("MILES_SCRATCH_ROOT", f"{root} has {free:.0f} GiB free, this task needs {min_free_gib}")
        if _free_gib(Path("/")) < 20:
            raise NotConfigured("MILES_SCRATCH_ROOT", "the root filesystem has less than 20 GiB free")
        target = root / task_dir
        target.mkdir(parents=True, exist_ok=True)
        return target

    def base_image_id(self, kind: str, pinned: str) -> str:
        """Image ID a task build must find for its base image.

        ``MILES_<KIND>_BASE_ID`` if set; else the ``image_id`` that scripts/miles_base_images/build.sh
        recorded on this host (``<MILES_BASE_IMAGE_RECORDS>/<kind>.json``); else the ``pinned`` ID of
        the qualified build."""
        env = self.environ.get(f"MILES_{kind.upper().replace('-', '_')}_BASE_ID")
        if env:
            return env
        record = Path(self.value("MILES_BASE_IMAGE_RECORDS")) / f"{kind}.json"
        if record.is_file():
            return json.loads(record.read_text())["image_id"]
        return pinned


def _free_gib(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    hc = HostConfig()
    try:
        if argv[0] == "show":
            for name in sorted(KNOWN):
                found = hc.source(name)
                print(f"{name:24} {found[0] if found else '-':60} {found[1] if found else 'NOT SET'}")
            return 0
        if argv[0] == "shell":
            for name in argv[1:]:
                print(f"{name}={shlex.quote(hc.value(name))}")
            return 0
        if argv[0] == "scratch" and len(argv) == 3:
            print(hc.scratch(argv[1], int(argv[2])))
            return 0
        if argv[0] == "upstream" and len(argv) == 3:
            print(hc.upstream(argv[1], argv[2]))
            return 0
        if argv[0] == "base-id" and len(argv) == 3:
            print(hc.base_image_id(argv[1], argv[2]))
            return 0
    except NotConfigured as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_READY
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
