"""Strict, backend-neutral scientific task contracts.

Portable execution profiles contain requirements only. Actual hardware,
drivers, containers, nodes, measurement campaigns, and baseline epochs belong
to a later runtime attestation and must never enter this task-side contract.
Deployment-value lint below is defense in depth, not a security proof or a
substitute for runtime attestation.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


PORTABLE_SCHEMA_VERSION = 1
CANONICAL_JSON_VERSION = "sforge-canonical-json-v1"
BASELINE_POLICIES = frozenset(
    {"none", "cached_once_per_campaign", "paired_per_candidate"}
)
COMPARABILITY_POLICIES = frozenset({"same_measurement_profile", "portable"})
EXECUTION_VENDORS = frozenset({"nvidia", "amd", "intel", "any"})
PORTABLE_COMPARABLE_UNITS = frozenset({"correctness", "dimensionless"})

_PORTABLE_KEYS = {
    "schema_version",
    "task_core",
    "evaluation_suites",
    "execution_profiles",
    "metrics",
    "defaults",
}
_TASK_CORE_KEYS = {
    "contract_id",
    "definition",
    "interface",
    "allowed_changes",
    "invariants",
    "resource_manifest_sha256",
}
_SUITE_KEYS = {"roles", "oracle", "selection"}
_ROLE_KEYS = {"manifest_sha256", "case_count"}
_PROTOCOL_DIGEST_KEYS = {"protocol_id", "spec_sha256"}
_PROFILE_KEYS = {
    "vendor",
    "count",
    "min_memory_bytes",
    "min_compute_capability",
    "features",
    "software_abi",
    "calibration_protocol",
}
_METRIC_KEYS = {
    "direction",
    "unit",
    "comparability",
    "aggregation",
    "baseline_policy",
    "baseline",
}
_AGGREGATION_KEYS = {"protocol_id", "spec"}
_DEFAULT_KEYS = {
    "evaluation_suite_id",
    "execution_profile_id",
    "metric_id",
}
_LOGICAL_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_NVIDIA_CAPABILITY_RE = re.compile(r"(?:[1-9]|1[0-9])\.(?:0|[1-9])")
_GPU_UUID_VALUE_RE = re.compile(
    r"(?i)(?:(?:MIG-)?GPU-[0-9a-f]{8}(?:-[0-9a-f-]{4,})?|"
    r"MIG-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_GPU_MODEL_VALUE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:H100|H200|GH200|GB200|B100|B200|A100|A6000|"
    r"MI300X|MI325X)(?![A-Za-z0-9])"
)
_DEVICE_PATH_VALUE_RE = re.compile(
    r"(?i)/dev/(?:kfd|nvidia(?:ctl|uvm|modeset|[0-9]+)|"
    r"dri/(?:renderD|card|controlD)[0-9]+)(?![A-Za-z0-9])"
)
_IPV4_VALUE_RE = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
_BARE_HOST_VALUE_RE = re.compile(
    r"(?i)^(?:gpu-)?(?:host|node|worker)-[a-z0-9.-]+$"
)
_OCI_TAG_VALUE_RE = re.compile(
    r"^(?:[a-z0-9][a-z0-9._-]*(?::[0-9]+)?/)*"
    r"[a-z0-9][a-z0-9._-]*:[A-Za-z0-9_.-]+$"
)
_OCI_DIGEST_REF_RE = re.compile(
    r"(?i)^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$"
)
_PCI_BDF_VALUE_RE = re.compile(
    r"(?i)^(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$"
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_INTERNAL_HOST_VALUE_RE = re.compile(
    r"(?i)^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.(?:internal|local)$"
)
_URL_VALUE_RE = re.compile(r"(?i)^[a-z][a-z0-9+.-]*://")
_DEPLOYMENT_VALUE_HINT_KEYS = frozenset(
    {
        "deployment",
        "endpoint",
        "host",
        "hostname",
        "location",
        "placement",
        "placement_hint",
        "target_host",
        "worker",
        "worker_name",
    }
)

_PHYSICAL_KEY_NAMES = {
    "accelerator_model",
    "accelerator_name",
    "backend",
    "container_id",
    "container_ids",
    "container_name",
    "cuda_visible_devices",
    "device_id",
    "device_ids",
    "device_uuid",
    "device_uuids",
    "docker_container_id",
    "docker_image",
    "gpu_id",
    "gpu_ids",
    "gpu_model",
    "gpu_name",
    "gpu_uuid",
    "gpu_uuids",
    "hardware",
    "hardware_model",
    "host_path",
    "host_paths",
    "image",
    "image_ref",
    "image_tag",
    "mount_path",
    "node_name",
    "node_selector",
    "pci_bus_id",
    "pci_bus_ids",
    "physical_device",
    "physical_devices",
    "physical_gpu",
}


class PortableSpecError(ValueError):
    """A portable task contract is malformed or deployment-specific."""


def _child_path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _normal_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def _is_physical_key(key: str) -> bool:
    normalized = _normal_key(key)
    tokens = set(normalized.split("_"))
    composed_device_identity = bool(
        tokens & {"gpu", "device", "accelerator"}
        and tokens & {"id", "ids", "uuid", "uuids"}
    )
    return (
        normalized in _PHYSICAL_KEY_NAMES
        or normalized.endswith(("_container_id", "_device_id", "_pci_bus_id"))
        or composed_device_identity
    )


def _is_deployment_value_hint(key: str) -> bool:
    normalized = _normal_key(key)
    tokens = set(normalized.split("_"))
    return (
        normalized in _DEPLOYMENT_VALUE_HINT_KEYS
        or "endpoint" in tokens
        or "placement" in tokens
        or normalized.endswith(("_host", "_hostname"))
    )


def _physical_string_kind(
    value: str,
    *,
    deployment_hint: bool,
) -> str | None:
    """Identify obvious deployment literals; this is lint, not attestation."""

    stripped = value.strip()
    if _SHA256_RE.fullmatch(stripped):
        return None
    if _GPU_UUID_VALUE_RE.search(stripped):
        return "GPU UUID"
    if _GPU_MODEL_VALUE_RE.search(stripped):
        return "fixed GPU model"
    if _DEVICE_PATH_VALUE_RE.search(stripped):
        return "host device path"
    if (
        stripped.startswith("/")
        or stripped.startswith(("../", "./"))
        or _WINDOWS_ABSOLUTE_PATH_RE.match(stripped)
    ):
        return "absolute host path"
    if _PCI_BDF_VALUE_RE.fullmatch(stripped):
        return "PCI device address"
    if _OCI_DIGEST_REF_RE.fullmatch(stripped):
        return "OCI image reference"
    try:
        parsed_ip = ipaddress.ip_address(stripped.strip("[]"))
    except ValueError:
        parsed_ip = None
    if parsed_ip is not None and (
        parsed_ip.version == 6
        or deployment_hint
    ):
        return "host address"
    if (
        stripped.casefold() == "localhost"
        or "[::1]" in stripped
        or stripped == "::1"
    ):
        return "host address"
    if _INTERNAL_HOST_VALUE_RE.fullmatch(stripped):
        return "host name"
    if deployment_hint and _IPV4_VALUE_RE.search(stripped):
        return "host address"
    if deployment_hint and _BARE_HOST_VALUE_RE.fullmatch(stripped):
        return "host name"
    if deployment_hint and _URL_VALUE_RE.match(stripped):
        return "deployment endpoint"
    if deployment_hint and (
        stripped.casefold().endswith(":latest")
        or _OCI_TAG_VALUE_RE.fullmatch(stripped)
    ):
        return "mutable OCI tag"
    return None


def _validate_bundle_path(value: object, *, path: str) -> None:
    if not isinstance(value, str) or not value or _CONTROL_RE.search(value):
        raise PortableSpecError(
            f"{path} must be a non-empty bundle-relative POSIX path"
        )
    if "\\" in value or PurePosixPath(value).is_absolute():
        raise PortableSpecError(f"{path} must be a bundle-relative POSIX path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise PortableSpecError(
            f"{path} must not contain empty, '.' or '..' path components"
        )


def _validate_json(
    value: object,
    *,
    path: str,
    reject_physical: bool,
    deployment_hint: bool = False,
) -> None:
    """Validate the sforge-canonical-json-v1 value domain."""

    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        physical_kind = (
            _physical_string_kind(value, deployment_hint=deployment_hint)
            if reject_physical
            else None
        )
        if physical_kind is not None:
            raise PortableSpecError(
                f"{path} contains a physical deployment value ({physical_kind}); "
                "portable contracts must declare capabilities instead"
            )
        return
    if isinstance(value, float):
        raise PortableSpecError(
            f"{path} must not contain floats; use an integer or decimal string"
        )
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(
                item,
                path=f"{path}[{index}]",
                reject_physical=reject_physical,
                deployment_hint=deployment_hint,
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or _LOGICAL_KEY_RE.fullmatch(key) is None:
                raise PortableSpecError(
                    f"{path} contains invalid key {key!r}; canonical JSON keys "
                    "must be stable ASCII logical keys"
                )
            item_path = _child_path(path, key)
            if reject_physical and _is_physical_key(key):
                raise PortableSpecError(
                    f"{item_path} is a physical deployment identity; portable "
                    "contracts must declare capabilities instead"
                )
            if key == "path" or key.endswith("_path"):
                _validate_bundle_path(item, path=item_path)
            _validate_json(
                item,
                path=item_path,
                reject_physical=reject_physical,
                deployment_hint=(
                    deployment_hint or _is_deployment_value_hint(key)
                ),
            )
        return
    raise PortableSpecError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def canonical_json(value: object) -> str:
    """Encode a value with ``sforge-canonical-json-v1``."""

    _validate_json(value, path="value", reject_physical=False)
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    """Hash the UTF-8 bytes of ``sforge-canonical-json-v1`` JSON."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def verify_sha256_bytes(payload: bytes, declared_sha256: str) -> str:
    """Verify bytes against a declared digest and return the observed digest."""

    if not isinstance(payload, bytes):
        raise PortableSpecError("digest verification payload must be bytes")
    expected = _sha256(declared_sha256, path="declared_sha256")
    observed = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if observed != expected:
        raise PortableSpecError(
            f"content SHA-256 mismatch: declared {expected}, observed {observed}"
        )
    return observed


def verify_sha256_file(path: Path, declared_sha256: str) -> str:
    """Verify one regular, non-symlink file against a declared digest."""

    expected = _sha256(declared_sha256, path="declared_sha256")
    descriptor = -1
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    try:
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PortableSpecError("digest verification path must be a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    except PortableSpecError:
        raise
    except OSError as exc:
        raise PortableSpecError(
            f"cannot read digest verification file: {path}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    observed = f"sha256:{digest.hexdigest()}"
    if observed != expected:
        raise PortableSpecError(
            f"content SHA-256 mismatch: declared {expected}, observed {observed}"
        )
    return observed


def _object(
    value: object, *, path: str, nonempty: bool = False
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PortableSpecError(f"{path} must be an object")
    if nonempty and not value:
        raise PortableSpecError(f"{path} must not be empty")
    return value


def _exact_keys(
    value: Mapping[str, object],
    *,
    path: str,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise PortableSpecError(f"{path} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise PortableSpecError(f"{path} is missing field(s): {', '.join(missing)}")


def _logical_id(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _LOGICAL_KEY_RE.fullmatch(value) is None:
        raise PortableSpecError(
            f"{path} must be a 1-128 character ASCII logical identifier"
        )
    return value


def _sha256(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PortableSpecError(
            f"{path} must be canonical sha256:<64 lowercase hex>"
        )
    return value


def _nonempty_strings(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PortableSpecError(f"{path} must be a non-empty string array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise PortableSpecError(f"{path} must contain only non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class ProtocolDigest:
    protocol_id: str
    spec_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"protocol_id": self.protocol_id, "spec_sha256": self.spec_sha256}


@dataclass(frozen=True)
class TaskCore:
    """Scientific task definition with a declared resource-manifest identity.

    Parsing records ``resource_manifest_sha256`` but deliberately performs no
    filesystem lookup. Bundle assembly/admission must call a verifier helper.
    """

    contract_id: str
    definition: Mapping[str, Any]
    interface: Mapping[str, Any]
    allowed_changes: tuple[str, ...]
    invariants: tuple[str, ...]
    resource_manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "definition", _freeze_json(self.definition))
        object.__setattr__(self, "interface", _freeze_json(self.interface))
        object.__setattr__(self, "allowed_changes", tuple(self.allowed_changes))
        object.__setattr__(self, "invariants", tuple(self.invariants))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "definition": _plain_json(self.definition),
            "interface": _plain_json(self.interface),
            "allowed_changes": list(self.allowed_changes),
            "invariants": list(self.invariants),
            "resource_manifest_sha256": self.resource_manifest_sha256,
        }


@dataclass(frozen=True)
class EvaluationRole:
    manifest_sha256: str
    case_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "case_count": self.case_count,
        }


@dataclass(frozen=True)
class EvaluationSuite:
    roles: Mapping[str, EvaluationRole]
    oracle: ProtocolDigest
    selection: ProtocolDigest

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": {
                role_id: role.to_dict() for role_id, role in self.roles.items()
            },
            "oracle": self.oracle.to_dict(),
            "selection": self.selection.to_dict(),
        }


@dataclass(frozen=True)
class ExecutionProfile:
    """Capability and software requirements, never a measured deployment."""

    vendor: str
    count: int
    min_memory_bytes: int
    features: tuple[str, ...]
    software_abi: ProtocolDigest
    calibration_protocol: ProtocolDigest
    min_compute_capability: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "vendor": self.vendor,
            "count": self.count,
            "min_memory_bytes": self.min_memory_bytes,
            "features": list(self.features),
        }
        if self.min_compute_capability is not None:
            value["min_compute_capability"] = self.min_compute_capability
        value["software_abi"] = self.software_abi.to_dict()
        value["calibration_protocol"] = self.calibration_protocol.to_dict()
        return value


@dataclass(frozen=True)
class Aggregation:
    protocol_id: str
    spec: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec", _freeze_json(self.spec))

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_id": self.protocol_id, "spec": _plain_json(self.spec)}


@dataclass(frozen=True)
class MetricSpec:
    direction: str
    unit: str
    comparability: str
    aggregation: Aggregation
    baseline_policy: str
    baseline: ProtocolDigest | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "direction": self.direction,
            "unit": self.unit,
            "comparability": self.comparability,
            "aggregation": self.aggregation.to_dict(),
            "baseline_policy": self.baseline_policy,
        }
        if self.baseline is not None:
            value["baseline"] = self.baseline.to_dict()
        return value


@dataclass(frozen=True)
class PortableDefaults:
    evaluation_suite_id: str
    execution_profile_id: str
    metric_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "evaluation_suite_id": self.evaluation_suite_id,
            "execution_profile_id": self.execution_profile_id,
            "metric_id": self.metric_id,
        }


@dataclass(frozen=True)
class PortableSpec:
    """A deeply immutable, normalized portable scientific contract."""

    task_core: TaskCore
    evaluation_suites: Mapping[str, EvaluationSuite]
    execution_profiles: Mapping[str, ExecutionProfile]
    metrics: Mapping[str, MetricSpec]
    defaults: PortableDefaults
    schema_version: int = PORTABLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_suites",
            MappingProxyType(dict(self.evaluation_suites)),
        )
        object.__setattr__(
            self,
            "execution_profiles",
            MappingProxyType(dict(self.execution_profiles)),
        )
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive mutable copy of the normalized contract."""

        return {
            "schema_version": self.schema_version,
            "task_core": self.task_core.to_dict(),
            "evaluation_suites": {
                suite_id: suite.to_dict()
                for suite_id, suite in self.evaluation_suites.items()
            },
            "execution_profiles": {
                profile_id: profile.to_dict()
                for profile_id, profile in self.execution_profiles.items()
            },
            "metrics": {
                metric_id: metric.to_dict()
                for metric_id, metric in self.metrics.items()
            },
            "defaults": self.defaults.to_dict(),
        }

    @property
    def layer_digests(self) -> dict[str, str]:
        value = self.to_dict()
        return {
            layer: canonical_sha256(value[layer])
            for layer in (
                "task_core",
                "evaluation_suites",
                "execution_profiles",
                "metrics",
                "defaults",
            )
        }

    @property
    def task_core_sha256(self) -> str:
        return self.layer_digests["task_core"]

    @property
    def evaluation_suites_sha256(self) -> str:
        return self.layer_digests["evaluation_suites"]

    @property
    def execution_profiles_sha256(self) -> str:
        return self.layer_digests["execution_profiles"]

    @property
    def metrics_sha256(self) -> str:
        return self.layer_digests["metrics"]

    @property
    def defaults_sha256(self) -> str:
        return self.layer_digests["defaults"]

    @property
    def evaluation_contract_sha256(self) -> str:
        return canonical_sha256(
            {
                "canonicalization": CANONICAL_JSON_VERSION,
                "schema_version": self.schema_version,
                "layers": self.layer_digests,
            }
        )

    def digest_descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonicalization": CANONICAL_JSON_VERSION,
            "layer_sha256": self.layer_digests,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
        }


def _parse_protocol_digest(value: object, *, path: str) -> ProtocolDigest:
    raw = _object(value, path=path)
    _exact_keys(
        raw,
        path=path,
        allowed=_PROTOCOL_DIGEST_KEYS,
        required=_PROTOCOL_DIGEST_KEYS,
    )
    return ProtocolDigest(
        protocol_id=_logical_id(raw["protocol_id"], path=f"{path}.protocol_id"),
        spec_sha256=_sha256(raw["spec_sha256"], path=f"{path}.spec_sha256"),
    )


def _parse_task_core(value: object, *, path: str) -> TaskCore:
    raw = _object(value, path=path)
    _exact_keys(
        raw,
        path=path,
        allowed=_TASK_CORE_KEYS,
        required=_TASK_CORE_KEYS,
    )
    definition = _object(raw["definition"], path=f"{path}.definition", nonempty=True)
    interface = _object(raw["interface"], path=f"{path}.interface", nonempty=True)
    _validate_json(definition, path=f"{path}.definition", reject_physical=True)
    _validate_json(interface, path=f"{path}.interface", reject_physical=True)
    return TaskCore(
        contract_id=_logical_id(raw["contract_id"], path=f"{path}.contract_id"),
        definition=definition,
        interface=interface,
        allowed_changes=_nonempty_strings(
            raw["allowed_changes"], path=f"{path}.allowed_changes"
        ),
        invariants=_nonempty_strings(raw["invariants"], path=f"{path}.invariants"),
        resource_manifest_sha256=_sha256(
            raw["resource_manifest_sha256"],
            path=f"{path}.resource_manifest_sha256",
        ),
    )


def _parse_role(value: object, *, path: str) -> EvaluationRole:
    raw = _object(value, path=path)
    _exact_keys(raw, path=path, allowed=_ROLE_KEYS, required=_ROLE_KEYS)
    case_count = raw["case_count"]
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count <= 0
    ):
        raise PortableSpecError(f"{path}.case_count must be a positive integer")
    return EvaluationRole(
        manifest_sha256=_sha256(
            raw["manifest_sha256"], path=f"{path}.manifest_sha256"
        ),
        case_count=case_count,
    )


def _parse_suite(value: object, *, path: str) -> EvaluationSuite:
    raw = _object(value, path=path)
    _exact_keys(raw, path=path, allowed=_SUITE_KEYS, required=_SUITE_KEYS)
    raw_roles = _object(raw["roles"], path=f"{path}.roles", nonempty=True)
    roles = {
        _logical_id(role_id, path=f"{path}.roles key"): _parse_role(
            role, path=f"{path}.roles.{role_id}"
        )
        for role_id, role in raw_roles.items()
    }
    missing_roles = sorted({"public", "final"} - set(roles))
    if missing_roles:
        raise PortableSpecError(
            f"{path}.roles is missing required role(s): {', '.join(missing_roles)}"
        )
    return EvaluationSuite(
        roles=roles,
        oracle=_parse_protocol_digest(raw["oracle"], path=f"{path}.oracle"),
        selection=_parse_protocol_digest(
            raw["selection"], path=f"{path}.selection"
        ),
    )


def _parse_profile(value: object, *, path: str) -> ExecutionProfile:
    raw = _object(value, path=path)
    _validate_json(raw, path=path, reject_physical=True)
    _exact_keys(
        raw,
        path=path,
        allowed=_PROFILE_KEYS,
        required=_PROFILE_KEYS - {"min_compute_capability"},
    )
    vendor = raw["vendor"]
    if not isinstance(vendor, str):
        raise PortableSpecError(f"{path}.vendor must be a string")
    vendor = vendor.strip().casefold()
    if vendor not in EXECUTION_VENDORS:
        raise PortableSpecError(
            f"{path}.vendor must be one of: {', '.join(sorted(EXECUTION_VENDORS))}"
        )
    count = raw["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise PortableSpecError(f"{path}.count must be a positive integer")
    memory = raw["min_memory_bytes"]
    if isinstance(memory, bool) or not isinstance(memory, int) or memory < 0:
        raise PortableSpecError(
            f"{path}.min_memory_bytes must be a non-negative integer"
        )
    capability = raw.get("min_compute_capability")
    if vendor == "nvidia":
        if (
            not isinstance(capability, str)
            or _NVIDIA_CAPABILITY_RE.fullmatch(capability) is None
        ):
            raise PortableSpecError(
                f"{path}.min_compute_capability is required for nvidia and "
                "must be canonical major.minor"
            )
    elif "min_compute_capability" in raw:
        raise PortableSpecError(
            f"{path}.min_compute_capability is NVIDIA-specific and forbidden "
            f"for vendor {vendor}"
        )
    features_value = raw["features"]
    if not isinstance(features_value, list):
        raise PortableSpecError(f"{path}.features must be a string array")
    features = tuple(
        _logical_id(feature.strip().casefold(), path=f"{path}.features[{index}]")
        if isinstance(feature, str)
        else _logical_id(feature, path=f"{path}.features[{index}]")
        for index, feature in enumerate(features_value)
    )
    if len(features) != len(set(features)):
        raise PortableSpecError(f"{path}.features must not contain duplicates")
    calibration_protocol = _parse_protocol_digest(
        raw["calibration_protocol"],
        path=f"{path}.calibration_protocol",
    )
    return ExecutionProfile(
        vendor=vendor,
        count=count,
        min_memory_bytes=memory,
        features=tuple(sorted(features)),
        software_abi=_parse_protocol_digest(
            raw["software_abi"], path=f"{path}.software_abi"
        ),
        calibration_protocol=calibration_protocol,
        min_compute_capability=capability,
    )


def _parse_aggregation(value: object, *, path: str) -> Aggregation:
    raw = _object(value, path=path)
    _exact_keys(raw, path=path, allowed=_AGGREGATION_KEYS, required=_AGGREGATION_KEYS)
    spec = _object(raw["spec"], path=f"{path}.spec", nonempty=True)
    _validate_json(spec, path=f"{path}.spec", reject_physical=True)
    return Aggregation(
        protocol_id=_logical_id(raw["protocol_id"], path=f"{path}.protocol_id"),
        spec=spec,
    )


def _parse_metric(value: object, *, path: str) -> MetricSpec:
    raw = _object(value, path=path)
    _exact_keys(
        raw,
        path=path,
        allowed=_METRIC_KEYS,
        required=_METRIC_KEYS - {"baseline"},
    )
    direction = raw["direction"]
    if direction not in {"maximize", "minimize"}:
        raise PortableSpecError(f"{path}.direction must be maximize or minimize")
    unit = raw["unit"]
    if not isinstance(unit, str) or not unit.strip():
        raise PortableSpecError(f"{path}.unit must be a non-empty string")
    comparability = raw["comparability"]
    if comparability not in COMPARABILITY_POLICIES:
        raise PortableSpecError(
            f"{path}.comparability must be one of: "
            f"{', '.join(sorted(COMPARABILITY_POLICIES))}"
        )
    normalized_unit = unit.strip()
    if (
        comparability == "portable"
        and normalized_unit.casefold() not in PORTABLE_COMPARABLE_UNITS
    ):
        raise PortableSpecError(
            f"{path}.comparability portable is only valid for units: "
            f"{', '.join(sorted(PORTABLE_COMPARABLE_UNITS))}"
        )
    baseline_policy = raw["baseline_policy"]
    if baseline_policy not in BASELINE_POLICIES:
        raise PortableSpecError(
            f"{path}.baseline_policy must be one of: "
            f"{', '.join(sorted(BASELINE_POLICIES))}"
        )
    baseline_present = "baseline" in raw
    if baseline_policy == "none" and baseline_present:
        raise PortableSpecError(f"{path}.baseline is forbidden when policy is none")
    if baseline_policy != "none" and not baseline_present:
        raise PortableSpecError(
            f"{path}.baseline is required when baseline_policy is {baseline_policy}"
        )
    return MetricSpec(
        direction=direction,
        unit=normalized_unit,
        comparability=comparability,
        aggregation=_parse_aggregation(
            raw["aggregation"], path=f"{path}.aggregation"
        ),
        baseline_policy=baseline_policy,
        baseline=(
            _parse_protocol_digest(raw["baseline"], path=f"{path}.baseline")
            if baseline_present
            else None
        ),
    )


def _parse_logical_map(
    value: object,
    *,
    path: str,
    parser: Any,
) -> dict[str, Any]:
    raw = _object(value, path=path, nonempty=True)
    result: dict[str, Any] = {}
    for raw_id, item in raw.items():
        item_id = _logical_id(raw_id, path=f"{path} key")
        result[item_id] = parser(item, path=f"{path}.{item_id}")
    return result


def parse_portable_spec(value: object, *, task_id: str = "<unknown>") -> PortableSpec:
    """Parse, normalize, and deeply snapshot a task's ``portable`` object."""

    path = f"Task '{task_id}' portable"
    raw = _object(value, path=path)
    _validate_json(raw, path=path, reject_physical=True)
    _exact_keys(raw, path=path, allowed=_PORTABLE_KEYS, required=_PORTABLE_KEYS)
    if (
        isinstance(raw["schema_version"], bool)
        or raw["schema_version"] != PORTABLE_SCHEMA_VERSION
    ):
        raise PortableSpecError(
            f"{path}.schema_version must be {PORTABLE_SCHEMA_VERSION}"
        )

    task_core = _parse_task_core(raw["task_core"], path=f"{path}.task_core")
    suites = _parse_logical_map(
        raw["evaluation_suites"],
        path=f"{path}.evaluation_suites",
        parser=_parse_suite,
    )
    profiles = _parse_logical_map(
        raw["execution_profiles"],
        path=f"{path}.execution_profiles",
        parser=_parse_profile,
    )
    metrics = _parse_logical_map(
        raw["metrics"], path=f"{path}.metrics", parser=_parse_metric
    )

    raw_defaults = _object(raw["defaults"], path=f"{path}.defaults")
    _exact_keys(
        raw_defaults,
        path=f"{path}.defaults",
        allowed=_DEFAULT_KEYS,
        required=_DEFAULT_KEYS,
    )
    defaults = PortableDefaults(
        evaluation_suite_id=_logical_id(
            raw_defaults["evaluation_suite_id"],
            path=f"{path}.defaults.evaluation_suite_id",
        ),
        execution_profile_id=_logical_id(
            raw_defaults["execution_profile_id"],
            path=f"{path}.defaults.execution_profile_id",
        ),
        metric_id=_logical_id(
            raw_defaults["metric_id"], path=f"{path}.defaults.metric_id"
        ),
    )
    references = (
        ("evaluation_suite_id", defaults.evaluation_suite_id, suites),
        ("execution_profile_id", defaults.execution_profile_id, profiles),
        ("metric_id", defaults.metric_id, metrics),
    )
    for field, selected, choices in references:
        if selected not in choices:
            raise PortableSpecError(
                f"{path}.defaults.{field} references unknown ID '{selected}'"
            )

    return PortableSpec(
        task_core=task_core,
        evaluation_suites=suites,
        execution_profiles=profiles,
        metrics=metrics,
        defaults=defaults,
    )


# ---------------------------------------------------------------------------
# Capability envelope: the lightweight ``portable`` declaration.
#
# Tasks that have not been promoted to the full portable contract still declare
# their capability envelope (execution profiles, metric comparability, default
# profile) in the same ``portable`` object.  The envelope is requirement
# metadata enforced by the trusted task runtime; it carries no TaskCore digest
# and therefore never produces an evaluation-contract identity.  A ``portable``
# object that names ``task_core`` or ``evaluation_suites`` is a full contract
# and must satisfy :func:`parse_portable_spec` in full; the envelope parser
# never accepts a half-promoted contract.
# ---------------------------------------------------------------------------

_FULL_CONTRACT_MARKER_KEYS = frozenset({"task_core", "evaluation_suites"})
_ENVELOPE_KEYS = {"schema_version", "execution_profiles", "metrics", "defaults"}
_ENVELOPE_PROFILE_KEYS = {
    "vendor",
    "count",
    "min_memory_bytes",
    "min_compute_capability",
    "features",
    "interconnect",
    "required_nvlink_pairs",
    "min_host_memory_bytes",
    "software_abi",
    "notes",
}
_ENVELOPE_PROFILE_REQUIRED = {"vendor", "count", "min_memory_bytes"}
_ENVELOPE_METRIC_KEYS = {
    "comparability",
    "baseline_policy",
    "direction",
    "unit",
    "notes",
}
_ENVELOPE_METRIC_REQUIRED = {"comparability", "baseline_policy"}
_ENVELOPE_DEFAULT_KEYS = {"execution_profile_id", "metric_id"}
_ENVELOPE_FREE_TEXT_KEYS = frozenset({"notes"})


def is_full_portable_contract(value: object) -> bool:
    """True when a ``portable`` object declares the full scientific contract."""

    return isinstance(value, Mapping) and bool(
        _FULL_CONTRACT_MARKER_KEYS & set(value)
    )


def _free_text(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PortableSpecError(f"{path} must be a non-empty string")
    if _CONTROL_RE.search(value):
        raise PortableSpecError(f"{path} must not contain control characters")
    return value


def _without_free_text(value: object) -> object:
    """Strip free-text commentary so structured fields alone face the lint."""

    if isinstance(value, Mapping):
        return {
            key: _without_free_text(item)
            for key, item in value.items()
            if key not in _ENVELOPE_FREE_TEXT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_free_text(item) for item in value]
    return value


def _optional_non_negative_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PortableSpecError(f"{path} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class EnvelopeProfile:
    """Capability requirement for one execution profile (no runtime facts)."""

    vendor: str
    count: int
    min_memory_bytes: int
    features: tuple[str, ...] = ()
    min_compute_capability: str | None = None
    interconnect: str | None = None
    required_nvlink_pairs: tuple[tuple[int, int], ...] = ()
    min_host_memory_bytes: int | None = None
    software_abi: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "vendor": self.vendor,
            "count": self.count,
            "min_memory_bytes": self.min_memory_bytes,
        }
        if self.min_compute_capability is not None:
            payload["min_compute_capability"] = self.min_compute_capability
        if self.features:
            payload["features"] = list(self.features)
        if self.interconnect is not None:
            payload["interconnect"] = self.interconnect
        if self.required_nvlink_pairs:
            payload["required_nvlink_pairs"] = [list(pair) for pair in self.required_nvlink_pairs]
        if self.min_host_memory_bytes is not None:
            payload["min_host_memory_bytes"] = self.min_host_memory_bytes
        if self.software_abi is not None:
            payload["software_abi"] = self.software_abi
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload


@dataclass(frozen=True)
class EnvelopeMetric:
    comparability: str
    baseline_policy: str
    direction: str | None = None
    unit: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "comparability": self.comparability,
            "baseline_policy": self.baseline_policy,
        }
        if self.direction is not None:
            payload["direction"] = self.direction
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload


@dataclass(frozen=True)
class EnvelopeDefaults:
    execution_profile_id: str
    metric_id: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"execution_profile_id": self.execution_profile_id}
        if self.metric_id is not None:
            payload["metric_id"] = self.metric_id
        return payload


@dataclass(frozen=True)
class CapabilityEnvelope:
    """Parsed lightweight ``portable`` declaration.

    It is deliberately a different type from :class:`PortableSpec`: nothing
    that consumes an evaluation-contract digest can accept it by accident.
    """

    execution_profiles: Mapping[str, EnvelopeProfile]
    metrics: Mapping[str, EnvelopeMetric]
    defaults: EnvelopeDefaults

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_profiles",
            MappingProxyType(dict(self.execution_profiles)),
        )
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def default_profile(self) -> EnvelopeProfile:
        return self.execution_profiles[self.defaults.execution_profile_id]

    @property
    def default_metric(self) -> EnvelopeMetric | None:
        if self.defaults.metric_id is None:
            return None
        return self.metrics[self.defaults.metric_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PORTABLE_SCHEMA_VERSION,
            "execution_profiles": {
                profile_id: profile.to_dict()
                for profile_id, profile in self.execution_profiles.items()
            },
            "metrics": {
                metric_id: metric.to_dict()
                for metric_id, metric in self.metrics.items()
            },
            "defaults": self.defaults.to_dict(),
        }

    @property
    def envelope_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _parse_envelope_profile(value: object, *, path: str) -> EnvelopeProfile:
    raw = _object(value, path=path)
    _exact_keys(
        raw,
        path=path,
        allowed=_ENVELOPE_PROFILE_KEYS,
        required=_ENVELOPE_PROFILE_REQUIRED,
    )
    _validate_json(_without_free_text(raw), path=path, reject_physical=True)
    vendor = raw["vendor"]
    if not isinstance(vendor, str):
        raise PortableSpecError(f"{path}.vendor must be a string")
    vendor = vendor.strip().casefold()
    if vendor not in EXECUTION_VENDORS:
        raise PortableSpecError(
            f"{path}.vendor must be one of: {', '.join(sorted(EXECUTION_VENDORS))}"
        )
    count = raw["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise PortableSpecError(f"{path}.count must be a positive integer")
    memory = _optional_non_negative_int(
        raw["min_memory_bytes"], path=f"{path}.min_memory_bytes"
    )
    capability = raw.get("min_compute_capability")
    if vendor == "nvidia":
        if (
            not isinstance(capability, str)
            or _NVIDIA_CAPABILITY_RE.fullmatch(capability) is None
        ):
            raise PortableSpecError(
                f"{path}.min_compute_capability is required for nvidia and "
                "must be canonical major.minor"
            )
    elif "min_compute_capability" in raw:
        raise PortableSpecError(
            f"{path}.min_compute_capability is NVIDIA-specific and forbidden "
            f"for vendor {vendor}"
        )
    features: tuple[str, ...] = ()
    if "features" in raw:
        features_value = raw["features"]
        if not isinstance(features_value, list) or not features_value:
            raise PortableSpecError(f"{path}.features must be a non-empty string array")
        normalized = []
        for index, feature in enumerate(features_value):
            if not isinstance(feature, str) or not feature.strip():
                raise PortableSpecError(
                    f"{path}.features[{index}] must be a non-empty string"
                )
            normalized.append(feature.strip().casefold())
        if len(normalized) != len(set(normalized)):
            raise PortableSpecError(f"{path}.features must not contain duplicates")
        features = tuple(sorted(normalized))
    host_memory = (
        _optional_non_negative_int(
            raw["min_host_memory_bytes"], path=f"{path}.min_host_memory_bytes"
        )
        if "min_host_memory_bytes" in raw
        else None
    )
    required_nvlink_pairs: tuple[tuple[int, int], ...] = ()
    if "required_nvlink_pairs" in raw:
        pairs = raw["required_nvlink_pairs"]
        pair_path = f"{path}.required_nvlink_pairs"
        if vendor != "nvidia":
            raise PortableSpecError(f"{pair_path} is NVIDIA-specific")
        if not isinstance(pairs, list):
            raise PortableSpecError(f"{pair_path} must be an array of local GPU index pairs")
        normalized_pairs = []
        for index, pair in enumerate(pairs):
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(gpu) is not int or not 0 <= gpu < count for gpu in pair)
                    or pair[0] == pair[1]):
                raise PortableSpecError(
                    f"{pair_path}[{index}] must name two distinct local GPU indices within count"
                )
            normalized_pairs.append(tuple(sorted(pair)))
        if len(set(normalized_pairs)) != len(normalized_pairs):
            raise PortableSpecError(f"{pair_path} must not contain duplicate pairs")
        required_nvlink_pairs = tuple(sorted(normalized_pairs))
    return EnvelopeProfile(
        vendor=vendor,
        count=count,
        min_memory_bytes=memory,
        features=features,
        min_compute_capability=capability,
        interconnect=(
            _free_text(raw["interconnect"], path=f"{path}.interconnect")
            if "interconnect" in raw
            else None
        ),
        required_nvlink_pairs=required_nvlink_pairs,
        min_host_memory_bytes=host_memory,
        software_abi=(
            _free_text(raw["software_abi"], path=f"{path}.software_abi")
            if "software_abi" in raw
            else None
        ),
        notes=_free_text(raw["notes"], path=f"{path}.notes") if "notes" in raw else None,
    )


def _parse_envelope_metric(value: object, *, path: str) -> EnvelopeMetric:
    raw = _object(value, path=path)
    _exact_keys(
        raw,
        path=path,
        allowed=_ENVELOPE_METRIC_KEYS,
        required=_ENVELOPE_METRIC_REQUIRED,
    )
    _validate_json(_without_free_text(raw), path=path, reject_physical=True)
    direction = raw.get("direction")
    if "direction" in raw and direction not in {"maximize", "minimize"}:
        raise PortableSpecError(f"{path}.direction must be maximize or minimize")
    unit = raw.get("unit")
    if "unit" in raw and (not isinstance(unit, str) or not unit.strip()):
        raise PortableSpecError(f"{path}.unit must be a non-empty string")
    return EnvelopeMetric(
        comparability=_logical_id(
            raw["comparability"], path=f"{path}.comparability"
        ),
        baseline_policy=_logical_id(
            raw["baseline_policy"], path=f"{path}.baseline_policy"
        ),
        direction=direction,
        unit=unit.strip() if isinstance(unit, str) else None,
        notes=_free_text(raw["notes"], path=f"{path}.notes") if "notes" in raw else None,
    )


def parse_capability_envelope(
    value: object, *, task_id: str = "<unknown>"
) -> CapabilityEnvelope:
    """Parse the lightweight ``portable`` capability envelope.

    Raises :class:`PortableSpecError` for a full contract: those must go
    through :func:`parse_portable_spec` so a partially promoted task can never
    be silently downgraded to an envelope.
    """

    path = f"Task '{task_id}' portable"
    raw = _object(value, path=path)
    if is_full_portable_contract(raw):
        raise PortableSpecError(
            f"{path} declares a full portable contract; parse it with "
            "parse_portable_spec"
        )
    _exact_keys(raw, path=path, allowed=_ENVELOPE_KEYS, required=_ENVELOPE_KEYS)
    if (
        isinstance(raw["schema_version"], bool)
        or raw["schema_version"] != PORTABLE_SCHEMA_VERSION
    ):
        raise PortableSpecError(
            f"{path}.schema_version must be {PORTABLE_SCHEMA_VERSION}"
        )
    profiles = _parse_logical_map(
        raw["execution_profiles"],
        path=f"{path}.execution_profiles",
        parser=_parse_envelope_profile,
    )
    metrics = _parse_logical_map(
        raw["metrics"], path=f"{path}.metrics", parser=_parse_envelope_metric
    )
    raw_defaults = _object(raw["defaults"], path=f"{path}.defaults")
    _exact_keys(
        raw_defaults,
        path=f"{path}.defaults",
        allowed=_ENVELOPE_DEFAULT_KEYS,
        required={"execution_profile_id"},
    )
    defaults = EnvelopeDefaults(
        execution_profile_id=_logical_id(
            raw_defaults["execution_profile_id"],
            path=f"{path}.defaults.execution_profile_id",
        ),
        metric_id=(
            _logical_id(raw_defaults["metric_id"], path=f"{path}.defaults.metric_id")
            if "metric_id" in raw_defaults
            else None
        ),
    )
    if defaults.execution_profile_id not in profiles:
        raise PortableSpecError(
            f"{path}.defaults.execution_profile_id references unknown ID "
            f"'{defaults.execution_profile_id}'"
        )
    if defaults.metric_id is not None and defaults.metric_id not in metrics:
        raise PortableSpecError(
            f"{path}.defaults.metric_id references unknown ID '{defaults.metric_id}'"
        )
    return CapabilityEnvelope(
        execution_profiles=profiles, metrics=metrics, defaults=defaults
    )
