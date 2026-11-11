#!/usr/bin/env python3
"""Trusted public/calibration/final controller for closed-loop mini-SWE workloads."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

from runner.metrics import geometric_mean
from runner.corpus import corpus_fingerprint, swebench_partition_ready
from runner.workload import Profile, load_model_adapter_contract, load_profiles, run_benchmark


MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://model-api:8100").rstrip("/")
BASELINE_URL = os.environ.get("BASELINE_URL", "http://model-api:8100/v1").rstrip("/")
CANDIDATE_URL = os.environ.get("CANDIDATE_URL", "http://main:9000/v1").rstrip("/")
TOKEN_FILE = Path(os.environ.get("ADMIN_TOKEN_FILE", "/run/kvbench-admin/token"))
FEEDBACK_ROOT = Path(os.environ.get("FEEDBACK_ROOT", "/run/public-feedback"))
TRUSTED_RESULTS_ROOT = Path(os.environ.get("TRUSTED_RESULTS_ROOT", "/run/kvbench-results"))
PUBLIC_FAILURE_SLOT_COUNT = 8
PUBLIC_FAILURE_FILE_MAX_BYTES = 4 * 1024
PUBLIC_FAILURE_TOTAL_MAX_BYTES = 32 * 1024
_PUBLIC_FAILURE_LOCK = threading.Lock()
_PUBLIC_FAILURE_SATURATED = False
PUBLIC_PROFILES = Path(os.environ.get("PUBLIC_PROFILES", "/opt/bench-controller/profiles/public.json"))
CALIBRATION_PROFILES = Path(
    os.environ.get("CALIBRATION_PROFILES", "/opt/bench-controller/profiles/calibration.json")
)
HIDDEN_PROFILES = Path(os.environ.get("HIDDEN_PROFILES", "/opt/bench-controller/profiles/hidden.json"))
SCORING_POLICY = Path(os.environ.get("SCORING_POLICY", "/opt/bench-controller/profiles/scoring-policy.json"))
CORPUS_MANIFEST = Path(
    os.environ.get("SWEBENCH_CORPUS_ROOT", "/opt/kvbench-corpus")
) / "manifest.json"
QWEN_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LEGACY_PROFILE_ID = "qwen3.6-27b-6a9e13bd-h100-nvl-tp2-kv30g-v2"
LEGACY_PROFILE_SHA256 = (
    "7289ed315324b9b5165d4d680efb93004c59e0d34f0f98ef9d85406ed5a9153a"
)
LEGACY_RUNTIME_IMAGE_ID = (
    "sha256:1d8369b9e3c91aa8cba5c7aeeef26e3c4afd455e5d2ce9aebe6f30f6dd9344bd"
)
# Revision v2.5 states the backend requirement as a capability predicate
# (2 devices, compute capability >= 8.0, >= 95000 MiB each) instead of an
# "NVIDIA H100 NVL" name whitelist.  Both revisions describe the same serving
# system, so a leg or anchor recorded under either digest pairs with a leg
# measured under the other; the launcher preflight re-derives that equivalence
# from the two contract files before a run starts.
BACKEND_SERVING_CONTRACT_SHA256 = (
    "1cf6d2e6ff7af8448d6b7735590621d083d6a5071252bc60b40c9756ffe328c4"
)
LEGACY_BACKEND_SERVING_CONTRACT_SHA256 = (
    "44e4a3408b298a5fdc53d7bb84f0739a22f047da1974212754a98b5755777ba8"
)
PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256 = (
    "18d27d97c736f90e1d74822e80ce1f212685e26627b31a08e47569a1457f52c6"
)
ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256 = frozenset(
    {
        BACKEND_SERVING_CONTRACT_SHA256,
        LEGACY_BACKEND_SERVING_CONTRACT_SHA256,
        PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256,
    }
)
STABLE_MODEL_PROVENANCE_FIELDS = (
    "profile_id",
    "profile_sha256",
    "model_revision",
    "runtime_image_id",
    "tensor_parallel_size",
    "kv_cache_dtype",
)
LIFECYCLE_ATTESTATION_CONTEXT = b"kvbench-final-candidate-lifecycle-v1\n"
MAIN_EGRESS_SEAL_CONTEXT = b"kvbench-main-egress-seal-v2\n"
MAIN_EGRESS_SEAL_ACTION = "seal_main_egress"
MAIN_EGRESS_CHALLENGE_TTL_SEC = 120.0
FINAL_LEG_ORDER_CONTEXT = b"kvbench-final-leg-order-v1\n"
FINAL_FORMAL_LEG_COUNT = 2
RELEASE_PUBLIC_SUITE = "public"
RELEASE_PUBLIC_PROFILE = "public-pressure-192"
RELEASE_PUBLIC_ID = "qwen36-27b-swe192-v2-1-gpu45"
RELEASE_PUBLIC_DIRECT_ANCHOR_KIND = "kvbench-release-public-direct-anchor"
RELEASE_PUBLIC_DIRECT_ANCHOR_FILE_ENV = "RELEASE_PUBLIC_DIRECT_ANCHOR_FILE"
RELEASE_PUBLIC_DIRECT_ANCHOR_SHA256_ENV = "RELEASE_PUBLIC_DIRECT_ANCHOR_SHA256"
RELEASE_PUBLIC_DIRECT_ANCHOR_REQUIRED_ENV = (
    "RELEASE_PUBLIC_DIRECT_ANCHOR_REQUIRED"
)
RELEASE_PUBLIC_DIRECT_ANCHOR_MAX_BYTES = 64 * 1024 * 1024

# Candidate-visible failures are deliberately a closed vocabulary.  Raw
# exception strings, controller objects, and benchmark failure records belong
# only in the root-owned evidence stores below; they must never be reflected by
# a /public endpoint.
PUBLIC_ERROR_SPECS: dict[str, tuple[int, str]] = {
    "invalid_json": (400, "the request body is invalid"),
    "invalid_mode": (400, "the requested benchmark mode is invalid"),
    "invalid_profile": (400, "the requested benchmark profile is invalid"),
    "unknown_or_disabled_profile": (404, "the requested public profile is unavailable"),
    "benchmark_lease_busy": (409, "another trusted benchmark operation is active"),
    "baseline_required": (409, "a new trusted baseline is required"),
    "baseline_infrastructure_failure": (409, "the trusted baseline is not reusable"),
    "baseline_audit_invalid": (409, "the trusted baseline audit is invalid"),
    "baseline_model_identity_invalid": (409, "the trusted model runtime changed; rerun baseline"),
    "baseline_not_pressurized": (409, "the trusted baseline pressure gate failed"),
    "baseline_window_invalid": (409, "the trusted baseline measurement window is invalid"),
    "baseline_quality_invalid": (409, "the trusted baseline quality gate failed"),
    "baseline_interval_invalid": (409, "the trusted baseline timing contract is invalid"),
    "baseline_controller_identity_invalid": (409, "the trusted controller changed; rerun baseline"),
    "benchmark_failed": (500, "the trusted benchmark leg failed"),
    "missing_valid_pair": (404, "a complete public pair is required"),
    "pair_not_ready": (409, "the latest public pair is not ready"),
    "invalid_latest_pair": (409, "the latest public pair failed a trusted gate"),
    "public_internal_error": (500, "the trusted public controller failed"),
}

PUBLIC_PAIR_STATUSES = frozenset(
    {
        "baseline_running",
        "baseline_complete",
        "baseline_failed",
        "candidate_running",
        "candidate_complete",
        "candidate_failed",
    }
)

PUBLIC_FAILURE_TYPES = frozenset(
    {
        "backend_admission_stall",
        "candidate_gateway_admission_stall",
        "candidate_leg_wall_timeout",
        "candidate_watchdog_audit_epoch_changed",
        "candidate_watchdog_telemetry_invalid",
        "measurement_window_control_failed",
        "model_response_limit_exceeded",
        "workload_watchdog_failure",
    }
)

PUBLIC_PRESSURE_REASON_CODES = frozenset(
    {
        "insufficient_kv_samples",
        "kv_p95_below_floor",
        "metrics_sampler_error",
        "metrics_sampling_gap",
        "metrics_sampling_timestamp_error",
        "pressure_duration_below_floor",
        "pressure_evidence_invalid",
    }
)

_FINITE_INTERVAL_FAILURES = frozenset(
    {
        "agent_run_count",
        "duration_clock",
        "end_event",
        "finite_window_contract",
        "incomplete",
        "live_samples",
        "metrics_sampler_errors",
        "missing",
        "post_end_live_sample",
        "rate_identity",
        "schema_fields",
        "schema_version",
        "start_event",
        "timestamps",
        "top_level_elapsed_sec",
        "top_level_ended_unix",
        "top_level_started_unix",
        "unix_order",
        "wall_elapsed",
    }
)

PUBLIC_PAIR_GATE_CODES = frozenset(
    {
        "unclassified_pair_gate",
        "baseline_infrastructure_failure",
        "baseline_candidate_failure",
        "baseline_audit_failed",
        "candidate_infrastructure_failure",
        "candidate_candidate_failure",
        "candidate_audit_failed",
        "baseline_profile_not_pressurized",
        "workload_pair_mismatch",
        "corpus_mismatch",
        "baseline_model_provenance_missing",
        "candidate_model_provenance_missing",
        "baseline_model_provenance_pair_state_mismatch",
        "candidate_model_provenance_pair_state_mismatch",
        "model_provenance_mismatch",
        "fixed_window_contains_finite_agent_execution_interval",
        "measurement_window_contract_mismatch",
        *(
            f"{leg}_{gate}"
            for leg in ("baseline", "candidate")
            for gate in (
                "measurement_window_not_reached",
                "measurement_window_failed",
                "window_zero_progress_above_0.05",
                "window_request_success_below_0.98",
            )
        ),
        *(
            f"{leg}_agent_execution_interval_{failure}"
            for leg in ("baseline", "candidate")
            for failure in _FINITE_INTERVAL_FAILURES
        ),
    }
)


class ClassifiedLegFailure(RuntimeError):
    """A trusted controller attribution for a leg that produced no result."""

    def __init__(self, classification: str, failure_type: str, detail: str) -> None:
        if classification not in {"candidate_failed", "infra_invalid"}:
            raise ValueError(f"invalid leg-failure classification: {classification}")
        if not re.fullmatch(r"[a-z0-9_]+", failure_type):
            raise ValueError(f"invalid leg-failure type: {failure_type}")
        self.classification = classification
        self.failure_type = failure_type
        self.detail = detail[:2000]
        super().__init__(f"{failure_type}: {self.detail}")


def candidate_leg_failure(failure_type: str, detail: str) -> ClassifiedLegFailure:
    return ClassifiedLegFailure("candidate_failed", failure_type, detail)


def infrastructure_leg_failure(failure_type: str, detail: str) -> ClassifiedLegFailure:
    return ClassifiedLegFailure("infra_invalid", failure_type, detail)


def classified_failure_fields(
    classification: str,
    failure_type: str,
    detail: str,
) -> dict[str, Any]:
    """Build the fail-closed score fragment consumed by the shared verifier."""

    if classification not in {"candidate_failed", "infra_invalid"}:
        raise ValueError(f"invalid terminal classification: {classification}")
    if not re.fullmatch(r"[a-z0-9_]+", failure_type):
        raise ValueError(f"invalid terminal failure type: {failure_type}")
    reason = f"trusted_controller:{failure_type}"
    return {
        "status": classification,
        "hard_gates_passed": False,
        "infra_failures": [reason] if classification == "infra_invalid" else [],
        "candidate_failures": [reason] if classification == "candidate_failed" else [],
        "performance_score": 0.0,
        "tail_score": 0.0,
        "reward": 0.0,
        "failure_attribution": {
            "schema_version": 1,
            "authority": "trusted-bench-controller",
            "classification": classification,
            "type": failure_type,
            "detail": detail[:2000],
        },
        "profiles": {},
    }


def token_text() -> str:
    return TOKEN_FILE.read_text().strip()


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token_text()}"}


def require_admin(request: web.Request) -> None:
    supplied = request.headers.get("Authorization", "")
    expected = f"Bearer {token_text()}"
    if not secrets.compare_digest(supplied, expected):
        raise web.HTTPNotFound()


def _hex_text(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def formal_leg_order(workload_nonce_sha256: str) -> tuple[str, str]:
    """Derive the concealed formal A/B order from the public nonce commitment.

    The raw workload nonce never enters the final report.  Domain-separating a
    hash of its SHA-256 commitment gives the exporter and verifier a stable way
    to recompute the order without access to held-out workload state.
    """

    if not _hex_text(workload_nonce_sha256, 64):
        raise ValueError("workload nonce commitment must be lowercase SHA-256 hex")
    digest = hashlib.sha256(
        FINAL_LEG_ORDER_CONTEXT + workload_nonce_sha256.encode("ascii")
    ).digest()
    if digest[0] & 1:
        return ("candidate", "baseline")
    return ("baseline", "candidate")


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"network-seal {label} has an invalid schema")
    return value


def _network_seal_record(value: Any) -> dict[str, Any]:
    record = _exact_object(
        value,
        {"network_id", "internal", "compose_project", "compose_network"},
        "network record",
    )
    if not _hex_text(record["network_id"], 64):
        raise ValueError("network-seal record has an invalid network ID")
    if not isinstance(record["internal"], bool):
        raise ValueError("network-seal record has an invalid Internal flag")
    for key in ("compose_project", "compose_network"):
        if record[key] is not None and not isinstance(record[key], str):
            raise ValueError(f"network-seal record has an invalid {key}")
    return record


def validate_main_egress_seal_attestation(
    attestation: Any,
    *,
    expected_challenge: str,
    admin_token: str,
    seen_nonces: set[str],
) -> dict[str, Any]:
    """Authenticate a worker-observed, same-project Docker network seal."""

    evidence = _exact_object(
        attestation,
        {
            "schema_version",
            "action",
            "challenge",
            "attestation_nonce",
            "attested_unix_ns",
            "worker",
            "target",
            "controller",
            "topology",
            "signature",
        },
        "attestation",
    )
    signature = evidence["signature"]
    unsigned = {key: value for key, value in evidence.items() if key != "signature"}
    canonical = json.dumps(
        unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    expected_signature = hmac.new(
        admin_token.encode(), MAIN_EGRESS_SEAL_CONTEXT + canonical, hashlib.sha256
    ).hexdigest()
    if not _hex_text(signature, 64) or not hmac.compare_digest(
        signature, expected_signature
    ):
        raise ValueError("network-seal HMAC is invalid")
    if evidence["schema_version"] != 2 or isinstance(evidence["schema_version"], bool):
        raise ValueError("network-seal schema version is invalid")
    if evidence["action"] != MAIN_EGRESS_SEAL_ACTION:
        raise ValueError("network-seal action is invalid")
    if not _hex_text(expected_challenge, 64) or not secrets.compare_digest(
        str(evidence["challenge"]), expected_challenge
    ):
        raise ValueError("network-seal challenge is stale or mismatched")
    nonce = evidence["attestation_nonce"]
    if not _hex_text(nonce, 64) or nonce in seen_nonces:
        raise ValueError("network-seal nonce is invalid or replayed")
    attested_unix_ns = evidence["attested_unix_ns"]
    if (
        isinstance(attested_unix_ns, bool)
        or not isinstance(attested_unix_ns, int)
        or attested_unix_ns <= 0
        or abs(time.time_ns() - attested_unix_ns) > 5 * 60 * 1_000_000_000
    ):
        raise ValueError("network-seal timestamp is invalid")

    worker = _exact_object(
        evidence["worker"],
        {"container_id", "image_id", "compose_project", "compose_service"},
        "worker identity",
    )
    target = _exact_object(
        evidence["target"],
        {"container_id", "image_id", "compose_project", "compose_service"},
        "target identity",
    )
    controller = _exact_object(
        evidence["controller"],
        {"container_id", "image_id", "compose_project", "compose_service"},
        "controller identity",
    )
    if (
        worker["compose_service"] != "sandbox-worker"
        or target["compose_service"] != "main"
        or controller["compose_service"] != "bench-controller"
    ):
        raise ValueError("network-seal Compose services are invalid")
    identities = (worker, target, controller)
    if any(
        not _hex_text(identity["container_id"], 64)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(identity["image_id"])) is None
        for identity in identities
    ):
        raise ValueError("network-seal container or image identity is invalid")
    if len({identity["container_id"] for identity in identities}) != 3:
        raise ValueError("network-seal trusted services must be distinct containers")
    project = worker["compose_project"]
    if (
        not isinstance(project, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", project) is None
        or target["compose_project"] != project
        or controller["compose_project"] != project
    ):
        raise ValueError("network-seal Compose project identity is invalid")

    topology = _exact_object(
        evidence["topology"],
        {
            "front_network_id",
            "before",
            "detached_network_ids",
            "after",
            "non_internal_before",
            "non_internal_after",
            "front_preserved",
            "already_sealed",
        },
        "topology",
    )
    if not isinstance(topology["before"], list) or not isinstance(topology["after"], list):
        raise ValueError("network-seal topology lists are invalid")
    before = [_network_seal_record(record) for record in topology["before"]]
    after = [_network_seal_record(record) for record in topology["after"]]
    before_ids = [record["network_id"] for record in before]
    after_ids = [record["network_id"] for record in after]
    if not before or not after or len(before_ids) != len(set(before_ids)) or len(after_ids) != len(
        set(after_ids)
    ):
        raise ValueError("network-seal topology has empty or duplicate network IDs")
    external_before = [record for record in before if record["internal"] is False]
    external_after = [record for record in after if record["internal"] is False]
    detached = topology["detached_network_ids"]
    if (
        not isinstance(detached, list)
        or any(not _hex_text(network_id, 64) for network_id in detached)
        or len(detached) != len(set(detached))
        or set(detached) != {record["network_id"] for record in external_before}
    ):
        raise ValueError("network-seal detached network evidence is invalid")
    expected_after = {
        record["network_id"] for record in before if record["internal"] is True
    }
    if set(after_ids) != expected_after or external_after:
        raise ValueError("network-seal target retained or gained an external network")
    front_id = topology["front_network_id"]
    front_before = [
        record
        for record in before
        if record["network_id"] == front_id
        and record["internal"] is True
        and record["compose_project"] == project
        and record["compose_network"] == "front"
    ]
    front_after = [
        record
        for record in after
        if record["network_id"] == front_id
        and record["internal"] is True
        and record["compose_project"] == project
        and record["compose_network"] == "front"
    ]
    if len(front_before) != 1 or len(front_after) != 1 or topology["front_preserved"] is not True:
        raise ValueError("network-seal did not preserve the internal front network")
    if (
        isinstance(topology["non_internal_before"], bool)
        or topology["non_internal_before"] != len(external_before)
        or isinstance(topology["non_internal_after"], bool)
        or topology["non_internal_after"] != 0
        or not isinstance(topology["already_sealed"], bool)
        or topology["already_sealed"] != (len(external_before) == 0)
    ):
        raise ValueError("network-seal topology counters are invalid")

    seen_nonces.add(nonce)
    return {
        "schema_version": 2,
        "protocol": "same-compose-main-egress-hmac-v2",
        "sealed": True,
        "controller_image_id": controller["image_id"],
        "external_network_count_before": len(external_before),
        "detached_network_count": len(detached),
        "external_network_count_after": 0,
        "internal_front_preserved": True,
        "already_sealed": topology["already_sealed"],
        "attestation_sha256": hashlib.sha256(canonical).hexdigest(),
        "compose_project_sha256": hashlib.sha256(project.encode()).hexdigest(),
        "target_container_sha256": hashlib.sha256(
            target["container_id"].encode()
        ).hexdigest(),
        "controller_container_sha256": hashlib.sha256(
            controller["container_id"].encode()
        ).hexdigest(),
        "front_network_sha256": hashlib.sha256(front_id.encode()).hexdigest(),
    }


def validate_lifecycle_attestation(
    session: dict[str, Any],
    attestation: Any,
    *,
    expected_mode: str,
    expected_leg_ordinal: int,
    admin_token: str,
) -> dict[str, Any]:
    """Authenticate one verifier-owned candidate reset epoch and consume it."""

    if not isinstance(attestation, dict):
        raise ValueError("missing candidate lifecycle attestation")
    signature = attestation.get("signature")
    unsigned = {key: value for key, value in attestation.items() if key != "signature"}
    canonical = json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode()
    expected_signature = hmac.new(
        admin_token.encode(), LIFECYCLE_ATTESTATION_CONTEXT + canonical, hashlib.sha256
    ).hexdigest()
    if not _hex_text(signature, 64) or not hmac.compare_digest(signature, expected_signature):
        raise ValueError("candidate lifecycle signature is invalid")

    expected_values = {
        "schema_version": 1,
        "session_id": session["session_id"],
        "leg_ordinal": expected_leg_ordinal,
        "mode": expected_mode,
        "challenge": session["pending_challenge"],
        "frozen_submission_sha256": session["frozen_submission_sha256"],
        "prelaunch_candidate_pids": [],
        "port_closed_before_launch": True,
        "runtime_state_scrubbed": True,
        "candidate_owned_runtime_entries": [],
        "candidate_owned_ipc_objects": [],
    }
    for key, value in expected_values.items():
        if attestation.get(key) != value:
            raise ValueError(f"candidate lifecycle attestation mismatch: {key}")
    epoch_nonce = attestation.get("epoch_nonce")
    if not _hex_text(epoch_nonce, 32) or epoch_nonce in session["seen_epoch_nonces"]:
        raise ValueError("candidate lifecycle epoch nonce is invalid or replayed")
    terminated = attestation.get("terminated_candidate_pids")
    if (
        not isinstance(terminated, list)
        or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid < 2
            for pid in terminated
        )
        or len(set(terminated)) != len(terminated)
    ):
        raise ValueError("candidate lifecycle terminated PID evidence is invalid")

    candidate_pids = attestation.get("candidate_pids")
    if (
        not isinstance(candidate_pids, list)
        or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid < 2
            for pid in candidate_pids
        )
        or len(set(candidate_pids)) != len(candidate_pids)
    ):
        raise ValueError("candidate lifecycle live PID evidence is invalid")
    gateway_pid = attestation.get("gateway_pid")
    gateway_start_ticks = attestation.get("gateway_start_ticks")
    if expected_mode == "candidate":
        if (
            isinstance(gateway_pid, bool)
            or not isinstance(gateway_pid, int)
            or gateway_pid < 2
            or isinstance(gateway_start_ticks, bool)
            or not isinstance(gateway_start_ticks, int)
            or gateway_start_ticks < 1
            or gateway_pid not in candidate_pids
            or attestation.get("gateway_healthy_after_launch") is not True
        ):
            raise ValueError("candidate lifecycle fresh gateway evidence is invalid")
        process_identity = f"{gateway_pid}:{gateway_start_ticks}"
        if process_identity in session["seen_candidate_process_identities"]:
            raise ValueError("candidate lifecycle reused a gateway process epoch")
        session["seen_candidate_process_identities"].add(process_identity)
    elif (
        gateway_pid is not None
        or gateway_start_ticks is not None
        or candidate_pids
        or attestation.get("gateway_healthy_after_launch") is not False
    ):
        raise ValueError("baseline leg must be attested candidate-quiescent")

    session["seen_epoch_nonces"].add(epoch_nonce)
    summary = {
        "schema_version": 1,
        "leg_ordinal": expected_leg_ordinal,
        "mode": expected_mode,
        "frozen_submission_sha256": session["frozen_submission_sha256"],
        "challenge_sha256": hashlib.sha256(attestation["challenge"].encode()).hexdigest(),
        "epoch_nonce_sha256": hashlib.sha256(epoch_nonce.encode()).hexdigest(),
        "attestation_sha256": hashlib.sha256(canonical).hexdigest(),
        "terminated_candidate_process_count": len(terminated),
    }
    if expected_mode == "candidate":
        summary["gateway_process_identity_sha256"] = hashlib.sha256(
            f"{gateway_pid}:{gateway_start_ticks}".encode()
        ).hexdigest()
    return summary


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_contract_sha256(policy: dict[str, Any]) -> str:
    immutable = {
        key: value
        for key, value in policy.items()
        if key not in {"release_ready", "not_ready_reasons"}
    }
    encoded = json.dumps(immutable, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def public_profiles_ready(profiles: dict[str, Profile]) -> bool:
    return all(
        name in profiles and profiles[name].enabled
        for name in ("smoke-api", "public-12", "public-pressure-192")
    )


def real_workload_ready(profiles: dict[str, Profile]) -> bool:
    """Fail closed until the pressure leg is a real SWE-bench workload."""
    profile = profiles.get("public-pressure-192")
    diagnostic = profiles.get("public-12")
    return bool(
        profile
        and diagnostic
        and profile.enabled
        and diagnostic.enabled
        and profile.workload_family == "swebench-lite"
        and diagnostic.workload_family == "swebench-lite"
        and profile.corpus_partition == "public"
        and diagnostic.corpus_partition == "public"
        and profile.workflows == 192
        and profile.concurrency == 192
        and profile.max_steps == 100
        and profile.context_bytes == 0
        and profile.max_tokens == 2048
        and profile.workflow_timeout_sec == 4500
        and profile.request_timeout_sec == 2700
        and profile.measurement_window_sec == 600
        and profile.measurement_warmup_sec == 0
        and profile.tool_delay_min_sec == 0.0
        and profile.tool_delay_max_sec == 0.0
        and profile.warmup_requests == 4
        and profile.warmup_context_bytes == 0
        and profile.warmup_steps == 3
        and profile.metrics_sample_interval_sec == 1.0
        and profile.metrics_request_timeout_sec == 30
        and profile.pressure_required
        and profile.pressure_kv_threshold == 0.8
        and profile.pressure_kv_p95_min == 0.9
        and profile.pressure_time_fraction_min == 0.25
        and profile.pressure_min_samples == 60
        and profile.backend_stall_timeout_sec == 180
        and profile.candidate_admission_stall_timeout_sec == 300
        and swebench_partition_ready(
            "public",
            max(
                profile.corpus_unique_instances_min,
                diagnostic.corpus_unique_instances_min,
            ),
        )
    )


def validate_policy(policy: dict[str, Any], hidden_profiles: dict[str, Profile]) -> None:
    entries = policy.get("hidden_profiles")
    if not isinstance(entries, list) or not entries:
        raise ValueError("scoring policy has no hidden profiles")
    if len(entries) != 1:
        raise ValueError("formal hidden scoring requires exactly one A/B profile")
    names = [entry.get("name") for entry in entries if isinstance(entry, dict)]
    enabled_names = {name for name, profile in hidden_profiles.items() if profile.enabled}
    if len(names) != len(entries) or len(set(names)) != len(names) or set(names) != enabled_names:
        raise ValueError("scoring policy and enabled hidden profiles do not match exactly")
    if any(float(entry.get("weight", 0)) <= 0 for entry in entries):
        raise ValueError("hidden profile weights must be positive")
    # Hidden profiles may be finite or trusted fixed-window workloads.  The
    # latter intentionally score throughput without waiting for SWE task
    # completion; Profile validates the one-wave shape and total wall budget.
    noise_floor = float(policy["noise_floor"])
    if not 0 < noise_floor <= 1:
        raise ValueError("noise floor must be greater than zero and at most one")
    for entry in entries:
        reference = entry.get("reference_speedup")
        if not isinstance(reference, (int, float)) or isinstance(reference, bool):
            raise ValueError(f"invalid reference speedup for {entry['name']}")
        if not math.isfinite(float(reference)) or float(reference) <= noise_floor:
            raise ValueError(f"invalid reference speedup for {entry['name']}")
        if policy.get("release_ready") and float(reference) <= 1.0:
            raise ValueError(
                f"release-ready reference speedup must be greater than one for {entry['name']}"
            )
    if policy.get("release_ready"):
        if str(policy.get("release_id", "")).startswith("development"):
            raise ValueError("release-ready policy still has a development release ID")
        for name in enabled_names:
            profile = hidden_profiles[name]
            if profile.workload_family != "swebench-lite":
                raise ValueError(f"release-ready hidden profile is not SWE-bench Lite: {name}")
            if not profile.pressure_required:
                raise ValueError(f"release-ready hidden profile has no pressure validity gate: {name}")
            if not swebench_partition_ready(
                profile.corpus_partition,
                profile.corpus_unique_instances_min,
            ):
                raise ValueError(f"release-ready hidden corpus is not fully staged: {name}")
    for key in ("min_request_success_rate", "min_valid_step_ratio"):
        if not 0 <= float(policy[key]) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    value = policy.get("min_tests_passed_workflows")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("min_tests_passed_workflows must be a positive integer")
    value = policy.get("min_completed_workflows")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("min_completed_workflows must be a non-negative integer")
    value = policy.get("minimum_workflow_progress")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("minimum_workflow_progress must be a non-negative integer")
    for key in ("tests_passed_workflows_tolerance", "completed_workflows_tolerance"):
        value = policy.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    value = policy.get("max_p99_step_latency_ratio")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 1.0
    ):
        raise ValueError("max_p99_step_latency_ratio must be finite and greater than one")
    value = policy.get("min_baseline_valid_steps_per_min")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("min_baseline_valid_steps_per_min must be finite and positive")
    if float(policy["backend_work_ratio_min"]) > float(policy["backend_work_ratio_max"]):
        raise ValueError("backend work ratio bounds are inverted")


async def model_status(app: web.Application) -> tuple[bool, bool]:
    healthy = False
    contract_available = False
    try:
        async with app["client"].get(f"{MODEL_API_URL}/health") as response:
            healthy = response.status == 200
            await response.read()
        async with app["client"].get(
            f"{MODEL_API_URL}/admin/backend-contract", headers=admin_headers()
        ) as response:
            contract_available = response.status == 200
            await response.read()
    except (OSError, asyncio.TimeoutError):
        pass
    return healthy, contract_available


def project_stable_model_provenance(provenance: object) -> dict[str, Any]:
    """Project current or legacy provenance onto the stable serving contract.

    Legacy Direct evidence contains a per-process boot digest.  It is
    intentionally ignored: the authoritative identity is the topology-free
    backend serving contract.  The six older fields remain diagnostic
    metadata and are not a substitute for that contract digest.
    """

    if not isinstance(provenance, dict):
        raise ValueError("model provenance is not an object")
    projected = {
        field: provenance.get(field) for field in STABLE_MODEL_PROVENANCE_FIELDS
    }
    if (
        not isinstance(projected["profile_id"], str)
        or not projected["profile_id"]
        or not _hex_text(projected["profile_sha256"], 64)
        or projected["model_revision"] != QWEN_REVISION
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(projected["runtime_image_id"])
        )
        is None
        or projected["tensor_parallel_size"] != 2
        or projected["kv_cache_dtype"] != "fp8"
    ):
        raise ValueError("model provenance metadata is invalid")
    digest = provenance.get("backend_serving_contract_sha256")
    if digest is not None and digest not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256:
        raise ValueError("backend serving contract digest is unexpected")
    if digest is None:
        if projected != {
            "profile_id": LEGACY_PROFILE_ID,
            "profile_sha256": LEGACY_PROFILE_SHA256,
            "model_revision": QWEN_REVISION,
            "runtime_image_id": LEGACY_RUNTIME_IMAGE_ID,
            "tensor_parallel_size": 2,
            "kv_cache_dtype": "fp8",
        }:
            raise ValueError("legacy model provenance cannot be upgraded safely")
        digest = LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    projected["backend_serving_contract_sha256"] = digest
    return projected


def same_backend_serving_contract(left: object, right: object) -> bool:
    try:
        left_contract = project_stable_model_provenance(left)[
            "backend_serving_contract_sha256"
        ]
        right_contract = project_stable_model_provenance(right)[
            "backend_serving_contract_sha256"
        ]
    except ValueError:
        return False
    return (
        left_contract in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        and right_contract in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
    )


async def model_provenance(app: web.Application) -> dict[str, Any]:
    async with app["client"].get(
        f"{MODEL_API_URL}/admin/backend-contract", headers=admin_headers()
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"backend contract returned HTTP {response.status}"
            )
        contract = await response.json()
    if not isinstance(contract, dict):
        raise RuntimeError("backend contract is not an object")
    model = contract.get("model") or {}
    runtime = contract.get("runtime") or {}
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise RuntimeError("backend contract model/runtime sections are invalid")
    raw_provenance = {
        "profile_id": contract.get("profile_id"),
        "profile_sha256": contract.get("profile_sha256"),
        "model_revision": model.get("revision"),
        "runtime_image_id": runtime.get("runtime_image_id"),
        "tensor_parallel_size": runtime.get("tensor_parallel_size"),
        "kv_cache_dtype": runtime.get("kv_cache_dtype"),
        "backend_serving_contract_sha256": contract.get(
            "backend_serving_contract_sha256"
        ),
    }
    try:
        provenance = project_stable_model_provenance(raw_provenance)
    except ValueError as exc:
        raise RuntimeError(
            f"unexpected active model provenance: {raw_provenance}"
        ) from exc
    if raw_provenance["backend_serving_contract_sha256"] not in (
        ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
    ):
        raise RuntimeError("backend contract endpoint omitted its stable digest")
    return provenance


async def health(request: web.Request) -> web.Response:
    healthy, attested = await model_status(request.app)
    status = 200 if healthy and attested else 503
    return web.json_response(
        {
            "status": "ok" if status == 200 else "starting",
            "model_api": healthy,
            "infrastructure_attested": attested,
            "controller_config_sha256": request.app["config_sha256"],
            "corpus_manifest_sha256": file_sha256(CORPUS_MANIFEST),
            "model_adapter": request.app["model_adapter"],
        },
        status=status,
    )


async def doctor(request: web.Request) -> web.Response:
    healthy, attested = await model_status(request.app)
    public_ready = public_profiles_ready(request.app["profile_sets"]["public"])
    real_ready = real_workload_ready(request.app["profile_sets"]["public"])
    checks = [
        (healthy, "fixed model API reachable"),
        (attested, "stable backend serving contract available"),
        (True, "candidate port 9000 reserved"),
        (public_ready, "public profiles smoke-api/public-12/public-pressure-192 installed"),
        (
            real_ready,
            "public coding profiles use six pinned instances with fresh independent repeated rollouts",
        ),
    ]
    rendered = "\n".join(f"[{'ok' if ok else 'not-ready'}] {label}" for ok, label in checks)
    return web.json_response({"ready": all(ok for ok, _ in checks), "phase": 1, "text": rendered})


async def candidate_healthy(app: web.Application) -> bool:
    url = f"{CANDIDATE_URL.removesuffix('/v1')}/health"
    try:
        async with app["client"].get(url) as response:
            await response.read()
            return 200 <= response.status < 300
    except (OSError, asyncio.TimeoutError):
        return False


async def candidate_port_open() -> bool:
    endpoint = CANDIDATE_URL.removesuffix("/v1")
    match = re.fullmatch(r"http://([^/:]+)(?::([0-9]+))?", endpoint)
    if match is None:
        raise RuntimeError("candidate URL is not a frozen HTTP host/port")
    host = match.group(1)
    port = int(match.group(2) or "80")
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2)
    except (OSError, asyncio.TimeoutError):
        return False
    del reader
    writer.close()
    await writer.wait_closed()
    return True


def release_public_direct_anchor_required_from_env() -> bool:
    """Read the release/author mode switch without truthy-string ambiguity."""

    value = os.environ.get(RELEASE_PUBLIC_DIRECT_ANCHOR_REQUIRED_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"{RELEASE_PUBLIC_DIRECT_ANCHOR_REQUIRED_ENV} must be exactly 0 or 1"
        )
    return value == "1"


def model_provenance_sha256(provenance: object) -> str | None:
    """Return a public-safe identity for one trusted model provenance record."""

    try:
        stable = project_stable_model_provenance(provenance)
    except ValueError:
        return None
    encoded = json.dumps(
        stable,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_release_public_direct_anchor(
    path_text: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Read one private release anchor through a race-resistant regular-file FD."""

    if not isinstance(path_text, str) or not path_text or not Path(path_text).is_absolute():
        raise ValueError("release public Direct anchor path must be absolute")
    if not _hex_text(expected_sha256, 64):
        raise ValueError("release public Direct anchor SHA-256 is missing or invalid")
    path = Path(path_text)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) & 0o077
        or not stat.S_IMODE(before.st_mode) & stat.S_IRUSR
        or before.st_size <= 0
        or before.st_size > RELEASE_PUBLIC_DIRECT_ANCHOR_MAX_BYTES
    ):
        raise ValueError("release public Direct anchor is not a private regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise ValueError("release public Direct anchor path identity changed")
        chunks: list[bytes] = []
        remaining = RELEASE_PUBLIC_DIRECT_ANCHOR_MAX_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) != opened.st_size
            or len(raw) > RELEASE_PUBLIC_DIRECT_ANCHOR_MAX_BYTES
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        ):
            raise ValueError("release public Direct anchor changed while reading")
    finally:
        os.close(descriptor)
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed_sha256, expected_sha256):
        raise ValueError("release public Direct anchor SHA-256 mismatch")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("release public Direct anchor contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("release public Direct anchor is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("release public Direct anchor is not a JSON object")
    return payload, observed_sha256


def validate_release_public_direct_anchor(
    app: web.Application,
    anchor: dict[str, Any],
    *,
    artifact_sha256: str,
    current_model_provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and project the author-created Direct anchor fail closed."""

    exact_keys = {
        "schema_version",
        "kind",
        "created_at_utc",
        "release_id",
        "policy_release_id",
        "suite",
        "profile",
        "workload_nonce",
        "workload_nonce_sha256",
        "baseline_run_id",
        "baseline",
        "source_report_sha256",
        "scoring_policy_sha256",
        "controller_config_sha256",
        "controller_image_id",
        "corpus_namespace",
        "corpus_fingerprint",
        "model_provenance",
        "model_provenance_sha256",
    }
    if set(anchor) != exact_keys:
        raise ValueError("release public Direct anchor schema fields are not exact")
    policy = app["policy"]
    created_at_utc = anchor.get("created_at_utc")
    try:
        created_at = datetime.fromisoformat(
            created_at_utc.removesuffix("Z") + "+00:00"
        )
        created_unix = created_at.timestamp()
    except (AttributeError, OverflowError, ValueError) as exc:
        raise ValueError("release public Direct anchor timestamp is invalid") from exc
    if (
        anchor.get("schema_version") != 1
        or anchor.get("kind") != RELEASE_PUBLIC_DIRECT_ANCHOR_KIND
        or not isinstance(created_at_utc, str)
        or not created_at_utc.endswith("Z")
        or len(created_at_utc) > 64
        or created_at.tzinfo is None
        or created_at.utcoffset() != timezone.utc.utcoffset(created_at)
        or not math.isfinite(created_unix)
        or policy.get("release_ready") is not True
        or policy.get("release_id") != RELEASE_PUBLIC_ID
        or anchor.get("release_id") != RELEASE_PUBLIC_ID
        or anchor.get("policy_release_id") != RELEASE_PUBLIC_ID
        or anchor.get("suite") != RELEASE_PUBLIC_SUITE
        or anchor.get("profile") != RELEASE_PUBLIC_PROFILE
        or not _hex_text(artifact_sha256, 64)
    ):
        raise ValueError("release public Direct anchor release binding is invalid")
    if (
        anchor.get("scoring_policy_sha256") != file_sha256(SCORING_POLICY)
        or anchor.get("controller_config_sha256") != app["config_sha256"]
        or not _hex_text(anchor.get("source_report_sha256"), 64)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(anchor.get("controller_image_id")))
        is None
    ):
        raise ValueError("release public Direct anchor source binding is invalid")

    workload_nonce = anchor.get("workload_nonce")
    if (
        not isinstance(workload_nonce, str)
        or re.fullmatch(r"[0-9a-f]{64}", workload_nonce) is None
    ):
        raise ValueError("release public Direct anchor private nonce is invalid")
    workload_nonce_sha256 = hashlib.sha256(workload_nonce.encode()).hexdigest()
    if anchor.get("workload_nonce_sha256") != workload_nonce_sha256:
        raise ValueError("release public Direct anchor nonce commitment is invalid")

    profiles: dict[str, Profile] = app["profile_sets"]["public"]
    profile = profiles.get(RELEASE_PUBLIC_PROFILE)
    if profile is None or not profile.enabled:
        raise ValueError("release public Direct anchor profile is unavailable")
    expected_namespace = f"{RELEASE_PUBLIC_SUITE}-{RELEASE_PUBLIC_PROFILE}"
    if anchor.get("corpus_namespace") != expected_namespace:
        raise ValueError("release public Direct anchor corpus namespace is invalid")
    # Reading the manifest digest makes absence or replacement of the mounted
    # corpus an explicit startup failure.  The recomputed fingerprint below is
    # the stronger binding because the namespace controls deterministic task
    # ordering as well as the pinned per-instance source/runtime digests.
    if not _hex_text(file_sha256(CORPUS_MANIFEST), 64):  # pragma: no cover - invariant
        raise ValueError("release public Direct anchor corpus manifest is invalid")
    expected_fingerprint = corpus_fingerprint(
        expected_namespace,
        profile.workflows,
        workload_family=profile.workload_family,
        corpus_partition=profile.corpus_partition,
        seed=profile.seed,
    )
    expected_model_adapter = copy.deepcopy(app["model_adapter"])
    expected_model_adapter["effective_max_tokens"] = profile.max_tokens
    if (
        not _hex_text(anchor.get("corpus_fingerprint"), 64)
        or anchor["corpus_fingerprint"] != expected_fingerprint
    ):
        raise ValueError("release public Direct anchor corpus fingerprint is stale")

    raw_anchor_provenance = anchor.get("model_provenance")
    raw_anchor_provenance_digest = (
        hashlib.sha256(
            json.dumps(
                raw_anchor_provenance,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if isinstance(raw_anchor_provenance, dict) and raw_anchor_provenance
        else None
    )
    try:
        anchor_provenance = project_stable_model_provenance(
            raw_anchor_provenance
        )
    except ValueError as exc:
        raise ValueError(
            "release public Direct anchor model provenance is invalid"
        ) from exc
    provenance_digest = model_provenance_sha256(current_model_provenance)
    if (
        not isinstance(current_model_provenance, dict)
        or current_model_provenance.get("backend_serving_contract_sha256")
        not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        or not same_backend_serving_contract(
            anchor_provenance, current_model_provenance
        )
        or anchor.get("model_provenance_sha256")
        not in {
            raw_anchor_provenance_digest,
            model_provenance_sha256(anchor_provenance),
        }
    ):
        raise ValueError(
            "release public Direct anchor backend serving contract is stale"
        )

    baseline = anchor.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("release public Direct anchor baseline is missing")
    try:
        baseline_provenance = project_stable_model_provenance(
            baseline.get("model_provenance")
        )
    except ValueError as exc:
        raise ValueError(
            "release public Direct anchor baseline provenance is invalid"
        ) from exc
    if (
        baseline.get("schema_version") != 1
        or baseline.get("run_id") != anchor.get("baseline_run_id")
        or not isinstance(baseline.get("run_id"), str)
        or not baseline["run_id"]
        or baseline.get("mode") != "baseline"
        or baseline.get("profile") != RELEASE_PUBLIC_PROFILE
        or baseline.get("release_id") != RELEASE_PUBLIC_ID
        or baseline.get("controller_config_sha256") != app["config_sha256"]
        or baseline.get("corpus_namespace") != expected_namespace
        or baseline.get("corpus_fingerprint") != expected_fingerprint
        or baseline.get("workload_nonce_sha256") != workload_nonce_sha256
        or not same_backend_serving_contract(
            baseline_provenance, current_model_provenance
        )
        or baseline.get("model_adapter") != expected_model_adapter
        or baseline.get("profile_config") != asdict(profile)
    ):
        raise ValueError("release public Direct anchor baseline identity is invalid")
    if baseline.get("infrastructure_failure") is not None or baseline.get("candidate_failure") is not None:
        raise ValueError("release public Direct anchor baseline contains a failure")

    backend_audit = baseline.get("backend_audit")
    warmup_audit = baseline.get("warmup_audit")
    for label, audit in (("backend", backend_audit), ("warmup", warmup_audit)):
        if (
            not isinstance(audit, dict)
            or audit.get("passed") is not True
            or audit.get("mismatches") != []
            or audit.get("extra_backend_requests") != 0
            or audit.get("missing_backend_requests") != 0
            or isinstance(audit.get("client_requests"), bool)
            or not isinstance(audit.get("client_requests"), int)
            or audit["client_requests"] <= 0
            or audit.get("audited_backend_requests") != audit["client_requests"]
        ):
            raise ValueError(f"release public Direct anchor {label} audit is invalid")
    pressure = baseline.get("pressure_validation")
    if (
        not isinstance(pressure, dict)
        or pressure.get("qualification_schema_version") != 2
        or pressure.get("gate_basis")
        != "sampling_completeness_and_high_kv_occupancy"
        or pressure.get("required") is not True
        or pressure.get("valid") is not True
        or pressure.get("reasons") != []
    ):
        raise ValueError("release public Direct anchor pressure gate is invalid")

    def finite_number(value: object) -> float | None:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
        return None

    window = baseline.get("measurement_window")
    elapsed = finite_number(baseline.get("elapsed_sec"))
    valid_steps = baseline.get("valid_steps")
    total_executed_steps = baseline.get("total_executed_steps")
    rate = finite_number(baseline.get("valid_steps_per_min"))
    request_success = finite_number(baseline.get("request_success_rate"))
    valid_step_ratio = finite_number(baseline.get("valid_step_ratio"))
    backend_work = finite_number(baseline.get("backend_tokens_per_valid_step"))
    started_unix = finite_number(baseline.get("started_unix"))
    ended_unix = finite_number(baseline.get("ended_unix"))
    wall_ended_unix = finite_number(baseline.get("wall_ended_unix"))
    wall_elapsed = finite_number(baseline.get("wall_elapsed_sec"))
    zero_progress = (
        finite_number(window.get("zero_progress_fraction"))
        if isinstance(window, dict)
        else None
    )
    if (
        not isinstance(window, dict)
        or window.get("configured") is not True
        or window.get("duration_sec") != profile.measurement_window_sec
        or window.get("warmup_duration_sec") != profile.measurement_warmup_sec
        or window.get("deadline_reached") is not True
        or window.get("termination") != "trusted_fixed_window"
        or window.get("control_passed") is not True
        or window.get("audit_passed") is not True
        or window.get("backend_quiescent_after_cutoff") is not True
        or window.get("passed") is not True
        or window.get("errors") != []
        or "agent_execution_interval" in baseline
        or elapsed != float(profile.measurement_window_sec or 0)
        or started_unix is None
        or ended_unix is None
        or wall_ended_unix is None
        or wall_elapsed is None
        or started_unix < 0.0
        or not math.isclose(
            ended_unix - started_unix, elapsed, rel_tol=1e-9, abs_tol=1e-6
        )
        or wall_ended_unix < ended_unix
        or not math.isclose(
            wall_ended_unix - started_unix,
            wall_elapsed,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        or wall_elapsed < elapsed
        or created_unix < wall_ended_unix
        or finite_number(window.get("measurement_elapsed_sec"))
        != float(profile.measurement_window_sec or 0)
        or not isinstance(valid_steps, int)
        or isinstance(valid_steps, bool)
        or valid_steps <= 0
        or not isinstance(total_executed_steps, int)
        or isinstance(total_executed_steps, bool)
        or total_executed_steps < valid_steps
        or not isinstance(window.get("total_executed_steps"), int)
        or isinstance(window.get("total_executed_steps"), bool)
        or window.get("total_executed_steps") != total_executed_steps
        or not isinstance(window.get("post_cutoff_steps"), int)
        or isinstance(window.get("post_cutoff_steps"), bool)
        or window.get("post_cutoff_steps") != total_executed_steps - valid_steps
        # Every mini-SWE workflow executes shell actions serially.  Once the
        # trusted ledger closes, at most its one already-running action can
        # finish after the cutoff; a following model request is rejected.
        or window.get("post_cutoff_steps") > profile.workflows
        or rate is None
        or not math.isclose(rate, valid_steps * 60.0 / elapsed, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("release public Direct anchor fixed-window contract is invalid")
    for key in (
        "request_count",
        "measured_request_count",
        "minimum_workflow_progress",
        "tests_passed_workflows",
        "completed_workflows",
        "zero_progress_workflows",
    ):
        value = baseline.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("release public Direct anchor quality counts are invalid")
    if baseline["request_count"] <= 0 or baseline["measured_request_count"] <= 0:
        raise ValueError("release public Direct anchor has no measured requests")
    workflows = profile.workflows
    gateway_wait = finite_number(backend_audit.get("p99_gateway_wait_sec"))
    if (
        request_success is None
        or request_success < float(policy["min_request_success_rate"])
        or valid_step_ratio is None
        or valid_step_ratio < float(policy["min_valid_step_ratio"])
        or zero_progress is None
        or not 0.0 <= zero_progress <= 0.05
        or not math.isclose(
            zero_progress,
            baseline["zero_progress_workflows"] / workflows,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or rate is None
        or rate < float(policy["min_baseline_valid_steps_per_min"])
        or backend_work is None
        or backend_work <= 0.0
        or gateway_wait is None
        or gateway_wait < 0.0
        or gateway_wait > float(policy["max_p99_gateway_wait_sec"])
    ):
        raise ValueError("release public Direct anchor baseline quality is invalid")

    stable_baseline = copy.deepcopy(baseline)
    # Old anchors retain their byte-level legacy provenance for artifact
    # verification, but all live pair state uses the current stable projection.
    stable_baseline["model_provenance"] = copy.deepcopy(current_model_provenance)
    summary = public_summary(stable_baseline)
    if (
        summary.get("mode") != "baseline"
        or summary.get("profile") != RELEASE_PUBLIC_PROFILE
        or summary.get("corpus_namespace") != expected_namespace
        or summary.get("corpus_fingerprint") != expected_fingerprint
        or summary.get("workload_nonce_sha256") != workload_nonce_sha256
        or summary.get("release_id") != RELEASE_PUBLIC_ID
        or summary.get("controller_config_sha256") != app["config_sha256"]
        or summary.get("model_provenance_sha256") != provenance_digest
        or "workload_nonce" in summary
        or "model_provenance" in summary
    ):
        raise ValueError("release public Direct anchor public summary is invalid")
    return summary, copy.deepcopy(current_model_provenance)


def release_public_anchor_material(app: web.Application) -> dict[str, Any] | None:
    """Return a fresh copy so no candidate attempt can mutate the anchor."""

    encoded = app.get("release_public_anchor_material_json")
    if encoded is None:
        return None
    if not isinstance(encoded, bytes):
        raise RuntimeError("release public Direct anchor state is corrupted")
    material = json.loads(encoded.decode("utf-8"))
    if not isinstance(material, dict):  # pragma: no cover - internal invariant
        raise RuntimeError("release public Direct anchor state is malformed")
    return material


def persist_release_public_anchor_feedback(summary: dict[str, Any]) -> None:
    """Create feedback once, or verify a byte-identical restart artifact."""

    run_dir = FEEDBACK_ROOT / summary["run_id"]
    if not run_dir.exists():
        persist_feedback(summary)
        return
    metadata = run_dir.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or run_dir.is_symlink():
        raise RuntimeError("release public Direct anchor feedback path is invalid")
    manifest = {
        "schema_version": 1,
        "run_id": summary["run_id"],
        "profile": summary["profile"],
        "mode": summary["mode"],
        "profile_config": summary["profile_config"],
        "corpus_fingerprint": summary["corpus_fingerprint"],
        "runner": "mini-swe-agent-1.14.4-pinned-core",
        "model": "target-model",
        "model_adapter": summary.get("model_adapter"),
    }
    expected = {
        "summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "summary.txt": format_run(summary) + "\n",
        "run_manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }
    if set(path.name for path in run_dir.iterdir()) != set(expected):
        raise RuntimeError("release public Direct anchor feedback files changed")
    for name, contents in expected.items():
        path = run_dir / name
        item = path.lstat()
        if not stat.S_ISREG(item.st_mode) or path.is_symlink() or path.read_text() != contents:
            raise RuntimeError("release public Direct anchor feedback content changed")


async def load_release_public_direct_anchor(app: web.Application) -> None:
    """Load the private release anchor, or retain explicit author A+B mode."""

    path_text = app.get("release_public_anchor_file")
    expected_sha256 = app.get("release_public_anchor_expected_sha256")
    required = app.get("release_public_anchor_required") is True
    if required and not path_text:
        raise RuntimeError("required release public Direct anchor file is missing")
    if path_text and not _hex_text(expected_sha256, 64):
        raise RuntimeError("configured release public Direct anchor has no SHA-256")
    if not path_text:
        if expected_sha256:
            raise RuntimeError("release public Direct anchor SHA-256 has no configured file")
        return
    # Explicit author calibration/local-verifier mode keeps live A+B behavior
    # even if a host accidentally leaves a valid anchor mount configured.
    if not required:
        return
    if app["policy"].get("release_ready") is not True:
        raise RuntimeError(
            "required release public Direct anchor needs release_ready=true"
        )
    anchor, artifact_sha256 = read_release_public_direct_anchor(
        str(path_text), str(expected_sha256)
    )
    current_provenance = await model_provenance(app)
    summary, trusted_provenance = validate_release_public_direct_anchor(
        app,
        anchor,
        artifact_sha256=artifact_sha256,
        current_model_provenance=current_provenance,
    )
    pair_id = f"{RELEASE_PUBLIC_PROFILE}-anchor-{artifact_sha256[:16]}"
    created_unix = float(summary["ended_unix"])
    pair = {
        "pair_id": pair_id,
        "baseline_anchor_pair_id": pair_id,
        "parent_pair_id": None,
        "profile": RELEASE_PUBLIC_PROFILE,
        "status": "baseline_complete",
        "created_unix": created_unix,
        "updated_unix": created_unix,
        "controller_config_sha256": app["config_sha256"],
        "workload_nonce": anchor["workload_nonce"],
        "baseline": summary,
        "candidate": None,
        "baseline_model_provenance_sha256": summary[
            "model_provenance_sha256"
        ],
        "candidate_model_provenance_sha256": None,
        "failure": None,
        "comparison": None,
    }
    material = {
        "schema_version": 1,
        "artifact_sha256": artifact_sha256,
        "corpus_namespace": anchor["corpus_namespace"],
        "model_provenance": trusted_provenance,
        "pair": pair,
    }
    app["release_public_anchor_material_json"] = json.dumps(
        material, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    app["release_public_anchor_pair_id"] = pair_id
    app["public_pair"][RELEASE_PUBLIC_PROFILE] = copy.deepcopy(pair)
    persist_public_pair_state(pair)
    persist_release_public_anchor_feedback(summary)
    app["history"].append(copy.deepcopy(summary))


async def current_release_public_anchor_material(
    app: web.Application,
) -> tuple[dict[str, Any], dict[str, Any]]:
    material = release_public_anchor_material(app)
    if material is None:
        raise RuntimeError("release public Direct anchor is unavailable")
    current_provenance = await model_provenance(app)
    if not same_backend_serving_contract(
        current_provenance, material.get("model_provenance")
    ):
        raise RuntimeError(
            "release public Direct anchor backend serving contract changed"
        )
    return material, current_provenance


def public_error_response(
    error: str,
    *,
    pair: dict[str, Any] | None = None,
    gates: list[str] | tuple[str, ...] = (),
) -> web.Response:
    """Render one candidate-visible error through a closed response schema."""

    spec = PUBLIC_ERROR_SPECS.get(error)
    if spec is None:
        raise ValueError("public error code is not allowlisted")
    status, message = spec
    payload: dict[str, Any] = {
        "schema_version": 1,
        "error": error,
        "message": message,
    }
    if pair is not None:
        for key in ("pair_id", "baseline_anchor_pair_id", "parent_pair_id"):
            value = pair.get(key)
            if value is None:
                continue
            if isinstance(value, str) and re.fullmatch(r"[a-z0-9-]+", value):
                payload[key] = value
        pair_status = pair.get("status")
        if pair_status in PUBLIC_PAIR_STATUSES:
            payload["pair_status"] = pair_status
    if gates:
        sanitized: list[str] = []
        for gate in gates:
            safe_gate = gate if gate in PUBLIC_PAIR_GATE_CODES else "unclassified_pair_gate"
            if safe_gate not in sanitized:
                sanitized.append(safe_gate)
        payload["gates"] = sanitized
    return web.json_response(payload, status=status)


def persist_public_failure_artifact(
    *,
    error: str,
    raw_detail: str,
    pair: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Retain bounded raw diagnostics in eight root-only atomic ring slots."""

    def bounded_json(value: object, limit: int) -> str | None:
        if value is None:
            return None
        try:
            rendered = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=lambda item: f"<{type(item).__name__}>",
            )
        except Exception:
            rendered = f"<{type(value).__name__}>"
        return rendered[:limit]

    payload = {
        "schema_version": 1,
        "kind": "kvbench-public-private-failure",
        "created_unix": time.time(),
        "error": error if error in PUBLIC_ERROR_SPECS else "public_internal_error",
        "raw_detail": str(raw_detail)[:2500],
        "pair_json": bounded_json(pair, 450),
        "context_json": bounded_json(context, 450),
    }
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if len(rendered) > PUBLIC_FAILURE_FILE_MAX_BYTES:
        # The fixed component is tiny; retain the beginning of the diagnostic
        # and remove optional context before failing closed on the hard cap.
        payload["pair_json"] = None
        payload["context_json"] = None
        payload["raw_detail"] = payload["raw_detail"][:1800]
        rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if len(rendered) > PUBLIC_FAILURE_FILE_MAX_BYTES:  # pragma: no cover - invariant
        raise RuntimeError("bounded public failure artifact exceeds its hard cap")

    global _PUBLIC_FAILURE_SATURATED
    with _PUBLIC_FAILURE_LOCK:
        if _PUBLIC_FAILURE_SATURATED:
            return
        TRUSTED_RESULTS_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        TRUSTED_RESULTS_ROOT.chmod(0o700)
        slots = [
            TRUSTED_RESULTS_ROOT / f"public-failure-slot-{index:02d}.json"
            for index in range(PUBLIC_FAILURE_SLOT_COUNT)
        ]
        slot_set = set(slots)
        # Migrate away from the old random-name scheme so restarts cannot leave
        # an unbounded tail of private exception artifacts.
        for old in TRUSTED_RESULTS_ROOT.glob("public-failure-*.json"):
            if old not in slot_set and old.is_file() and not old.is_symlink():
                old.unlink()
        existing = [path for path in slots if path.is_file() and not path.is_symlink()]
        existing_bytes = sum(path.stat().st_size for path in existing)
        if (
            len(existing) >= PUBLIC_FAILURE_SLOT_COUNT
            or existing_bytes + len(rendered) > PUBLIC_FAILURE_TOTAL_MAX_BYTES
        ):
            _PUBLIC_FAILURE_SATURATED = True
            return
        destination = next(path for path in slots if path not in existing)
        temporary = TRUSTED_RESULTS_ROOT / (
            f".{destination.name}.{secrets.token_hex(8)}.tmp"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(destination)
            destination.chmod(0o600)
            if (
                len(existing) + 1 >= PUBLIC_FAILURE_SLOT_COUNT
                or existing_bytes + len(rendered) >= PUBLIC_FAILURE_TOTAL_MAX_BYTES
            ):
                _PUBLIC_FAILURE_SATURATED = True
        finally:
            temporary.unlink(missing_ok=True)


@web.middleware
async def sanitize_public_exceptions(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """Prevent an uncaught /public exception from becoming a reflection API."""

    try:
        return await handler(request)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if not str(request.path).startswith("/public/"):
            raise
        try:
            persist_public_failure_artifact(
                error="public_internal_error",
                raw_detail=f"{type(exc).__name__}: {exc}",
                context={"method": request.method, "path": request.path},
            )
        except Exception:
            # The only safe fallback when even private evidence persistence is
            # unavailable is the same fixed public error, never the new error.
            pass
        return public_error_response("public_internal_error")


def require_single_model_provenance(
    legs: list[dict[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    """Require one stable backend serving contract across all paired legs."""

    if not legs:
        raise RuntimeError(f"{context} legs have no model provenance")
    try:
        projected = [
            project_stable_model_provenance(leg.get("model_provenance"))
            for leg in legs
        ]
    except ValueError as exc:
        raise RuntimeError(f"{context} model provenance is invalid") from exc
    contracts = {
        value["backend_serving_contract_sha256"] for value in projected
    }
    if len(contracts) != 1 or not contracts <= ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256:
        raise RuntimeError(f"{context} legs used different backend contracts")
    return projected[0]


def public_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Project one trusted leg into an explicit candidate-visible schema.

    This is intentionally a positive allowlist at every object boundary.  A
    newly added trusted-result field (including a nested field) must be
    reviewed here before it can enter public feedback.
    """

    scalar = object()
    scalar_list = object()

    def safe_scalar(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return None

    def project(value: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        projected: dict[str, Any] = {}
        for key, child_schema in schema.items():
            if key not in value:
                continue
            child = value[key]
            if child_schema is scalar:
                projected[key] = safe_scalar(child)
            elif child_schema is scalar_list:
                projected[key] = (
                    [
                        safe
                        for item in child
                        if (safe := safe_scalar(item)) is not None
                    ]
                    if isinstance(child, list)
                    else []
                )
            elif isinstance(child_schema, dict):
                projected[key] = (
                    None if child is None else project(child, child_schema)
                )
        return projected

    profile_schema = {
        key: scalar
        for key in (
            "name",
            "enabled",
            "workflows",
            "concurrency",
            "max_steps",
            "context_bytes",
            "max_tokens",
            "configured_max_tokens",
            "workflow_timeout_sec",
            "request_timeout_sec",
            "tool_delay_min_sec",
            "tool_delay_max_sec",
            "seed",
            "warmup_requests",
            "warmup_context_bytes",
            "warmup_steps",
            "workload_family",
            "metrics_sample_interval_sec",
            "metrics_request_timeout_sec",
            "pressure_required",
            "pressure_kv_threshold",
            "pressure_kv_p95_min",
            "pressure_time_fraction_min",
            "pressure_min_samples",
            "corpus_partition",
            "corpus_unique_instances_min",
            "tool_timeout_sec",
            "sandbox_prepare_concurrency",
            "backend_stall_timeout_sec",
            "candidate_admission_stall_timeout_sec",
            "measurement_window_sec",
            "measurement_warmup_sec",
        )
    }
    audit_schema = {
        key: scalar
        for key in (
            "passed",
            "window_censored",
            "client_requests",
            "audited_backend_requests",
            "strict_client_requests",
            "cutoff_client_requests",
            "cutoff_backend_admitted",
            "cutoff_backend_unadmitted",
            "extra_backend_requests",
            "missing_backend_requests",
            "p95_gateway_wait_sec",
            "p99_gateway_wait_sec",
            "max_gateway_wait_sec",
        )
    }
    distribution_schema = {
        key: scalar for key in ("sample_count", "total", "p50", "p95", "p99", "max")
    }
    measurement_schema = {
        key: scalar
        for key in (
            "schema_version",
            "configured",
            "duration_sec",
            "warmup_duration_sec",
            "workload_started_unix",
            "started_unix",
            "cutoff_unix",
            "triggered_unix",
            "sealed_unix",
            "trigger_lag_sec",
            "backend_seal_lag_sec",
            "deadline_reached",
            "termination",
            "control_passed",
            "backend_cancelled_requests",
            "measurement_elapsed_sec",
            "wall_elapsed_sec",
            "teardown_sec",
            "total_executed_steps",
            "pre_measurement_steps",
            "post_cutoff_steps",
            "semantic_evaluated_workflows",
            "truncated_workflows",
            "zero_progress_workflows",
            "zero_progress_fraction",
            "audit_passed",
            "backend_quiescent_after_cutoff",
            "passed",
        )
    }
    interval_schema = {
        key: scalar
        for key in (
            "schema_version",
            "start_event",
            "end_event",
            "duration_clock",
            "started_unix",
            "ended_unix",
            "elapsed_sec",
            "expected_agent_runs",
            "observed_agent_runs",
            "complete",
        )
    }
    failure_schema = {
        key: scalar for key in ("type", "cancelled")
    }
    public_schema: dict[str, Any] = {
        key: scalar
        for key in (
            "schema_version",
            "run_id",
            "mode",
            "profile",
            "corpus_namespace",
            "corpus_fingerprint",
            "workload_nonce_sha256",
            "workload_started_unix",
            "started_unix",
            "ended_unix",
            "wall_ended_unix",
            "sandbox_preparation_sec",
            "elapsed_sec",
            "wall_elapsed_sec",
            "candidate_leg_wall_timeout_sec",
            "valid_steps",
            "total_executed_steps",
            "valid_steps_per_min",
            "completed_workflows",
            "completed_workflows_per_min",
            "tests_passed_workflows",
            "tests_passed_workflows_per_min",
            "request_count",
            "measured_request_count",
            "request_success_rate",
            "valid_step_ratio",
            "minimum_workflow_progress",
            "zero_progress_workflows",
            "p50_step_latency_sec",
            "p95_step_latency_sec",
            "p99_step_latency_sec",
            "p95_query_latency_sec",
            "p99_query_latency_sec",
            "p50_tool_latency_sec",
            "p95_tool_latency_sec",
            "p99_tool_latency_sec",
            "aggregate_model_query_sec",
            "aggregate_tool_sec",
            "model_query_time_fraction",
            "p50_ttft_sec",
            "p95_ttft_sec",
            "p99_ttft_sec",
            "prompt_tokens",
            "completion_tokens",
            "backend_tokens_per_valid_step",
            "release_id",
            "controller_config_sha256",
        )
    }
    public_schema.update(
        {
            "profile_config": profile_schema,
            "model_adapter": {
                key: scalar
                for key in (
                    "mode",
                    "instruction_role",
                    "instruction_sha256",
                    "instance_template_sha256",
                    "format_error_sha256",
                    "prompt_contract",
                    "task_input",
                    "offline_constraint",
                    "temperature",
                    "top_p",
                    "max_tokens_policy",
                    "response_protocol",
                    "function_tools",
                    "reasoning_override",
                    "effective_max_tokens",
                )
            }
            | {"chat_template_kwargs": {"enable_thinking": scalar}},
            "sandbox_preparation": {
                "live_sandbox_limit": scalar,
                "initial": {
                    key: scalar
                    for key in (
                        "count",
                        "wall_sec",
                        "configured_create_concurrency",
                        "included_in_measured_interval",
                    )
                },
                "queued": {
                    key: scalar
                    for key in (
                        "count",
                        "attempts",
                        "prepared",
                        "failures",
                        "configured_create_concurrency",
                        "maximum_observed_create_concurrency",
                        "queue_wait_cumulative_sec",
                        "queue_wait_p95_sec",
                        "create_cumulative_sec",
                        "create_active_wall_sec",
                        "create_p50_sec",
                        "create_p95_sec",
                        "create_max_sec",
                        "included_in_measured_interval",
                    )
                },
            },
            "measurement_window": measurement_schema,
            "agent_execution_interval": interval_schema,
            "program_lifecycle": {
                "enabled": scalar,
                "warmup": {
                    key: scalar
                    for key in (
                        "program_release_attempted",
                        "program_release_acknowledged",
                        "program_release_failed",
                    )
                },
                "measured_attempted": scalar,
                "measured_acknowledged": scalar,
                "measured_unsupported": scalar,
                "measured_failed": scalar,
            },
            "request_token_distribution": {
                key: distribution_schema
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "agent_depth": {
                key: scalar
                for key in (
                    "sample_count",
                    "p25_model_calls",
                    "p50_model_calls",
                    "p95_model_calls",
                    "max_model_calls",
                    "target_max_steps",
                )
            },
            "model_response_outcomes": {
                key: scalar
                for key in (
                    "response_count",
                    "responses_with_tool_call",
                    "responses_without_tool_call",
                    "completion_budget_exhausted",
                    "no_tool_and_budget_exhausted",
                    "content_chars_total",
                    "reasoning_content_chars_total",
                    "internal_reasoning_chars_total",
                )
            },
            "warmup_audit": audit_schema,
            "warmup_protocol": {
                key: scalar
                for key in (
                    "request_count",
                    "required_valid_actions",
                    "valid_actions",
                    "format_corrections",
                    "program_release_attempted",
                    "program_release_acknowledged",
                    "program_release_failed",
                    "maximum_attempts_per_warmup",
                )
            },
            "backend_audit": audit_schema,
            "pressure_validation": {
                "qualification_schema_version": scalar,
                "gate_basis": scalar,
                "required": scalar,
                "valid": scalar,
                "status": scalar,
                "minimum_samples": scalar,
                "minimum_kv_p95": scalar,
                "minimum_time_fraction": scalar,
                "reasons": scalar_list,
            },
            "infrastructure_failure": failure_schema,
            "candidate_failure": failure_schema,
        }
    )
    allowed = project(result, public_schema)
    # Normalize all abnormal sub-objects after the recursive projection.  A
    # future runner failure type or pressure string becomes one fixed sentinel
    # rather than an accidental candidate-visible detail channel.
    for failure_key in ("infrastructure_failure", "candidate_failure"):
        raw_failure = result.get(failure_key)
        if raw_failure is None:
            allowed[failure_key] = None
            continue
        failure_type = (
            raw_failure.get("type") if isinstance(raw_failure, dict) else None
        )
        allowed[failure_key] = {
            "type": (
                failure_type
                if failure_type in PUBLIC_FAILURE_TYPES
                else "workload_watchdog_failure"
            ),
            "cancelled": (
                raw_failure.get("cancelled") is True
                if isinstance(raw_failure, dict)
                else False
            ),
        }
    pressure = allowed.get("pressure_validation")
    raw_pressure = result.get("pressure_validation")
    if isinstance(pressure, dict) and isinstance(raw_pressure, dict):
        public_reasons: list[str] = []
        for raw_reason in raw_pressure.get("reasons", []):
            prefix = str(raw_reason).partition(":")[0]
            code = (
                prefix
                if prefix in PUBLIC_PRESSURE_REASON_CODES
                else "pressure_evidence_invalid"
            )
            if code not in public_reasons:
                public_reasons.append(code)
        pressure["reasons"] = public_reasons
    provenance = result.get("model_provenance")
    # Some CPU isolation tests compile this pure helper from the AST without
    # importing the aiohttp server module. Keep that narrow use fail-closed
    # while production always resolves the module-level digest helper above.
    digest_helper = globals().get("model_provenance_sha256")
    allowed["model_provenance_sha256"] = (
        digest_helper(provenance) if callable(digest_helper) else None
    )
    # Public reports retain useful aggregate vLLM diagnostics but never expose
    # raw series labels, unknown future aggregates, or the hidden policy.
    raw_metrics = result.get("vllm_metrics")
    raw_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    metrics: dict[str, Any] = {}
    live_samples = raw_metrics.get("live_samples")

    def rows(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []
    # Finite public comparisons validate that the throughput denominator ends
    # with the final agent.run termination and that no metrics sample entered
    # after that boundary.  Preserve timestamps only; raw metric names, labels,
    # and values remain controller-private.  Fixed-window public reports do not
    # carry an agent_execution_interval and retain the previous compact shape.
    if "agent_execution_interval" in result:
        metrics["live_samples"] = [
            {"created_unix": safe_scalar(sample.get("created_unix"))}
            for sample in rows(live_samples)
            if isinstance(sample, dict)
        ]
    counter_schema = {"name": scalar, "delta": scalar, "value": scalar}
    gauge_schema = {"name": scalar, "value": scalar}
    metrics["counter_deltas"] = []
    for sample in rows(raw_metrics.get("counter_deltas")):
        clean = project(sample, counter_schema)
        name = clean.get("name")
        if isinstance(name, str) and any(
            term in name for term in ("preempt", "prefix_cache", "request", "token")
        ):
            metrics["counter_deltas"].append(clean)
    metrics["gauge_end"] = []
    for sample in rows(raw_metrics.get("gauge_end")):
        clean = project(sample, gauge_schema)
        name = clean.get("name")
        if isinstance(name, str) and any(
            term in name for term in ("cache_usage", "requests_running", "requests_waiting")
        ):
            metrics["gauge_end"].append(clean)
    metrics["derived"] = project(
        raw_metrics.get("derived"),
        {
            key: scalar
            for key in (
                "prefix_cache_hits",
                "prefix_cache_queries",
                "prefix_cache_hit_ratio",
                "computed_prompt_tokens",
                "estimated_kv_write_tokens",
                "preemptions",
                "prompt_tokens",
                "generation_tokens",
            )
        },
    )
    metrics["live_series"] = project(
        raw_metrics.get("live_series"),
        {
            key: scalar
            for key in (
                "sample_count",
                "kv_sample_count",
                "kv_cache_usage_p50",
                "kv_cache_usage_p95",
                "kv_cache_usage_max",
                "retained_kv_cache_usage_p50",
                "retained_kv_cache_usage_p95",
                "retained_kv_cache_usage_max",
                "pressure_threshold",
                "expected_sample_interval_sec",
                "sampling_gap_limit_sec",
                "sample_interval_sec_p95",
                "sample_interval_sec_max",
                "sampling_gap_count",
                "sampling_gap_max_sec",
                "sampling_gap_total_sec",
                "sampling_timestamp_error_count",
                "sample_fraction_at_or_above_pressure_threshold",
                "sampler_span_sec",
                "sampler_duration_sec",
                "observed_metric_duration_sec",
                "backend_active_duration_sec",
                "backend_active_time_fraction",
                "pressurized_duration_sec",
                "fraction_at_or_above_pressure_threshold",
                "waiting_duration_sec",
                "fraction_backend_active_time_with_waiting",
                "kv_token_capacity",
                "kv_token_capacity_unit",
                "kv_token_capacity_source",
                "kv_token_capacity_sample_count",
                "kv_token_capacity_inconsistent",
                "kv_token_capacity_incomplete_active_sample_count",
                "offered_token_demand_sample_count",
                "offered_token_demand_incomplete_sample_count",
                "offered_request_count_p50",
                "offered_request_count_p95",
                "offered_request_count_max",
                "offered_prompt_tokens_p50",
                "offered_prompt_tokens_p95",
                "offered_prompt_tokens_max",
                "offered_total_tokens_p50",
                "offered_total_tokens_p95",
                "offered_total_tokens_max",
                "offered_prompt_tokens_to_capacity_p50",
                "offered_prompt_tokens_to_capacity_p95",
                "offered_prompt_tokens_to_capacity_max",
                "high_kv_sample_count",
                "active_working_set_pressure_sample_count",
                "active_working_set_pressure_duration_sec",
                "fraction_backend_active_time_with_active_working_set_pressure",
                "oversubscribed_working_set_sample_count",
                "oversubscribed_working_set_duration_sec",
                "fraction_backend_active_time_with_oversubscribed_working_set",
                "oversubscribed_working_set_waiting_sample_count",
                "oversubscribed_working_set_waiting_duration_sec",
                "fraction_backend_active_time_with_oversubscribed_working_set_waiting",
                "pressurized_waiting_sample_count",
                "pressurized_waiting_duration_sec",
                "fraction_backend_active_time_pressurized_with_waiting",
                "requests_running_p50",
                "requests_running_p95",
                "requests_running_max",
                "requests_waiting_p50",
                "requests_waiting_p95",
                "requests_waiting_max",
                "sampler_error_count",
            )
        },
    )
    allowed["vllm_metrics"] = metrics
    return allowed


def format_run(summary: dict[str, Any]) -> str:
    audit = summary["backend_audit"]
    live = summary.get("vllm_metrics", {}).get("live_series", {})
    derived = summary.get("vllm_metrics", {}).get("derived", {})
    pressure = summary.get("pressure_validation", {})
    prompt_distribution = summary.get("request_token_distribution", {}).get("prompt_tokens", {})
    infrastructure_failure = summary.get("infrastructure_failure")
    candidate_failure = summary.get("candidate_failure")

    def number(value: object, digits: int = 4) -> str:
        return "unavailable" if not isinstance(value, (int, float)) else f"{float(value):.{digits}f}"

    return "\n".join(
        [
            f"run_id: {summary['run_id']}",
            f"mode: {summary['mode']}",
            f"profile: {summary['profile']}",
            f"elapsed: {summary['elapsed_sec']:.1f} s",
            f"wall elapsed/teardown: {summary.get('wall_elapsed_sec', summary['elapsed_sec']):.1f}/"
            f"{float(summary.get('measurement_window', {}).get('teardown_sec') or 0.0):.1f} s",
            "measurement window: "
            + (
                f"{summary.get('measurement_window', {}).get('duration_sec')} s, "
                f"deadline={summary.get('measurement_window', {}).get('deadline_reached')}, "
                f"censored={summary.get('backend_audit', {}).get('cutoff_client_requests', 0)}"
                if summary.get("measurement_window", {}).get("configured")
                else "finite workload"
            ),
            f"valid steps/min: {summary['valid_steps_per_min']:.3f}",
            f"test-passing workflows/min: {summary['tests_passed_workflows_per_min']:.3f}",
            f"completed workflows/min: {summary['completed_workflows_per_min']:.3f}",
            f"request success: {summary['request_success_rate']:.3%}",
            f"tool latency p95: {number(summary.get('p95_tool_latency_sec'))} s",
            f"model-query time fraction: {number(summary.get('model_query_time_fraction'))}",
            f"workload family: {summary['profile_config'].get('workload_family', 'unknown')}",
            f"pressure validity: {pressure.get('status', 'unavailable')}",
            f"KV usage p95/max: {number(live.get('kv_cache_usage_p95'))}/"
            f"{number(live.get('kv_cache_usage_max'))}",
            f"time at KV >= {number(live.get('pressure_threshold'))}: "
            f"{number(live.get('fraction_at_or_above_pressure_threshold'))}",
            f"running/waiting max: {number(live.get('requests_running_max'), 1)}/"
            f"{number(live.get('requests_waiting_max'), 1)}",
            f"prefix-cache hit ratio: {number(derived.get('prefix_cache_hit_ratio'))}",
            f"prompt tokens p95/max: {number(prompt_distribution.get('p95'), 1)}/"
            f"{number(prompt_distribution.get('max'), 1)}",
            "infrastructure failure: "
            + ("none" if infrastructure_failure is None else str(infrastructure_failure.get("type"))),
            "candidate failure: "
            + ("none" if candidate_failure is None else str(candidate_failure.get("type"))),
            f"backend calls/client calls: {audit['audited_backend_requests']}/{audit['client_requests']}",
            f"one-to-one audit: {'pass' if audit['passed'] else 'FAIL'}",
        ]
    )


def persist_feedback(summary: dict[str, Any]) -> None:
    run_dir = FEEDBACK_ROOT / summary["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "summary.txt").write_text(format_run(summary) + "\n")
    manifest = {
        "schema_version": 1,
        "run_id": summary["run_id"],
        "profile": summary["profile"],
        "mode": summary["mode"],
        "profile_config": summary["profile_config"],
        "corpus_fingerprint": summary["corpus_fingerprint"],
        "runner": "mini-swe-agent-1.14.4-pinned-core",
        "model": "target-model",
        "model_adapter": summary.get("model_adapter"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in run_dir.iterdir():
        path.chmod(0o600)
    # These are wrapper/verifier source records, not an agent-facing API.  The
    # public CLI returns its explicit allowlisted summary; keeping the backing
    # volume root-only prevents a later native Codex session from recovering a
    # prior round's raw controller artifacts through /app/feedback.
    run_dir.chmod(0o700)


def public_pair_payload(pair: dict[str, Any]) -> dict[str, Any]:
    """Remove the controller-private raw nonce from a public pair record."""

    baseline = pair.get("baseline")
    candidate = pair.get("candidate")
    return {
        "schema_version": 1,
        "kind": "kvbench-public-pair",
        "pair_id": pair["pair_id"],
        "baseline_anchor_pair_id": pair.get("baseline_anchor_pair_id"),
        "parent_pair_id": pair.get("parent_pair_id"),
        "profile": pair["profile"],
        "status": pair["status"],
        "created_unix": pair["created_unix"],
        "updated_unix": pair["updated_unix"],
        "controller_config_sha256": pair.get("controller_config_sha256"),
        "baseline_model_provenance_sha256": pair.get(
            "baseline_model_provenance_sha256"
        ),
        "candidate_model_provenance_sha256": pair.get(
            "candidate_model_provenance_sha256"
        ),
        "corpus_fingerprint": (
            baseline.get("corpus_fingerprint") if isinstance(baseline, dict) else None
        ),
        "workload_nonce_sha256": (
            baseline.get("workload_nonce_sha256") if isinstance(baseline, dict) else None
        ),
        "baseline": baseline,
        "candidate": candidate,
        "failure": pair.get("failure"),
        "comparison": pair.get("comparison"),
    }


def public_pair_directory(pair: dict[str, Any]) -> Path:
    pair_id = str(pair["pair_id"])
    if not re.fullmatch(r"[a-z0-9-]+", pair_id):
        raise ValueError("invalid public pair ID")
    return FEEDBACK_ROOT / "pairs" / pair_id


def atomic_public_json(destination: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a public artifact through a per-write unique file."""

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        for _ in range(8):
            candidate = destination.parent / (
                f".{destination.name}.{secrets.token_hex(12)}.tmp"
            )
            try:
                with candidate.open("x", encoding="utf-8") as handle:
                    temporary = candidate
                    handle.write(encoded)
                break
            except FileExistsError:  # pragma: no cover - cryptographic collision
                continue
        if temporary is None:  # pragma: no cover - cryptographic collision
            raise RuntimeError("could not allocate a unique public artifact temporary")
        temporary.chmod(0o600)
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def persist_public_pair_state(pair: dict[str, Any]) -> Path:
    pair_root = FEEDBACK_ROOT / "pairs"
    pair_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    pair_root.chmod(0o700)
    pair_dir = public_pair_directory(pair)
    pair_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = pair_dir / "pair.json"
    atomic_public_json(destination, public_pair_payload(pair))
    pair_dir.chmod(0o700)
    return pair_dir


def persist_public_compare(pair: dict[str, Any], payload: dict[str, Any]) -> Path:
    pair_dir = persist_public_pair_state(pair)
    destination = pair_dir / "compare.json"
    atomic_public_json(destination, payload)
    return destination


def persist_trusted_result(kind: str, identifier: str, payload: dict[str, Any]) -> Path:
    safe_identifier = "".join(character for character in identifier if character.isalnum() or character in "-_")
    if not safe_identifier:
        raise ValueError("trusted result identifier is empty")
    TRUSTED_RESULTS_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    TRUSTED_RESULTS_ROOT.chmod(0o700)
    destination = TRUSTED_RESULTS_ROOT / f"{kind}-{safe_identifier}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(destination)
    return destination


async def execute_leg(
    app: web.Application,
    *,
    profile: Profile,
    mode: str,
    corpus_namespace: str,
    workload_nonce: str,
    expected_model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {"baseline", "candidate"}:
        raise infrastructure_leg_failure(
            "controller_leg_mode_invalid",
            f"unexpected benchmark leg mode: {mode!r}",
        )
    if mode == "candidate":
        try:
            healthy = await candidate_healthy(app)
        except Exception as exc:
            raise candidate_leg_failure(
                "candidate_health_protocol_failure",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if not healthy:
            raise candidate_leg_failure(
                "candidate_gateway_unavailable",
                "candidate gateway did not return a successful health response",
            )
    endpoint = BASELINE_URL if mode == "baseline" else CANDIDATE_URL
    try:
        provenance_before = await model_provenance(app)
    except Exception as exc:
        raise infrastructure_leg_failure(
            "model_provenance_preflight_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if (
        expected_model_provenance is not None
        and not same_backend_serving_contract(
            provenance_before, expected_model_provenance
        )
    ):
        raise infrastructure_leg_failure(
            "model_provenance_preflight_mismatch",
            "backend serving contract changed before benchmark dispatch",
        )
    try:
        admin_token = token_text()
    except Exception as exc:
        raise infrastructure_leg_failure(
            "admin_token_unavailable",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        result = await asyncio.to_thread(
            run_benchmark,
            profile=profile,
            endpoint=endpoint,
            mode=mode,
            model_api_base=MODEL_API_URL,
            admin_token=admin_token,
            corpus_namespace=corpus_namespace,
            workload_nonce=workload_nonce,
        )
    except Exception as exc:
        if mode == "baseline":
            raise infrastructure_leg_failure(
                "baseline_runtime_failed",
                f"{type(exc).__name__}: {exc}",
            ) from exc

        # A candidate leg follows a successful A leg.  Attribute an exception
        # from its untrusted endpoint to the candidate unless a fresh trusted
        # control-plane probe proves that the fixed model infrastructure failed.
        try:
            model_healthy, model_attested = await model_status(app)
            provenance_after_failure = await model_provenance(app)
        except Exception as diagnostic_exc:
            raise infrastructure_leg_failure(
                "model_control_plane_diagnostic_failed",
                f"{type(diagnostic_exc).__name__}: {diagnostic_exc}",
            ) from exc
        if not model_healthy or not model_attested:
            raise infrastructure_leg_failure(
                "model_api_unhealthy_during_candidate_leg",
                f"healthy={model_healthy},attested={model_attested}",
            ) from exc
        if not same_backend_serving_contract(
            provenance_after_failure, provenance_before
        ):
            raise infrastructure_leg_failure(
                "model_provenance_changed_during_candidate_leg",
                "fixed model provenance changed while the candidate leg failed",
            ) from exc
        try:
            gateway_still_healthy = await candidate_healthy(app)
        except Exception as health_exc:
            raise candidate_leg_failure(
                "candidate_health_protocol_failure",
                f"{type(health_exc).__name__}: {health_exc}",
            ) from exc
        failure_type = (
            "candidate_runtime_failed"
            if gateway_still_healthy
            else "candidate_gateway_exited"
        )
        raise candidate_leg_failure(
            failure_type,
            f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        provenance_after = await model_provenance(app)
    except Exception as exc:
        raise infrastructure_leg_failure(
            "model_provenance_postflight_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if not same_backend_serving_contract(provenance_after, provenance_before):
        raise infrastructure_leg_failure(
            "model_provenance_changed",
            "fixed model provenance changed during benchmark leg",
        )
    result["release_id"] = app["policy"].get("release_id")
    result["controller_config_sha256"] = app["config_sha256"]
    result["model_provenance"] = provenance_before
    return result


async def public_run(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        return public_error_response("invalid_json")
    if not isinstance(payload, dict):
        return public_error_response("invalid_json")
    mode = payload.get("mode")
    profile_name = payload.get("profile")
    if not isinstance(mode, str) or mode not in {"baseline", "candidate"}:
        return public_error_response("invalid_mode")
    if not isinstance(profile_name, str):
        return public_error_response("invalid_profile")
    profiles: dict[str, Profile] = request.app["profile_sets"]["public"]
    if profile_name not in profiles or not profiles[profile_name].enabled:
        return public_error_response("unknown_or_disabled_profile")
    if request.app["lease"].locked() or request.app.get("final_session") is not None:
        return public_error_response("benchmark_lease_busy")

    async with request.app["lease"]:
        corpus_namespace = "public"
        current_provenance: dict[str, Any] | None = None
        use_release_anchor = (
            profile_name == RELEASE_PUBLIC_PROFILE
            and request.app.get("release_public_anchor_required") is True
        )
        if use_release_anchor:
            try:
                anchor_material, current_provenance = (
                    await current_release_public_anchor_material(request.app)
                )
            except Exception as exc:
                persist_public_failure_artifact(
                    error="baseline_model_identity_invalid",
                    raw_detail=f"{type(exc).__name__}: {exc}",
                    pair=request.app["public_pair"].get(profile_name),
                    context={"mode": mode, "profile": profile_name},
                )
                return public_error_response("baseline_model_identity_invalid")
            anchor_pair = anchor_material.get("pair")
            if (
                not isinstance(anchor_pair, dict)
                or anchor_pair.get("pair_id")
                != request.app.get("release_public_anchor_pair_id")
                or anchor_pair.get("baseline_anchor_pair_id")
                != anchor_pair.get("pair_id")
                or anchor_pair.get("parent_pair_id") is not None
                or anchor_pair.get("status") != "baseline_complete"
                or not isinstance(anchor_pair.get("baseline"), dict)
                or anchor_pair.get("candidate") is not None
                or not isinstance(anchor_material.get("corpus_namespace"), str)
            ):
                persist_public_failure_artifact(
                    error="baseline_model_identity_invalid",
                    raw_detail="release public Direct anchor material is invalid",
                    pair=request.app["public_pair"].get(profile_name),
                    context={"mode": mode, "profile": profile_name},
                )
                return public_error_response("baseline_model_identity_invalid")
            if mode == "baseline":
                # This response is a projection of immutable author evidence.
                # It deliberately performs no benchmark leg and therefore no
                # thirty-minute GPU measurement.
                pair = copy.deepcopy(anchor_pair)
                request.app["public_pair"][profile_name] = pair
                persist_public_pair_state(pair)
                summary = copy.deepcopy(pair["baseline"])
                return web.json_response(
                    {
                        "pair_id": pair["pair_id"],
                        "baseline_anchor_pair_id": pair[
                            "baseline_anchor_pair_id"
                        ],
                        "parent_pair_id": None,
                        "summary": summary,
                        "text": format_run(summary),
                    }
                )

            baseline = anchor_pair["baseline"]
            baseline_provenance = anchor_pair.get(
                "baseline_model_provenance_sha256"
            )
            workload_nonce = anchor_pair.get("workload_nonce")
            anchor_pair_id = anchor_pair["pair_id"]
            if (
                not isinstance(baseline_provenance, str)
                or not _hex_text(baseline_provenance, 64)
                or model_provenance_sha256(current_provenance)
                != baseline_provenance
                or not isinstance(workload_nonce, str)
                or not workload_nonce
            ):
                persist_public_failure_artifact(
                    error="baseline_model_identity_invalid",
                    raw_detail="release public Direct anchor identity is invalid",
                    pair=anchor_pair,
                    context={"mode": mode, "profile": profile_name},
                )
                return public_error_response("baseline_model_identity_invalid")
            now = time.time()
            pair = {
                "pair_id": f"{profile_name}-{secrets.token_hex(8)}",
                "baseline_anchor_pair_id": anchor_pair_id,
                "parent_pair_id": anchor_pair_id,
                "profile": profile_name,
                "status": "candidate_running",
                "created_unix": now,
                "updated_unix": now,
                "controller_config_sha256": request.app["config_sha256"],
                "workload_nonce": workload_nonce,
                "baseline": copy.deepcopy(baseline),
                "candidate": None,
                "baseline_model_provenance_sha256": baseline_provenance,
                "candidate_model_provenance_sha256": None,
                "failure": None,
                "comparison": None,
            }
            corpus_namespace = anchor_material["corpus_namespace"]
            request.app["public_pair"][profile_name] = pair
        elif mode == "baseline":
            workload_nonce = secrets.token_hex(16)
            now = time.time()
            pair_id = f"{profile_name}-{secrets.token_hex(8)}"
            pair = {
                "pair_id": pair_id,
                "baseline_anchor_pair_id": pair_id,
                "parent_pair_id": None,
                "profile": profile_name,
                "status": "baseline_running",
                "created_unix": now,
                "updated_unix": now,
                "controller_config_sha256": request.app["config_sha256"],
                "workload_nonce": workload_nonce,
                "baseline": None,
                "candidate": None,
                "baseline_model_provenance_sha256": None,
                "candidate_model_provenance_sha256": None,
                "failure": None,
                "comparison": None,
            }
            request.app["public_pair"][profile_name] = pair
        else:
            source_pair = request.app["public_pair"].get(profile_name)
            if source_pair is None:
                return public_error_response("baseline_required")
            baseline = source_pair.get("baseline")
            if not isinstance(baseline, dict):
                return public_error_response("baseline_required")
            if baseline.get("infrastructure_failure") is not None:
                return public_error_response("baseline_infrastructure_failure")
            if (
                baseline.get("candidate_failure") is not None
                or not isinstance(baseline.get("backend_audit"), dict)
                or baseline["backend_audit"].get("passed") is not True
                or not isinstance(baseline.get("warmup_audit"), dict)
                or baseline["warmup_audit"].get("passed") is not True
            ):
                return public_error_response("baseline_audit_invalid")
            baseline_provenance = baseline.get("model_provenance_sha256")
            if (
                not isinstance(baseline_provenance, str)
                or re.fullmatch(r"[0-9a-f]{64}", baseline_provenance) is None
                or source_pair.get("baseline_model_provenance_sha256")
                != baseline_provenance
            ):
                return public_error_response("baseline_model_identity_invalid")
            # A model profile/image digest is deliberately stable across a
            # host restart, while its cache contents and process epoch are not.
            # Re-attest immediately before a candidate leg so a thirty-minute
            # stale-baseline run is rejected before it consumes GPU time.
            try:
                current_provenance = await model_provenance(request.app)
            except Exception as exc:
                persist_public_failure_artifact(
                    error="baseline_model_identity_invalid",
                    raw_detail=f"{type(exc).__name__}: {exc}",
                    pair=source_pair,
                    context={"mode": mode, "profile": profile_name},
                )
                return public_error_response("baseline_model_identity_invalid")
            if model_provenance_sha256(current_provenance) != baseline_provenance:
                persist_public_failure_artifact(
                    error="baseline_model_identity_invalid",
                    raw_detail="baseline/current model runtime provenance digest mismatch",
                    pair=source_pair,
                    context={
                        "mode": mode,
                        "profile": profile_name,
                        "current_model_provenance": current_provenance,
                    },
                )
                return public_error_response("baseline_model_identity_invalid")
            if profiles[profile_name].pressure_required and not baseline.get("pressure_validation", {}).get("valid"):
                return public_error_response("baseline_not_pressurized")
            baseline_window = baseline.get("measurement_window", {})
            if profiles[profile_name].measurement_window_sec is not None and (
                baseline_window.get("deadline_reached") is not True
                or baseline_window.get("passed") is not True
            ):
                return public_error_response("baseline_window_invalid")
            if profiles[profile_name].measurement_window_sec is not None:
                zero_progress = baseline_window.get("zero_progress_fraction")
                request_success = baseline.get("request_success_rate")
                quality_errors: list[str] = []
                if (
                    not isinstance(zero_progress, (int, float))
                    or isinstance(zero_progress, bool)
                    or not math.isfinite(float(zero_progress))
                    or float(zero_progress) > 0.05
                ):
                    quality_errors.append("window_zero_progress_above_0.05")
                if (
                    not isinstance(request_success, (int, float))
                    or isinstance(request_success, bool)
                    or not math.isfinite(float(request_success))
                    or float(request_success) < 0.98
                ):
                    quality_errors.append("window_request_success_below_0.98")
                if quality_errors:
                    return public_error_response(
                        "baseline_quality_invalid",
                        gates=tuple(f"baseline_{error}" for error in quality_errors),
                    )
            else:
                interval_errors = finite_agent_execution_interval_errors(baseline)
                if interval_errors:
                    return public_error_response(
                        "baseline_interval_invalid",
                        gates=tuple(
                            f"baseline_agent_execution_interval_{error}"
                            for error in interval_errors
                        ),
                    )
            if source_pair.get("controller_config_sha256") != request.app["config_sha256"]:
                return public_error_response("baseline_controller_identity_invalid")
            workload_nonce = source_pair["workload_nonce"]
            now = time.time()
            pair_id = f"{profile_name}-{secrets.token_hex(8)}"
            anchor_pair_id = source_pair.get("baseline_anchor_pair_id")
            if not isinstance(anchor_pair_id, str):
                anchor_pair_id = str(source_pair["pair_id"])
            # A candidate retry is a new immutable evidence record.  It reuses
            # only the trusted baseline and private workload nonce; the source
            # pair (successful or failed) is never edited or unlinked.
            pair = {
                "pair_id": pair_id,
                "baseline_anchor_pair_id": anchor_pair_id,
                "parent_pair_id": str(source_pair["pair_id"]),
                "profile": profile_name,
                "status": "candidate_running",
                "created_unix": now,
                "updated_unix": now,
                "controller_config_sha256": request.app["config_sha256"],
                "workload_nonce": workload_nonce,
                "baseline": copy.deepcopy(baseline),
                "candidate": None,
                "baseline_model_provenance_sha256": baseline_provenance,
                "candidate_model_provenance_sha256": None,
                "failure": None,
                "comparison": None,
            }
            request.app["public_pair"][profile_name] = pair
        persist_public_pair_state(pair)
        try:
            result = await execute_leg(
                request.app,
                profile=profiles[profile_name],
                mode=mode,
                corpus_namespace=corpus_namespace,
                workload_nonce=workload_nonce,
                expected_model_provenance=(
                    current_provenance if mode == "candidate" else None
                ),
            )
        except Exception as exc:
            pair["status"] = f"{mode}_failed"
            pair["updated_unix"] = time.time()
            pair["failure"] = {
                "mode": mode,
                "type": type(exc).__name__,
                "detail": str(exc)[:2000],
            }
            persist_public_pair_state(pair)
            return public_error_response("benchmark_failed", pair=pair)
        # Store the complete leg before projecting the candidate-visible
        # summary.  This is where raw watchdog/measurement failure detail is
        # retained after the public recursive allowlist strips it.
        persist_trusted_result("public-leg", result["run_id"], result)
        summary = public_summary(result)
        if mode == "baseline":
            pair["baseline"] = summary
            pair["candidate"] = None
            pair["baseline_model_provenance_sha256"] = summary.get(
                "model_provenance_sha256"
            )
            pair["candidate_model_provenance_sha256"] = None
            pair["status"] = "baseline_complete"
        else:
            pair["candidate"] = summary
            pair["candidate_model_provenance_sha256"] = summary.get(
                "model_provenance_sha256"
            )
            pair["status"] = "candidate_complete"
        pair["updated_unix"] = time.time()
        persist_public_pair_state(pair)
        persist_feedback(summary)
        request.app["history"].append(summary)
        return web.json_response(
            {
                "pair_id": pair["pair_id"],
                "baseline_anchor_pair_id": pair.get(
                    "baseline_anchor_pair_id"
                ),
                "parent_pair_id": pair.get("parent_pair_id"),
                "summary": summary,
                "text": format_run(summary),
            }
        )


async def compare(request: web.Request) -> web.Response:
    if request.app["lease"].locked() or request.app.get("final_session") is not None:
        return public_error_response("benchmark_lease_busy")
    profile = request.query.get("profile")
    pair = request.app["public_pair"].get(profile)
    if not isinstance(pair, dict):
        return public_error_response("missing_valid_pair")
    if pair.get("status") != "candidate_complete" or pair.get("failure") is not None:
        return public_error_response("pair_not_ready", pair=pair)
    baseline = pair.get("baseline") if isinstance(pair, dict) else None
    candidate = pair.get("candidate") if isinstance(pair, dict) else None
    if not baseline or not candidate:
        return public_error_response("missing_valid_pair")
    invalid: list[str] = []
    for label, item in (("baseline", baseline), ("candidate", candidate)):
        if item.get("infrastructure_failure") is not None:
            invalid.append(f"{label}_infrastructure_failure")
        if item.get("candidate_failure") is not None:
            invalid.append(f"{label}_candidate_failure")
        if (
            not isinstance(item.get("backend_audit"), dict)
            or item["backend_audit"].get("passed") is not True
            or not isinstance(item.get("warmup_audit"), dict)
            or item["warmup_audit"].get("passed") is not True
        ):
            invalid.append(f"{label}_audit_failed")
    if not baseline.get("pressure_validation", {}).get("valid", True):
        invalid.append("baseline_profile_not_pressurized")
    if baseline["workload_nonce_sha256"] != candidate["workload_nonce_sha256"]:
        invalid.append("workload_pair_mismatch")
    if baseline["corpus_fingerprint"] != candidate["corpus_fingerprint"]:
        invalid.append("corpus_mismatch")
    baseline_provenance = baseline.get("model_provenance_sha256")
    candidate_provenance = candidate.get("model_provenance_sha256")
    for label, digest, pair_digest in (
        (
            "baseline",
            baseline_provenance,
            pair.get("baseline_model_provenance_sha256"),
        ),
        (
            "candidate",
            candidate_provenance,
            pair.get("candidate_model_provenance_sha256"),
        ),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            invalid.append(f"{label}_model_provenance_missing")
        elif pair_digest != digest:
            invalid.append(f"{label}_model_provenance_pair_state_mismatch")
    if baseline_provenance != candidate_provenance:
        invalid.append("model_provenance_mismatch")

    def finite_number(value: object) -> float | None:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
        return None

    baseline_window = baseline.get("measurement_window", {})
    candidate_window = candidate.get("measurement_window", {})
    if baseline_window.get("configured") or candidate_window.get("configured"):
        public_profile = request.app["profile_sets"]["public"].get(profile)
        expected_duration = (
            public_profile.measurement_window_sec
            if public_profile is not None
            else None
        )
        expected_warmup = (
            public_profile.measurement_warmup_sec
            if public_profile is not None
            else None
        )
        if (
            "agent_execution_interval" in baseline
            or "agent_execution_interval" in candidate
        ):
            invalid.append("fixed_window_contains_finite_agent_execution_interval")
        if not (
            baseline_window.get("configured") is True
            and candidate_window.get("configured") is True
            and baseline_window.get("duration_sec")
            == candidate_window.get("duration_sec")
            == expected_duration
            and baseline_window.get("warmup_duration_sec")
            == candidate_window.get("warmup_duration_sec")
            == expected_warmup
        ):
            invalid.append("measurement_window_contract_mismatch")
        for label, window in (
            ("baseline", baseline_window),
            ("candidate", candidate_window),
        ):
            if window.get("deadline_reached") is not True:
                invalid.append(f"{label}_measurement_window_not_reached")
            if window.get("passed") is not True:
                invalid.append(f"{label}_measurement_window_failed")
            zero_progress = finite_number(window.get("zero_progress_fraction"))
            request_success = finite_number(
                (baseline if label == "baseline" else candidate).get(
                    "request_success_rate"
                )
            )
            if zero_progress is None or zero_progress > 0.05:
                invalid.append(f"{label}_window_zero_progress_above_0.05")
            if request_success is None or request_success < 0.98:
                invalid.append(f"{label}_window_request_success_below_0.98")
    else:
        # Small public profiles are complete finite workloads.  They use the
        # same trusted denominator contract as the hidden two-leg pair, including the
        # timestamp-only public sample projection created by public_summary().
        for label, item in (("baseline", baseline), ("candidate", candidate)):
            interval_errors = finite_agent_execution_interval_errors(item)
            invalid.extend(
                f"{label}_agent_execution_interval_{error}"
                for error in interval_errors
            )
    if invalid:
        failure_payload = {
            "schema_version": 1,
            "kind": "kvbench-public-compare",
            "pair_id": pair["pair_id"],
            "baseline_anchor_pair_id": pair.get("baseline_anchor_pair_id"),
            "parent_pair_id": pair.get("parent_pair_id"),
            "profile": profile,
            "status": "invalid",
            "error": "invalid_latest_pair",
            "detail": invalid,
            "baseline_run_id": baseline["run_id"],
            "candidate_run_id": candidate["run_id"],
            "baseline": baseline,
            "candidate": candidate,
        }
        pair["comparison"] = {
            "status": "invalid",
            "updated_unix": time.time(),
            "detail": list(invalid),
        }
        pair["updated_unix"] = time.time()
        persist_public_compare(pair, failure_payload)
        return public_error_response(
            "invalid_latest_pair",
            pair=pair,
            gates=tuple(invalid),
        )
    base = baseline["valid_steps_per_min"]
    cand = candidate["valid_steps_per_min"]
    speedup = cand / base if base else 0.0
    baseline_live = baseline.get("vllm_metrics", {}).get("live_series", {})
    candidate_live = candidate.get("vllm_metrics", {}).get("live_series", {})
    baseline_derived = baseline.get("vllm_metrics", {}).get("derived", {})
    candidate_derived = candidate.get("vllm_metrics", {}).get("derived", {})

    def metric(item: dict[str, Any], key: str) -> float:
        value = item.get(key)
        return float(value) if isinstance(value, (int, float)) else math.nan

    rendered = "\n".join(
        [
            "metric                       baseline       candidate        change",
            f"valid steps/min             {base:12.3f} {cand:15.3f} {(speedup - 1) * 100:+12.2f}%",
            f"request success             {baseline['request_success_rate']:12.3%} {candidate['request_success_rate']:15.3%}",
            f"p99 TTFT (s)                {float(baseline['p99_ttft_sec'] or 0):12.3f} "
            f"{float(candidate['p99_ttft_sec'] or 0):15.3f}",
            f"KV usage p95                {metric(baseline_live, 'kv_cache_usage_p95'):12.4f} "
            f"{metric(candidate_live, 'kv_cache_usage_p95'):15.4f}",
            f"prefix-cache hit ratio      {metric(baseline_derived, 'prefix_cache_hit_ratio'):12.4f} "
            f"{metric(candidate_derived, 'prefix_cache_hit_ratio'):15.4f}",
            f"waiting requests max        {metric(baseline_live, 'requests_waiting_max'):12.1f} "
            f"{metric(candidate_live, 'requests_waiting_max'):15.1f}",
            f"one-to-one audit            {'pass' if baseline['backend_audit']['passed'] else 'FAIL':>12} "
            f"{'pass' if candidate['backend_audit']['passed'] else 'FAIL':>15}",
        ]
    )
    comparison = {
        "schema_version": 1,
        "kind": "kvbench-public-compare",
        "pair_id": pair["pair_id"],
        "baseline_anchor_pair_id": pair.get("baseline_anchor_pair_id"),
        "parent_pair_id": pair.get("parent_pair_id"),
        "profile": profile,
        "status": "valid",
        "speedup": speedup,
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "controller_config_sha256": baseline.get("controller_config_sha256"),
        "corpus_fingerprint": baseline["corpus_fingerprint"],
        "workload_nonce_sha256": baseline["workload_nonce_sha256"],
        "model_provenance_sha256": baseline_provenance,
        "baseline": baseline,
        "candidate": candidate,
        "text": rendered,
    }
    pair["comparison"] = {
        "status": "valid",
        "updated_unix": time.time(),
        "speedup": speedup,
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
    }
    pair["updated_unix"] = time.time()
    persist_public_compare(pair, comparison)
    return web.json_response(comparison)


async def readiness(request: web.Request) -> web.Response:
    require_admin(request)
    healthy, attested = await model_status(request.app)
    current_model_provenance: dict[str, Any] | None = None
    if healthy and attested:
        try:
            current_model_provenance = await model_provenance(request.app)
        except Exception:
            healthy = False
            attested = False
    policy = request.app["policy"]
    public_ready = public_profiles_ready(request.app["profile_sets"]["public"])
    workload_ready = real_workload_ready(request.app["profile_sets"]["public"])
    release_ready = bool(policy.get("release_ready"))
    return web.json_response(
        {
            "phase": 1,
            "release_id": policy.get("release_id"),
            "policy_contract_sha256": policy_contract_sha256(policy),
            "model_api": healthy,
            "infrastructure_attested": attested,
            "public_profiles": public_ready,
            "real_swebench_workload": workload_ready,
            "hidden_profiles": bool(request.app["profile_sets"]["hidden"]),
            "oracle_calibrated": release_ready,
            "real_agent_test_allowed": (
                release_ready and healthy and attested and public_ready and workload_ready
            ),
            "controller_config_sha256": request.app["config_sha256"],
            "corpus_manifest_sha256": file_sha256(CORPUS_MANIFEST),
            "model_adapter": request.app["model_adapter"],
            # Admin-only, opaque dynamic identity. Raw host process/container
            # attestation fields remain confined to model-api and verifier.
            "model_provenance": current_model_provenance,
        }
    )


async def admin_run(request: web.Request) -> web.Response:
    require_admin(request)
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": "invalid_json"}, status=400)
    suite = payload.get("suite")
    profile_name = payload.get("profile")
    mode = payload.get("mode")
    if suite not in request.app["profile_sets"] or suite not in {"public", "calibration", "hidden"}:
        return web.json_response({"error": "unknown_suite"}, status=404)
    profiles: dict[str, Profile] = request.app["profile_sets"][suite]
    if profile_name not in profiles or not profiles[profile_name].enabled:
        return web.json_response({"error": "unknown_or_disabled_profile"}, status=404)
    if mode not in {"baseline", "candidate"}:
        return web.json_response({"error": "invalid_mode"}, status=400)
    if request.app["lease"].locked() or request.app.get("final_session") is not None:
        return web.json_response({"error": "benchmark_lease_busy"}, status=409)
    async with request.app["lease"]:
        try:
            result = await execute_leg(
                request.app,
                profile=profiles[profile_name],
                mode=mode,
                corpus_namespace=f"{suite}-{profile_name}",
                workload_nonce=str(payload.get("workload_nonce") or secrets.token_hex(16)),
            )
        except Exception as exc:
            return web.json_response(
                {"error": "benchmark_failed", "detail": f"{type(exc).__name__}: {exc}"}, status=500
            )
    persist_trusted_result(f"{suite}-{mode}", result["run_id"], result)
    return web.json_response({"result": result})


def finite_agent_execution_interval_errors(leg: dict[str, Any]) -> list[str]:
    """Validate the frozen finite-leg timing/rate identity fail closed."""

    errors: list[str] = []
    interval = leg.get("agent_execution_interval")
    expected_keys = {
        "schema_version",
        "start_event",
        "end_event",
        "duration_clock",
        "started_unix",
        "ended_unix",
        "elapsed_sec",
        "expected_agent_runs",
        "observed_agent_runs",
        "complete",
    }
    if not isinstance(interval, dict):
        return ["missing"]
    if set(interval) != expected_keys:
        errors.append("schema_fields")
    if interval.get("schema_version") != 1:
        errors.append("schema_version")
    if interval.get("start_event") != "measured_leg_started":
        errors.append("start_event")
    if interval.get("end_event") != "last_agent_run_termination":
        errors.append("end_event")
    if interval.get("duration_clock") != "monotonic":
        errors.append("duration_clock")
    if interval.get("complete") is not True:
        errors.append("incomplete")

    def finite_number(value: object) -> float | None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    started = finite_number(interval.get("started_unix"))
    ended = finite_number(interval.get("ended_unix"))
    elapsed = finite_number(interval.get("elapsed_sec"))
    if started is None or ended is None or elapsed is None or elapsed <= 0:
        errors.append("timestamps")
    elif ended < started:
        errors.append("unix_order")

    profile = leg.get("profile_config")
    workflows = profile.get("workflows") if isinstance(profile, dict) else None
    expected = interval.get("expected_agent_runs")
    observed = interval.get("observed_agent_runs")
    if (
        not isinstance(workflows, int)
        or isinstance(workflows, bool)
        or workflows <= 0
        or expected != workflows
        or observed != workflows
    ):
        errors.append("agent_run_count")

    for field, interval_value in (
        ("started_unix", started),
        ("ended_unix", ended),
        ("elapsed_sec", elapsed),
    ):
        top_value = finite_number(leg.get(field))
        if (
            interval_value is None
            or top_value is None
            or not math.isclose(top_value, interval_value, rel_tol=1e-9, abs_tol=1e-6)
        ):
            errors.append(f"top_level_{field}")
    wall_elapsed = finite_number(leg.get("wall_elapsed_sec"))
    if elapsed is None or wall_elapsed is None or wall_elapsed + 1e-6 < elapsed:
        errors.append("wall_elapsed")

    valid_steps = leg.get("valid_steps")
    rate = finite_number(leg.get("valid_steps_per_min"))
    if (
        not isinstance(valid_steps, int)
        or isinstance(valid_steps, bool)
        or valid_steps < 0
        or elapsed is None
        or elapsed <= 0
        or rate is None
        or not math.isclose(rate, valid_steps * 60.0 / elapsed, rel_tol=1e-9, abs_tol=1e-9)
    ):
        errors.append("rate_identity")

    window = leg.get("measurement_window")
    if (
        not isinstance(window, dict)
        or window.get("configured") is not False
        or window.get("deadline_reached") is not False
        or window.get("termination") != "finite_workload"
    ):
        errors.append("finite_window_contract")
    samples = (leg.get("vllm_metrics") or {}).get("live_samples")
    if not isinstance(samples, list) or not samples or ended is None:
        errors.append("live_samples")
    elif any(
        finite_number(sample.get("created_unix")) is None
        or float(sample["created_unix"]) > ended + 1e-6
        for sample in samples
        if isinstance(sample, dict)
    ) or any(not isinstance(sample, dict) for sample in samples):
        errors.append("post_end_live_sample")
    sampler_errors = ((leg.get("vllm_metrics") or {}).get("live_series") or {}).get(
        "sampler_error_count"
    )
    if (
        not isinstance(sampler_errors, int)
        or isinstance(sampler_errors, bool)
        or sampler_errors != 0
    ):
        errors.append("metrics_sampler_errors")
    return sorted(set(errors))


def fixed_measurement_window_errors(leg: dict[str, Any]) -> list[str]:
    """Validate a fixed-window leg, including an optional unscored warmup."""

    errors: list[str] = []
    profile = leg.get("profile_config")
    window = leg.get("measurement_window")
    if not isinstance(profile, dict):
        return ["profile_config"]
    if not isinstance(window, dict):
        return ["measurement_window"]
    if "agent_execution_interval" in leg:
        errors.append("contains_finite_agent_execution_interval")

    duration = profile.get("measurement_window_sec")
    warmup = profile.get("measurement_warmup_sec", 0)
    if (
        not isinstance(duration, int)
        or isinstance(duration, bool)
        or duration <= 0
        or not isinstance(warmup, int)
        or isinstance(warmup, bool)
        or warmup < 0
    ):
        errors.append("profile_window")
        duration = None
        warmup = None

    def finite_number(value: object) -> float | None:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
        return None

    workload_started = finite_number(leg.get("workload_started_unix"))
    started = finite_number(leg.get("started_unix"))
    ended = finite_number(leg.get("ended_unix"))
    wall_ended = finite_number(leg.get("wall_ended_unix"))
    elapsed = finite_number(leg.get("elapsed_sec"))
    wall_elapsed = finite_number(leg.get("wall_elapsed_sec"))
    window_workload_started = finite_number(window.get("workload_started_unix"))
    window_started = finite_number(window.get("started_unix"))
    cutoff = finite_number(window.get("cutoff_unix"))
    measurement_elapsed = finite_number(window.get("measurement_elapsed_sec"))
    if None in (
        workload_started,
        started,
        ended,
        wall_ended,
        elapsed,
        wall_elapsed,
        window_workload_started,
        window_started,
        cutoff,
        measurement_elapsed,
    ):
        errors.append("timestamps")
    elif duration is not None and warmup is not None:
        if not math.isclose(
            started - workload_started, float(warmup), rel_tol=1e-9, abs_tol=1e-6
        ):
            errors.append("warmup_interval")
        if not math.isclose(ended - started, float(duration), rel_tol=1e-9, abs_tol=1e-6):
            errors.append("measurement_interval")
        if not math.isclose(elapsed, float(duration), rel_tol=1e-9, abs_tol=1e-6):
            errors.append("elapsed")
        if not math.isclose(
            wall_ended - started, wall_elapsed, rel_tol=1e-9, abs_tol=1e-6
        ) or wall_elapsed + 1e-6 < elapsed:
            errors.append("wall_elapsed")
        if (
            not math.isclose(window_workload_started, workload_started, abs_tol=1e-6)
            or not math.isclose(window_started, started, abs_tol=1e-6)
            or not math.isclose(cutoff, ended, abs_tol=1e-6)
            or not math.isclose(measurement_elapsed, elapsed, abs_tol=1e-6)
        ):
            errors.append("window_top_level_identity")

    if (
        window.get("configured") is not True
        or window.get("duration_sec") != duration
        or window.get("warmup_duration_sec") != warmup
        or window.get("deadline_reached") is not True
        or window.get("termination") != "trusted_fixed_window"
        or window.get("control_passed") is not True
        or window.get("audit_passed") is not True
        or window.get("backend_quiescent_after_cutoff") is not True
        or window.get("passed") is not True
        or window.get("errors") != []
    ):
        errors.append("window_control")

    valid_steps = leg.get("valid_steps")
    total_steps = leg.get("total_executed_steps")
    pre_steps = window.get("pre_measurement_steps")
    post_steps = window.get("post_cutoff_steps")
    rate = finite_number(leg.get("valid_steps_per_min"))
    if (
        not isinstance(valid_steps, int)
        or isinstance(valid_steps, bool)
        or valid_steps < 0
        or not isinstance(total_steps, int)
        or isinstance(total_steps, bool)
        or not isinstance(pre_steps, int)
        or isinstance(pre_steps, bool)
        or not isinstance(post_steps, int)
        or isinstance(post_steps, bool)
        or pre_steps < 0
        or post_steps < 0
        or total_steps != pre_steps + valid_steps + post_steps
    ):
        errors.append("step_partition")
    if (
        isinstance(valid_steps, int)
        and not isinstance(valid_steps, bool)
        and elapsed is not None
        and elapsed > 0
        and rate is not None
    ):
        if not math.isclose(
            rate, valid_steps * 60.0 / elapsed, rel_tol=1e-9, abs_tol=1e-9
        ):
            errors.append("rate_identity")
    else:
        errors.append("rate_identity")

    samples = (leg.get("vllm_metrics") or {}).get("live_samples")
    if not isinstance(samples, list) or not samples or started is None or ended is None:
        errors.append("live_samples")
    elif any(
        not isinstance(sample, dict)
        or finite_number(sample.get("created_unix")) is None
        or float(sample["created_unix"]) < started - 1e-6
        or float(sample["created_unix"]) > ended + 1e-6
        for sample in samples
    ):
        errors.append("out_of_window_live_sample")
    sampler_errors = ((leg.get("vllm_metrics") or {}).get("live_series") or {}).get(
        "sampler_error_count"
    )
    if (
        not isinstance(sampler_errors, int)
        or isinstance(sampler_errors, bool)
        or sampler_errors != 0
    ):
        errors.append("metrics_sampler_errors")
    return sorted(set(errors))


def normalized_performance(speedup: float, reference_speedup: float, noise_floor: float) -> float:
    if speedup <= noise_floor * (1 + 1e-12):
        return 0.0
    if reference_speedup <= noise_floor:
        raise ValueError("reference speedup must exceed the noise floor")
    if speedup >= reference_speedup * (1 - 1e-12):
        return 1.0
    return min(1.0, max(0.0, math.log(speedup / noise_floor) / math.log(reference_speedup / noise_floor)))


def score_final(policy: dict[str, Any], profile_legs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    noise_floor = float(policy["noise_floor"])
    profile_reports: dict[str, Any] = {}
    infra_failures: list[str] = []
    candidate_failures: list[str] = []
    weighted_scores: list[tuple[float, float]] = []
    tail_values: list[float] = []

    for profile_policy in policy["hidden_profiles"]:
        name = profile_policy["name"]
        weight = float(profile_policy["weight"])
        legs = profile_legs.get(name)
        if not isinstance(legs, list) or len(legs) != FINAL_FORMAL_LEG_COUNT:
            raise ValueError(f"formal hidden profile {name} must contain exactly two legs")
        modes = [leg.get("mode") if isinstance(leg, dict) else None for leg in legs]
        if sorted(str(mode) for mode in modes) != ["baseline", "candidate"]:
            raise ValueError(f"formal hidden profile {name} must contain one baseline and one candidate")
        commitment_values = [leg.get("workload_nonce_sha256") for leg in legs]
        if (
            not all(_hex_text(value, 64) for value in commitment_values)
            or commitment_values[0] != commitment_values[1]
        ):
            raise ValueError(f"formal hidden profile {name} has inconsistent nonce commitments")
        workload_nonce_sha256 = commitment_values[0]
        expected_order = formal_leg_order(workload_nonce_sha256)
        if tuple(modes) != expected_order:
            raise ValueError(
                f"formal hidden profile {name} leg order does not match its nonce commitment"
            )
        baseline_index = modes.index("baseline")
        candidate_index = modes.index("candidate")
        baseline_leg = legs[baseline_index]
        candidate_leg = legs[candidate_index]
        baseline_legs = [baseline_leg]
        candidate_legs = [candidate_leg]
        window_flags = [
            (leg.get("measurement_window") or {}).get("configured") is True
            for leg in legs
        ]
        windowed_profile = all(window_flags)
        if any(window_flags) and not windowed_profile:
            infra_failures.append(f"{name}:mixed_measurement_contract")
        for index, leg in enumerate(legs):
            interval_errors = (
                fixed_measurement_window_errors(leg)
                if windowed_profile
                else finite_agent_execution_interval_errors(leg)
            )
            if interval_errors:
                infra_failures.append(
                    f"{name}:leg{index + 1}:"
                    + (
                        "measurement_window:"
                        if windowed_profile
                        else "agent_execution_interval:"
                    )
                    + ",".join(interval_errors)
                )
        reset_epoch = candidate_leg.get("candidate_reset_epoch")
        quiescence_epoch = baseline_leg.get("candidate_quiescence_epoch")
        lifecycle_valid = (
            isinstance(reset_epoch, dict)
            and reset_epoch.get("schema_version") == 1
            and reset_epoch.get("leg_ordinal") == candidate_index + 1
            and reset_epoch.get("mode") == "candidate"
            and _hex_text(reset_epoch.get("frozen_submission_sha256"), 64)
            and _hex_text(reset_epoch.get("challenge_sha256"), 64)
            and _hex_text(reset_epoch.get("epoch_nonce_sha256"), 64)
            and _hex_text(reset_epoch.get("attestation_sha256"), 64)
            and _hex_text(reset_epoch.get("gateway_process_identity_sha256"), 64)
            and isinstance(quiescence_epoch, dict)
            and quiescence_epoch.get("schema_version") == 1
            and quiescence_epoch.get("leg_ordinal") == baseline_index + 1
            and quiescence_epoch.get("mode") == "baseline"
            and _hex_text(quiescence_epoch.get("frozen_submission_sha256"), 64)
            and _hex_text(quiescence_epoch.get("challenge_sha256"), 64)
            and _hex_text(quiescence_epoch.get("epoch_nonce_sha256"), 64)
            and _hex_text(quiescence_epoch.get("attestation_sha256"), 64)
            and baseline_leg.get("candidate_reset_epoch") is None
            and candidate_leg.get("candidate_quiescence_epoch") is None
        )
        if lifecycle_valid:
            lifecycle_valid = (
                reset_epoch["frozen_submission_sha256"]
                == quiescence_epoch["frozen_submission_sha256"]
                and reset_epoch["challenge_sha256"] != quiescence_epoch["challenge_sha256"]
                and reset_epoch["epoch_nonce_sha256"] != quiescence_epoch["epoch_nonce_sha256"]
                and reset_epoch["attestation_sha256"] != quiescence_epoch["attestation_sha256"]
            )
        if not lifecycle_valid:
            infra_failures.append(f"{name}:formal_lifecycle_evidence_invalid")
        baseline_rates = [float(leg["valid_steps_per_min"]) for leg in baseline_legs]
        candidate_rates = [float(leg["valid_steps_per_min"]) for leg in candidate_legs]
        baseline_rate = geometric_mean(baseline_rates)
        candidate_rate = geometric_mean(candidate_rates)
        speedup = candidate_rate / baseline_rate if baseline_rate else 0.0
        baseline_rate_floor = float(policy["min_baseline_valid_steps_per_min"])
        reference_speedup = float(profile_policy["reference_speedup"])
        performance = normalized_performance(speedup, reference_speedup, noise_floor)

        if baseline_rate < baseline_rate_floor:
            infra_failures.append(
                f"{name}:baseline_valid_steps_per_min="
                f"{baseline_rate:.4f}<{baseline_rate_floor:.4f}"
            )
        for index, leg in enumerate(legs):
            if leg.get("infrastructure_failure") is not None:
                infra_failures.append(f"{name}:leg{index + 1}:infrastructure_failure")
            if leg.get("candidate_failure") is not None:
                target = infra_failures if leg["mode"] == "baseline" else candidate_failures
                failure_type = str(
                    leg["candidate_failure"].get("type", "unspecified")
                )
                target.append(
                    f"{name}:leg{index + 1}:candidate_failure:{failure_type}"
                )
            if not leg["backend_audit"]["passed"] or not leg["warmup_audit"]["passed"]:
                target = infra_failures if leg["mode"] == "baseline" else candidate_failures
                target.append(f"{name}:leg{index + 1}:backend_audit")
        baseline_success = min(float(leg["request_success_rate"]) for leg in baseline_legs)
        candidate_success = min(float(leg["request_success_rate"]) for leg in candidate_legs)
        if baseline_success < float(policy["min_request_success_rate"]):
            infra_failures.append(f"{name}:baseline_request_success={baseline_success:.4f}")
        pressure = baseline_leg.get("pressure_validation", {})
        if (
            baseline_leg.get("profile_config", {}).get("pressure_required")
            and not pressure.get("valid")
        ):
            reasons = ",".join(str(reason) for reason in pressure.get("reasons", []))
            infra_failures.append(
                f"{name}:leg{baseline_index + 1}:profile_not_pressurized:{reasons}"
            )
        success_floor = max(
            float(policy["min_request_success_rate"]),
            baseline_success - float(policy["request_success_tolerance"]),
        )
        if candidate_success < success_floor:
            candidate_failures.append(f"{name}:request_success={candidate_success:.4f}<{success_floor:.4f}")
        baseline_valid_ratio = min(float(leg["valid_step_ratio"]) for leg in baseline_legs)
        candidate_valid_ratio = min(float(leg["valid_step_ratio"]) for leg in candidate_legs)
        if baseline_valid_ratio < float(policy["min_valid_step_ratio"]):
            infra_failures.append(f"{name}:baseline_valid_step_ratio={baseline_valid_ratio:.4f}")
        valid_ratio_floor = max(
            float(policy["min_valid_step_ratio"]),
            baseline_valid_ratio - float(policy["valid_step_ratio_tolerance"]),
        )
        if candidate_valid_ratio < valid_ratio_floor:
            candidate_failures.append(
                f"{name}:valid_step_ratio={candidate_valid_ratio:.4f}<{valid_ratio_floor:.4f}"
            )

        baseline_progress = min(int(leg["minimum_workflow_progress"]) for leg in baseline_legs)
        minimum_progress = min(int(leg["minimum_workflow_progress"]) for leg in candidate_legs)
        if windowed_profile:
            # Fixed-window throughput intentionally censors all 192 SWE tasks
            # at the trusted cutoff. Completion, tests, and per-task minimum
            # depth are diagnostics, not validity gates.
            semantic_report = {
                "skipped": True,
                "reason": "trusted_fixed_window_does_not_wait_for_task_completion",
                "tests_passed_workflows": {
                    "baseline": [int(leg["tests_passed_workflows"]) for leg in baseline_legs],
                    "candidate": [int(leg["tests_passed_workflows"]) for leg in candidate_legs],
                },
                "completed_workflows": {
                    "baseline": [int(leg["completed_workflows"]) for leg in baseline_legs],
                    "candidate": [int(leg["completed_workflows"]) for leg in candidate_legs],
                },
            }
        else:
            semantic_report = {}
            for metric, minimum_key, tolerance_key in (
                (
                    "tests_passed_workflows",
                    "min_tests_passed_workflows",
                    "tests_passed_workflows_tolerance",
                ),
                (
                    "completed_workflows",
                    "min_completed_workflows",
                    "completed_workflows_tolerance",
                ),
            ):
                absolute_floor = int(policy[minimum_key])
                tolerance = int(policy[tolerance_key])
                baseline_counts = [int(leg[metric]) for leg in baseline_legs]
                candidate_counts = [int(leg[metric]) for leg in candidate_legs]
                candidate_floors = [
                    max(absolute_floor, baseline_count - tolerance)
                    for baseline_count in baseline_counts
                ]
                for leg_index, count in zip((baseline_index,), baseline_counts):
                    if count < absolute_floor:
                        infra_failures.append(
                            f"{name}:leg{leg_index + 1}:baseline_{metric}="
                            f"{count}<{absolute_floor}"
                        )
                for leg_index, count, floor in zip(
                    (candidate_index,), candidate_counts, candidate_floors
                ):
                    if count < floor:
                        candidate_failures.append(
                            f"{name}:leg{leg_index + 1}:{metric}={count}<{floor}"
                        )
                semantic_report[metric] = {
                    "baseline": baseline_counts,
                    "candidate": candidate_counts,
                    "absolute_floor": absolute_floor,
                    "tolerance": tolerance,
                    "paired_candidate_floors": candidate_floors,
                }
            if baseline_progress < int(policy["minimum_workflow_progress"]):
                infra_failures.append(f"{name}:baseline_minimum_progress={baseline_progress}")
            if minimum_progress < int(policy["minimum_workflow_progress"]):
                candidate_failures.append(f"{name}:minimum_progress={minimum_progress}")
        candidate_wait = max(
            float(leg["backend_audit"]["p99_gateway_wait_sec"] or 0.0) for leg in candidate_legs
        )
        if candidate_wait > float(policy["max_p99_gateway_wait_sec"]):
            candidate_failures.append(f"{name}:p99_gateway_wait={candidate_wait:.3f}")

        baseline_p99_raw = baseline_leg.get("p99_step_latency_sec")
        if (
            not isinstance(baseline_p99_raw, (int, float))
            or isinstance(baseline_p99_raw, bool)
            or not math.isfinite(float(baseline_p99_raw))
            or float(baseline_p99_raw) <= 0.0
        ):
            if not windowed_profile:
                infra_failures.append(
                    f"{name}:leg{baseline_index + 1}:invalid_p99_step_latency"
                )
            baseline_p99_step = None
        else:
            baseline_p99_step = float(baseline_p99_raw)
        candidate_p99_raw = candidate_leg.get("p99_step_latency_sec")
        if (
            not isinstance(candidate_p99_raw, (int, float))
            or isinstance(candidate_p99_raw, bool)
            or not math.isfinite(float(candidate_p99_raw))
            or float(candidate_p99_raw) <= 0.0
        ):
            if not windowed_profile:
                candidate_failures.append(
                    f"{name}:leg{candidate_index + 1}:invalid_p99_step_latency"
                )
            candidate_p99_step = None
        else:
            candidate_p99_step = float(candidate_p99_raw)
        p99_step_ratio = (
            candidate_p99_step / baseline_p99_step
            if candidate_p99_step is not None and baseline_p99_step is not None
            else None
        )
        maximum_allowed_p99_step_ratio = float(policy["max_p99_step_latency_ratio"])
        if (
            not windowed_profile
            and
            p99_step_ratio is not None
            and p99_step_ratio > maximum_allowed_p99_step_ratio
        ):
            candidate_failures.append(
                f"{name}:p99_step_latency_ratio={p99_step_ratio:.4f}>"
                f"{maximum_allowed_p99_step_ratio:.4f}"
            )
        tail_anti_gaming = {
            "schema_version": 2,
            "metric": "p99_step_latency_sec",
            "pairing": "candidate_over_baseline:single_committed_pair",
            "baseline_p99_step_latency_sec": baseline_p99_step,
            "candidate_p99_step_latency_sec": candidate_p99_step,
            "ratio": p99_step_ratio,
            "maximum_allowed_ratio": maximum_allowed_p99_step_ratio,
            "skipped": windowed_profile,
            "skip_reason": (
                "trusted_fixed_window_is_right_censored"
                if windowed_profile
                else None
            ),
            "passed": (
                True
                if windowed_profile
                else p99_step_ratio is not None
                and p99_step_ratio <= maximum_allowed_p99_step_ratio
            ),
        }

        baseline_work = geometric_mean(
            float(leg["backend_tokens_per_valid_step"] or 0.0) for leg in baseline_legs
        )
        candidate_work = geometric_mean(
            float(leg["backend_tokens_per_valid_step"] or 0.0) for leg in candidate_legs
        )
        work_ratio = candidate_work / baseline_work if baseline_work else 0.0
        if baseline_work <= 0:
            infra_failures.append(f"{name}:baseline_backend_work={baseline_work:.4f}")
        if not float(policy["backend_work_ratio_min"]) <= work_ratio <= float(policy["backend_work_ratio_max"]):
            candidate_failures.append(f"{name}:backend_work_ratio={work_ratio:.4f}")

        weighted_scores.append((weight, performance))
        tail_values.append(candidate_wait)
        profile_reports[name] = {
            "weight": weight,
            "baseline_rates": baseline_rates,
            "candidate_rates": candidate_rates,
            "baseline_geomean": baseline_rate,
            "candidate_geomean": candidate_rate,
            "speedup": speedup,
            "reference_speedup": reference_speedup,
            "performance_score": performance,
            "minimum_baseline_valid_steps_per_min": baseline_rate_floor,
            "leg_order_commitment": {
                "schema_version": 1,
                "protocol": "sha256-domain-separated-first-byte-lsb-v1",
                "context": FINAL_LEG_ORDER_CONTEXT.decode("ascii").rstrip("\n"),
                "workload_nonce_sha256": workload_nonce_sha256,
                "order": list(expected_order),
            },
            "request_success_floor": success_floor,
            "candidate_request_success": candidate_success,
            "valid_step_ratio_floor": valid_ratio_floor,
            "candidate_valid_step_ratio": candidate_valid_ratio,
            "semantic_quality": semantic_report,
            "baseline_minimum_workflow_progress": baseline_progress,
            "candidate_minimum_workflow_progress": minimum_progress,
            "candidate_p99_gateway_wait_sec": candidate_wait,
            "tail_anti_gaming": tail_anti_gaming,
            "backend_work_ratio": work_ratio,
            "legs": legs,
        }

    if any(score <= 0 for _, score in weighted_scores):
        performance_score = 0.0
    else:
        total_weight = sum(weight for weight, _ in weighted_scores)
        performance_score = math.exp(
            sum(weight * math.log(score) for weight, score in weighted_scores) / total_weight
        )
    tail_component = min(
        1.0,
        float(policy["tail_target_sec"]) / max(max(tail_values), float(policy["tail_target_sec"])),
    )
    tail_score = tail_component
    hard_gates = not infra_failures and not candidate_failures
    # Tail behavior remains a quality multiplier on achieved optimization, not
    # a bonus for an otherwise unoptimized passthrough. Starvation is enforced
    # by progress/completion hard gates rather than distribution-equality scoring.
    reward = (
        min(1.0, 0.20 + performance_score * (0.70 + 0.10 * tail_score))
        if hard_gates
        else 0.0
    )
    status = "infra_invalid" if infra_failures else ("candidate_failed" if candidate_failures else "ok")
    return {
        "status": status,
        "hard_gates_passed": hard_gates,
        "infra_failures": infra_failures,
        "candidate_failures": candidate_failures,
        "performance_score": performance_score,
        "tail_score": tail_score,
        "reward": reward,
        "profiles": profile_reports,
    }


def retryable_measurement_failure(scored: dict[str, Any]) -> bool:
    """Keep the frozen hidden protocol to one concealed A/B attempt.

    Selectively retrying only noisy outcomes would alter the frozen measurement
    distribution and disclose the same held-out profile again.
    """

    return False


def final_next_leg(session: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any] | None:
    if session["profile_index"] >= len(policy["hidden_profiles"]):
        return None
    if session.get("pending_challenge") is None:
        session["pending_challenge"] = secrets.token_hex(32)
    profile_name = policy["hidden_profiles"][session["profile_index"]]["name"]
    workload_nonce_sha256 = hashlib.sha256(
        session["profile_nonces"][profile_name].encode()
    ).hexdigest()
    order = formal_leg_order(workload_nonce_sha256)
    return {
        # The root verifier needs only an opaque position and lifecycle action;
        # hidden profile names and workload nonces never enter candidate state.
        "leg_ordinal": len(session["lifecycle_epochs"]) + 1,
        "mode": order[session["leg_index"]],
        "challenge": session["pending_challenge"],
    }


def build_terminal_failure_report(
    app: web.Application,
    session: dict[str, Any],
    *,
    profile_name: str,
    mode: str,
    leg_ordinal: int,
    lifecycle_epoch: dict[str, Any],
    failure: ClassifiedLegFailure,
) -> dict[str, Any]:
    """Persist a typed terminal result instead of leaking a candidate error as 500."""

    policy = app["policy"]
    epochs = [*session["lifecycle_epochs"], lifecycle_epoch]
    attempts = {
        name: list(items) for name, items in session["measurement_attempts"].items()
    }
    attempts[profile_name] = [
        {
            "attempt": 1,
            "status": failure.classification,
            "infra_failures": (
                [f"trusted_controller:{failure.failure_type}"]
                if failure.classification == "infra_invalid"
                else []
            ),
            "candidate_failures": (
                [f"trusted_controller:{failure.failure_type}"]
                if failure.classification == "candidate_failed"
                else []
            ),
            "retryable": False,
            "run_ids": [
                leg["run_id"]
                for leg in session["profile_legs"].get(profile_name, [])
            ],
            "failed_leg_ordinal": leg_ordinal,
            "failed_leg_mode": mode,
        }
    ]
    scored = classified_failure_fields(
        failure.classification,
        failure.failure_type,
        failure.detail,
    )
    scored["failure_attribution"].update(
        {
            "profile": profile_name,
            "leg_ordinal": leg_ordinal,
            "mode": mode,
        }
    )
    report = {
        "schema_version": 2,
        "release_id": policy["release_id"],
        "policy_contract_sha256": policy_contract_sha256(policy),
        "controller_config_sha256": app["config_sha256"],
        "corpus_manifest_sha256": file_sha256(CORPUS_MANIFEST),
        "suite_nonce_sha256": hashlib.sha256(session["suite_nonce"].encode()).hexdigest(),
        "measurement_attempts": attempts,
        "recovered_transient_measurements": [],
        "candidate_lifecycle": {
            "schema_version": 1,
            "protocol": "trusted-verifier-challenge-hmac-v1",
            "frozen_submission_sha256": session["frozen_submission_sha256"],
            # final_session_begin always provisions the signed seal. Retain an
            # explicit false sentinel for isolated report-builder tests and
            # for any future trusted internal caller that violates the session
            # constructor invariant; such a terminal report already scores 0.
            "network_egress_seal": session.get("network_egress_seal")
            or {
                "schema_version": 1,
                "protocol": "missing-network-egress-seal",
                "sealed": False,
            },
            "epoch_count": len(epochs),
            "candidate_process_epoch_count": sum(
                epoch["mode"] == "candidate" for epoch in epochs
            ),
            "epochs": epochs,
        },
        **scored,
    }
    persist_trusted_result(
        "final-failure",
        f"{report['suite_nonce_sha256'][:16]}-{leg_ordinal}",
        report,
    )
    return report


def build_final_report(app: web.Application, session: dict[str, Any]) -> dict[str, Any]:
    policy = app["policy"]
    profile_legs = session["profile_legs"]
    scored = score_final(policy, profile_legs)
    final_legs = [leg for legs in profile_legs.values() for leg in legs]
    if not final_legs:
        raise RuntimeError("final benchmark produced no legs")
    lifecycle_epochs = session["lifecycle_epochs"]
    if (
        len(final_legs) != FINAL_FORMAL_LEG_COUNT
        or len(lifecycle_epochs) != FINAL_FORMAL_LEG_COUNT
        or [epoch.get("leg_ordinal") for epoch in lifecycle_epochs] != [1, 2]
        or [epoch.get("mode") for epoch in lifecycle_epochs]
        != [leg.get("mode") for leg in final_legs]
        or sum(epoch.get("mode") == "candidate" for epoch in lifecycle_epochs) != 1
    ):
        raise RuntimeError("formal hidden lifecycle is not one committed two-leg pair")
    shared_model_provenance = require_single_model_provenance(
        final_legs,
        context="final benchmark",
    )
    report = {
        "schema_version": 2,
        "release_id": policy["release_id"],
        "policy_contract_sha256": policy_contract_sha256(policy),
        "controller_config_sha256": app["config_sha256"],
        "corpus_manifest_sha256": file_sha256(CORPUS_MANIFEST),
        "model_profile_sha256": shared_model_provenance["profile_sha256"],
        "model_provenance": shared_model_provenance,
        "suite_nonce_sha256": hashlib.sha256(session["suite_nonce"].encode()).hexdigest(),
        "measurement_attempts": session["measurement_attempts"],
        "recovered_transient_measurements": [],
        "candidate_lifecycle": {
            "schema_version": 1,
            "protocol": "trusted-verifier-challenge-hmac-v1",
            "frozen_submission_sha256": session["frozen_submission_sha256"],
            "network_egress_seal": session["network_egress_seal"],
            "epoch_count": len(lifecycle_epochs),
            "candidate_process_epoch_count": sum(
                epoch["mode"] == "candidate" for epoch in lifecycle_epochs
            ),
            "epochs": lifecycle_epochs,
        },
        **scored,
    }
    persist_trusted_result("final", report["suite_nonce_sha256"][:16], report)
    return report


async def final_network_seal_challenge(request: web.Request) -> web.Response:
    """Issue no held-out data, only a one-use verifier/worker seal challenge."""

    require_admin(request)
    if request.app["lease"].locked() or request.app.get("final_session") is not None:
        return web.json_response({"error": "benchmark_lease_busy"}, status=409)
    challenge = secrets.token_hex(32)
    request.app["pending_network_seal_challenge"] = {
        "challenge": challenge,
        "expires_monotonic": time.monotonic() + MAIN_EGRESS_CHALLENGE_TTL_SEC,
    }
    return web.json_response(
        {
            "schema_version": 1,
            "challenge": challenge,
            "expires_in_sec": MAIN_EGRESS_CHALLENGE_TTL_SEC,
        }
    )


async def final_session_begin(request: web.Request) -> web.Response:
    require_admin(request)
    policy = request.app["policy"]
    if not policy.get("release_ready"):
        return web.json_response(
            {"error": "benchmark_not_ready", "release_id": policy.get("release_id")}, status=409
        )
    if request.app["lease"].locked() or request.app.get("final_session") is not None:
        return web.json_response({"error": "benchmark_lease_busy"}, status=409)
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": "invalid_json"}, status=400)
    snapshot_sha256 = payload.get("frozen_submission_sha256")
    if not _hex_text(snapshot_sha256, 64):
        return web.json_response({"error": "invalid_frozen_submission_sha256"}, status=400)
    pending_seal = request.app.get("pending_network_seal_challenge")
    # Consume the challenge on every begin attempt. A rejected or expired
    # attestation must obtain a fresh challenge and cannot be replayed.
    request.app["pending_network_seal_challenge"] = None
    if not isinstance(pending_seal, dict):
        return web.json_response({"error": "network_egress_seal_required"}, status=409)
    challenge = pending_seal.get("challenge")
    expires_monotonic = pending_seal.get("expires_monotonic")
    if (
        not _hex_text(challenge, 64)
        or isinstance(expires_monotonic, bool)
        or not isinstance(expires_monotonic, (int, float))
        or time.monotonic() > float(expires_monotonic)
    ):
        return web.json_response({"error": "network_egress_seal_challenge_expired"}, status=409)
    try:
        network_egress_seal = validate_main_egress_seal_attestation(
            payload.get("network_egress_attestation"),
            expected_challenge=challenge,
            admin_token=token_text(),
            seen_nonces=request.app["seen_network_seal_nonces"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return web.json_response(
            {"error": "network_egress_attestation_rejected", "detail": str(exc)},
            status=409,
        )
    profiles: dict[str, Profile] = request.app["profile_sets"]["hidden"]
    for profile_policy in policy["hidden_profiles"]:
        name = profile_policy["name"]
        if name not in profiles or not profiles[name].enabled:
            return web.json_response({"error": "hidden_profile_unavailable"}, status=500)
    try:
        session_model_provenance = await model_provenance(request.app)
    except Exception:
        return web.json_response(
            {"error": "backend_serving_contract_unavailable"},
            status=503,
        )
    session = {
        "session_id": secrets.token_hex(32),
        "suite_nonce": secrets.token_hex(16),
        "frozen_submission_sha256": snapshot_sha256,
        "network_egress_seal": network_egress_seal,
        "model_provenance": session_model_provenance,
        "profile_index": 0,
        "leg_index": 0,
        "profile_legs": {},
        "profile_nonces": {
            entry["name"]: secrets.token_hex(16) for entry in policy["hidden_profiles"]
        },
        "measurement_attempts": {},
        "lifecycle_epochs": [],
        "seen_epoch_nonces": set(),
        "seen_candidate_process_identities": set(),
        "pending_challenge": None,
        "in_progress": False,
        "failed": None,
    }
    request.app["final_session"] = session
    return web.json_response(
        {"session_id": session["session_id"], "next_leg": final_next_leg(session, policy)}
    )


async def final_session_leg(request: web.Request) -> web.Response:
    require_admin(request)
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": "invalid_json"}, status=400)
    session = request.app.get("final_session")
    if session is None or not secrets.compare_digest(
        str(payload.get("session_id", "")), str(session.get("session_id", ""))
    ):
        return web.json_response({"error": "unknown_final_session"}, status=404)
    if session["in_progress"] or request.app["lease"].locked():
        return web.json_response({"error": "benchmark_lease_busy"}, status=409)
    if session["failed"] is not None:
        return web.json_response({"error": "final_session_failed", "detail": session["failed"]}, status=409)
    policy = request.app["policy"]
    next_leg = final_next_leg(session, policy)
    if next_leg is None:
        return web.json_response({"error": "final_session_already_complete"}, status=409)
    try:
        lifecycle_epoch = validate_lifecycle_attestation(
            session,
            payload.get("lifecycle_attestation"),
            expected_mode=next_leg["mode"],
            expected_leg_ordinal=next_leg["leg_ordinal"],
            admin_token=token_text(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        session["failed"] = f"{type(exc).__name__}: {exc}"
        return web.json_response(
            {"error": "candidate_lifecycle_attestation_rejected", "detail": str(exc)}, status=409
        )

    profile_policy = policy["hidden_profiles"][session["profile_index"]]
    profile_name = profile_policy["name"]
    profile: Profile = request.app["profile_sets"]["hidden"][profile_name]
    profile_nonce = session["profile_nonces"][profile_name]
    mode = next_leg["mode"]
    session["in_progress"] = True
    session["pending_challenge"] = None
    try:
        if mode == "baseline" and await candidate_port_open():
            raise infrastructure_leg_failure(
                "baseline_candidate_port_open",
                "baseline leg refused while candidate port 9000 is open",
            )
        async with request.app["lease"]:
            result = await execute_leg(
                request.app,
                profile=profile,
                mode=mode,
                corpus_namespace=f"heldout-{profile_nonce}",
                workload_nonce=profile_nonce,
                expected_model_provenance=session["model_provenance"],
            )
            if not same_backend_serving_contract(
                result.get("model_provenance"), session["model_provenance"]
            ):
                raise infrastructure_leg_failure(
                    "model_provenance_changed_across_final_session",
                    "backend serving contract changed between hidden legs",
                )
    except ClassifiedLegFailure as failure:
        session["in_progress"] = False
        try:
            report = build_terminal_failure_report(
                request.app,
                session,
                profile_name=profile_name,
                mode=mode,
                leg_ordinal=next_leg["leg_ordinal"],
                lifecycle_epoch=lifecycle_epoch,
                failure=failure,
            )
        except Exception as report_exc:
            session["failed"] = f"{type(report_exc).__name__}: {report_exc}"
            return web.json_response(
                {
                    "error": "trusted_failure_report_failed",
                    "detail": session["failed"],
                },
                status=500,
            )
        request.app["final_session"] = None
        return web.json_response({"done": True, "report": report})
    except Exception as exc:
        # Candidate-controlled endpoint failures are converted inside
        # execute_leg.  Anything reaching this branch is a controller/lifecycle
        # defect and is explicitly attributed to trusted infrastructure.
        session["in_progress"] = False
        failure = infrastructure_leg_failure(
            "trusted_controller_internal_failure",
            f"{type(exc).__name__}: {exc}",
        )
        try:
            report = build_terminal_failure_report(
                request.app,
                session,
                profile_name=profile_name,
                mode=mode,
                leg_ordinal=next_leg["leg_ordinal"],
                lifecycle_epoch=lifecycle_epoch,
                failure=failure,
            )
        except Exception as report_exc:
            session["failed"] = f"{type(report_exc).__name__}: {report_exc}"
            return web.json_response(
                {
                    "error": "trusted_failure_report_failed",
                    "detail": session["failed"],
                },
                status=500,
            )
        request.app["final_session"] = None
        return web.json_response({"done": True, "report": report})
    session["in_progress"] = False
    if mode == "candidate":
        result["candidate_reset_epoch"] = lifecycle_epoch
    else:
        result["candidate_quiescence_epoch"] = lifecycle_epoch
    session["lifecycle_epochs"].append(lifecycle_epoch)
    legs = session["profile_legs"].setdefault(profile_name, [])
    legs.append(result)
    session["leg_index"] += 1
    if session["leg_index"] == FINAL_FORMAL_LEG_COUNT:
        single_policy = {**policy, "hidden_profiles": [profile_policy]}
        attempt_score = score_final(single_policy, {profile_name: legs})
        session["measurement_attempts"][profile_name] = [
            {
                "attempt": 1,
                "status": attempt_score["status"],
                "infra_failures": attempt_score["infra_failures"],
                "candidate_failures": attempt_score["candidate_failures"],
                "retryable": retryable_measurement_failure(attempt_score),
                "run_ids": [leg["run_id"] for leg in legs],
            }
        ]
        session["profile_index"] += 1
        session["leg_index"] = 0
    next_leg = final_next_leg(session, policy)
    if next_leg is not None:
        return web.json_response({"done": False, "next_leg": next_leg})
    try:
        report = build_final_report(request.app, session)
    except Exception as exc:
        session["failed"] = f"{type(exc).__name__}: {exc}"
        return web.json_response(
            {"error": "final_scoring_failed", "detail": session["failed"]}, status=500
        )
    request.app["final_session"] = None
    return web.json_response({"done": True, "report": report})


async def final_session_abort(request: web.Request) -> web.Response:
    require_admin(request)
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": "invalid_json"}, status=400)
    session = request.app.get("final_session")
    if session is None:
        return web.json_response({"aborted": False, "detail": "no_active_session"})
    if not secrets.compare_digest(
        str(payload.get("session_id", "")), str(session.get("session_id", ""))
    ):
        return web.json_response({"error": "unknown_final_session"}, status=404)
    if session["in_progress"] or request.app["lease"].locked():
        return web.json_response({"error": "benchmark_lease_busy"}, status=409)
    request.app["final_session"] = None
    return web.json_response({"aborted": True})


async def final_run(request: web.Request) -> web.Response:
    """Reject the legacy stateful endpoint even after a policy is released."""

    require_admin(request)
    policy = request.app["policy"]
    if not policy.get("release_ready"):
        return web.json_response(
            {"error": "benchmark_not_ready", "release_id": policy.get("release_id")}, status=409
        )
    return web.json_response(
        {
            "error": "candidate_lifecycle_reset_required",
            "detail": "use the trusted final-session challenge protocol",
        },
        status=409,
    )


async def on_startup(app: web.Application) -> None:
    FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)
    FEEDBACK_ROOT.chmod(0o700)
    TRUSTED_RESULTS_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    TRUSTED_RESULTS_ROOT.chmod(0o700)
    app["client"] = ClientSession(timeout=ClientTimeout(total=30))
    try:
        await load_release_public_direct_anchor(app)
    except Exception:
        await app["client"].close()
        raise


async def on_cleanup(app: web.Application) -> None:
    await app["client"].close()


def make_app() -> web.Application:
    paths = [PUBLIC_PROFILES, CALIBRATION_PROFILES, HIDDEN_PROFILES, SCORING_POLICY]
    profile_sets = {
        # The promoted public, author-calibration, and held-out profiles all use
        # the frozen 192-sequence Qwen transport ceiling. The generic loader
        # retains its conservative default for callers that do not opt in.
        "public": load_profiles(PUBLIC_PROFILES, concurrency_limit=192),
        "calibration": load_profiles(CALIBRATION_PROFILES, concurrency_limit=192),
        "hidden": load_profiles(HIDDEN_PROFILES, concurrency_limit=192),
    }
    policy = json.loads(SCORING_POLICY.read_text())
    validate_policy(policy, profile_sets["hidden"])
    model_adapter = load_model_adapter_contract().summary()
    digest_parts = [f"{path.name}:{file_sha256(path)}" for path in paths]
    digest_parts.append(
        "model-adapter:" + json.dumps(model_adapter, separators=(",", ":"), sort_keys=True)
    )
    digest_payload = "\n".join(digest_parts).encode()
    app = web.Application(
        client_max_size=1024**2,
        middlewares=[sanitize_public_exceptions],
    )
    app["profile_sets"] = profile_sets
    app["policy"] = policy
    app["model_adapter"] = model_adapter
    app["config_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    app["lease"] = asyncio.Lock()
    app["final_session"] = None
    app["pending_network_seal_challenge"] = None
    app["seen_network_seal_nonces"] = set()
    app["history"] = []
    app["public_pair"] = {}
    app["release_public_anchor_file"] = os.environ.get(
        RELEASE_PUBLIC_DIRECT_ANCHOR_FILE_ENV, ""
    ).strip()
    app["release_public_anchor_expected_sha256"] = os.environ.get(
        RELEASE_PUBLIC_DIRECT_ANCHOR_SHA256_ENV, ""
    ).strip()
    app["release_public_anchor_material_json"] = None
    app["release_public_anchor_pair_id"] = None
    app["release_public_anchor_required"] = (
        release_public_direct_anchor_required_from_env()
    )
    app.router.add_get("/health", health)
    app.router.add_get("/public/doctor", doctor)
    app.router.add_post("/public/run", public_run)
    app.router.add_get("/public/compare", compare)
    app.router.add_get("/admin/readiness", readiness)
    app.router.add_post("/admin/run", admin_run)
    app.router.add_post(
        "/admin/final/network-seal/challenge", final_network_seal_challenge
    )
    app.router.add_post("/admin/final/session/begin", final_session_begin)
    app.router.add_post("/admin/final/session/leg", final_session_leg)
    app.router.add_post("/admin/final/session/abort", final_session_abort)
    app.router.add_post("/admin/final/run", final_run)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=7000, access_log=None)
