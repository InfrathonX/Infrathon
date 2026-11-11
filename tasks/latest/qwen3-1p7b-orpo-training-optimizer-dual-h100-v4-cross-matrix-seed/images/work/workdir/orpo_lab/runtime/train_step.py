"""Trusted full-model training-step construction for the public GPU smoke probe.

This module is task-owned. Candidate code implements the objective ABI and may
optionally supply a declarative ``Qwen3ModelPatch``. The runtime owns patch
installation; candidates cannot choose the model path, batch, optimizer, step
count, timing boundary, or token accounting used by the probe.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
from torch import nn

from .api import (
    ModelOptimizer,
    OFFICIAL_TRL_IGNORE_INDEX,
    ORPOResult,
    ObjectiveContract,
    Qwen3ModelPatch,
    build_official_trl_objective,
    normalize_result,
)
from .loader import load_objective


MODEL_REPO_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_TOTAL_SIZE = 4_063_479_808
MODEL_UNIQUE_PARAMETERS = 1_720_574_976
EXPECTED_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}

# Resolve topology readers before candidate import.  Canonical source
# admission also rejects external/nested attribute writes, while these handles
# keep hook validation independent of later module-attribute lookup.
_TRUSTED_NAMED_PARAMETERS = nn.Module.named_parameters
_TRUSTED_NAMED_BUFFERS = nn.Module.named_buffers
_TRUSTED_METHOD_TYPE = MethodType


@dataclass(frozen=True)
class _TensorTopology:
    name: str
    tensor: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    value: torch.Tensor | None


@dataclass(frozen=True)
class _ModelTopology:
    model: nn.Module
    parameters: tuple[_TensorTopology, ...]
    buffers: tuple[_TensorTopology, ...]
    input_embedding_weight: torch.Tensor
    output_embedding_weight: torch.Tensor
    training: bool
    use_cache: object


def _tensor_topology(
    entries,
    *,
    clone_values: bool = False,
) -> tuple[_TensorTopology, ...]:
    return tuple(
        _TensorTopology(
            name=name,
            tensor=tensor,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=bool(tensor.requires_grad),
            value=tensor.detach().clone() if clone_values else None,
        )
        for name, tensor in entries
    )


def _snapshot_model_topology(model: nn.Module) -> _ModelTopology:
    try:
        parameters = _TRUSTED_NAMED_PARAMETERS(
            model,
            recurse=True,
            remove_duplicate=False,
        )
        buffers = _TRUSTED_NAMED_BUFFERS(
            model,
            recurse=True,
            remove_duplicate=False,
        )
    except TypeError as exc:  # pragma: no cover - pinned PyTorch supports this
        raise RuntimeError("pinned PyTorch must expose duplicate-aware model traversal") from exc
    try:
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        use_cache = model.config.use_cache
    except AttributeError as exc:
        raise ValueError("model optimizer requires causal-LM embeddings and use_cache") from exc
    if input_weight is not output_weight:
        raise ValueError("model optimizer boundary requires tied input/output weights")
    return _ModelTopology(
        model=model,
        parameters=_tensor_topology(parameters),
        buffers=_tensor_topology(buffers, clone_values=True),
        input_embedding_weight=input_weight,
        output_embedding_weight=output_weight,
        training=bool(model.training),
        use_cache=use_cache,
    )


def _validate_tensor_topology(
    before: tuple[_TensorTopology, ...],
    after: tuple[_TensorTopology, ...],
    *,
    kind: str,
    validate_values: bool = False,
) -> None:
    before_names = tuple(item.name for item in before)
    after_names = tuple(item.name for item in after)
    if before_names != after_names:
        raise ValueError(f"model optimizer changed {kind} names or alias topology")
    for expected, actual in zip(before, after, strict=True):
        if actual.tensor is not expected.tensor:
            raise ValueError(f"model optimizer replaced {kind} {expected.name!r}")
        if actual.shape != expected.shape:
            raise ValueError(f"model optimizer changed {kind} shape for {expected.name!r}")
        if actual.dtype != expected.dtype:
            raise ValueError(f"model optimizer changed {kind} dtype for {expected.name!r}")
        if actual.device != expected.device:
            raise ValueError(f"model optimizer moved {kind} {expected.name!r}")
        if actual.requires_grad != expected.requires_grad:
            raise ValueError(
                f"model optimizer changed {kind} requires_grad for {expected.name!r}"
            )
        if validate_values and (
            expected.value is None
            or actual.value is None
            or not torch.equal(actual.value, expected.value)
        ):
            raise ValueError(f"model optimizer changed {kind} value for {expected.name!r}")


def _apply_qwen3_model_patch(
    model: nn.Module,
    patch: Qwen3ModelPatch,
) -> dict[str, object]:
    """Install candidate callables using task-owned attribute writes only."""

    from transformers.models.qwen3 import modeling_qwen3

    try:
        base_model = model.model
        layers = tuple(base_model.layers)
        norm_modules = [base_model.norm]
        mlp_modules = []
        for layer in layers:
            norm_modules.extend(
                (
                    layer.input_layernorm,
                    layer.post_attention_layernorm,
                    layer.self_attn.q_norm,
                    layer.self_attn.k_norm,
                )
            )
            mlp_modules.append(layer.mlp)
    except AttributeError as exc:
        raise ValueError("Qwen3ModelPatch requires the pinned Qwen3 module layout") from exc

    applied: list[str] = []
    if patch.rope is not None:
        modeling_qwen3.apply_rotary_pos_emb = patch.rope
        applied.append("rope")
    if patch.rms_norm_forward is not None:
        for module in norm_modules:
            module.offset = 0.0
            module.casting_mode = "llama"
            module.variance_epsilon = (
                getattr(module, "variance_epsilon", None)
                or getattr(module, "eps", None)
                or 1e-6
            )
            module.in_place = True
            module.row_mode = None
            module.forward = _TRUSTED_METHOD_TYPE(patch.rms_norm_forward, module)
        applied.append("rms_norm_forward")
    if patch.swiglu_forward is not None:
        for module in mlp_modules:
            module.forward = _TRUSTED_METHOD_TYPE(patch.swiglu_forward, module)
        applied.append("swiglu_forward")
    return {
        "exported": True,
        "applied": True,
        "fields": applied,
        "norm_module_count": len(norm_modules),
        "mlp_module_count": len(mlp_modules),
        "attribute_writes_owned_by": "task_runtime",
    }


def apply_model_optimizer(
    model: nn.Module,
    optimizer: ModelOptimizer | None,
    contract: ObjectiveContract,
    model_patch: Qwen3ModelPatch | None = None,
) -> dict[str, object]:
    """Install the declarative patch and validate all model-state invariants.

    ``optimizer`` is retained only for old host-side calibration archives.
    Participant admission rejects that export, and participant-visible
    telemetry describes only the supported declarative ABI.
    """

    if optimizer is None and model_patch is None:
        return {
            "abi": "Qwen3ModelPatch",
            "exported": False,
            "invoked": False,
            "return": "not_exported",
            "invariants_validated": False,
            "patch_spec": {"exported": False, "applied": False},
        }

    before = _snapshot_model_topology(model)
    patch_report = (
        {"exported": False, "applied": False}
        if model_patch is None
        else _apply_qwen3_model_patch(model, model_patch)
    )
    returned = None if optimizer is None else optimizer(model, contract)
    if returned is not None and returned is not model:
        raise TypeError("optimize_model must return the same model object or None")
    after = _snapshot_model_topology(model)
    if after.model is not before.model:
        raise ValueError("model optimizer replaced the root model")
    _validate_tensor_topology(before.parameters, after.parameters, kind="parameter")
    _validate_tensor_topology(
        before.buffers,
        after.buffers,
        kind="buffer",
        validate_values=True,
    )
    if (
        after.input_embedding_weight is not before.input_embedding_weight
        or after.output_embedding_weight is not before.output_embedding_weight
        or after.input_embedding_weight is not after.output_embedding_weight
    ):
        raise ValueError("model optimizer changed tied input/output weight identity")
    if after.training != before.training:
        raise ValueError("model optimizer changed model training mode")
    if after.use_cache != before.use_cache:
        raise ValueError("model optimizer changed config.use_cache")
    return {
        "abi": "Qwen3ModelPatch",
        "exported": True,
        "invoked": True,
        "return": (
            "patch_spec_only"
            if optimizer is None
            else ("none" if returned is None else "same_model")
        ),
        "patch_spec": patch_report,
        "invariants_validated": True,
        "invariants": {
            "root_model_identity": True,
            "parameter_names_identity_shape_dtype_device_requires_grad": True,
            "buffer_names_identity_shape_dtype_device_requires_grad_value": True,
            "tied_input_output_weight_identity": True,
            "training_mode": True,
            "config_use_cache": True,
        },
    }


@dataclass(frozen=True)
class SyntheticPreferenceBatch:
    """A task-owned, already-tokenized preference batch.

    The historical class name is retained for compatibility with the
    synthetic probe.  Frozen TLDR batches use the same immutable carrier and
    identify their provenance through ``source`` and the artifact fields.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    scored_tokens: int
    non_padding_tokens: int
    fingerprint: str
    source: str
    record_indices: tuple[int, ...] = ()
    data_partition: str | None = None
    manifest_sha256: str | None = None
    tensors_sha256: str | None = None

    def to(self, device: torch.device) -> "SyntheticPreferenceBatch":
        return SyntheticPreferenceBatch(
            input_ids=self.input_ids.to(device=device, non_blocking=False),
            attention_mask=self.attention_mask.to(device=device, non_blocking=False),
            labels=self.labels.to(device=device, non_blocking=False),
            scored_tokens=self.scored_tokens,
            non_padding_tokens=self.non_padding_tokens,
            fingerprint=self.fingerprint,
            source=self.source,
            record_indices=self.record_indices,
            data_partition=self.data_partition,
            manifest_sha256=self.manifest_sha256,
            tensors_sha256=self.tensors_sha256,
        )


@dataclass(frozen=True)
class FrozenPreferenceDataset:
    """Validated, pinned preference tensors held on CPU.

    The on-disk layout is ``all_chosen_then_all_rejected``.  Batches are
    materialized as chosen rows followed by their matching rejected rows so
    candidate code never controls pairing or ordering.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    partition: str
    manifest_sha256: str
    tensors_sha256: str
    record_count: int
    sequence_length: int
    non_padding_tokens: int
    scored_tokens: int

    def required_records(self, *, total_steps: int, pair_batch: int, world_size: int) -> int:
        if total_steps < 1 or pair_batch < 1 or world_size < 1:
            raise ValueError("steps, pair batch, and world size must be positive")
        return total_steps * pair_batch * world_size

    def require_capacity(self, *, total_steps: int, pair_batch: int, world_size: int) -> None:
        required = self.required_records(
            total_steps=total_steps,
            pair_batch=pair_batch,
            world_size=world_size,
        )
        if required > self.record_count:
            raise ValueError(
                "frozen preference partition has insufficient records: "
                f"need {required} (= {total_steps} steps * {world_size} ranks * "
                f"{pair_batch} pairs), found {self.record_count}"
            )

    def batch_for(
        self,
        *,
        rank: int,
        global_step: int,
        pair_batch: int,
        world_size: int,
        ignore_index: int = OFFICIAL_TRL_IGNORE_INDEX,
    ) -> SyntheticPreferenceBatch:
        """Select a disjoint contiguous record range for one step and rank."""

        if rank not in range(world_size):
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        if ignore_index != OFFICIAL_TRL_IGNORE_INDEX:
            raise ValueError("official TRL batch labels require ignore_index=-100")
        if global_step < 0 or pair_batch < 1:
            raise ValueError("global step must be non-negative and pair batch positive")
        start = (global_step * world_size + rank) * pair_batch
        stop = start + pair_batch
        if stop > self.record_count:
            raise ValueError(
                f"frozen preference record selection [{start}, {stop}) exceeds "
                f"partition size {self.record_count}"
            )
        record_indices = tuple(range(start, stop))
        chosen_rows = torch.arange(start, stop, dtype=torch.long)
        rejected_rows = chosen_rows + self.record_count
        rows = torch.cat((chosen_rows, rejected_rows), dim=0)

        # Normalize integer types only after safe deserialization and
        # validation.  Qwen embeddings and cross entropy both consume int64.
        input_ids = self.input_ids.index_select(0, rows).to(dtype=torch.long)
        attention_mask = self.attention_mask.index_select(0, rows).to(dtype=torch.long)
        labels = self.labels.index_select(0, rows).to(dtype=torch.long)
        scored_tokens = int(labels[:, 1:].ne(ignore_index).sum().item())
        non_padding_tokens = int(attention_mask.sum().item())
        if scored_tokens <= 0 or non_padding_tokens <= 0:
            raise AssertionError("validated frozen batch unexpectedly has no task-owned tokens")

        digest = hashlib.sha256()
        digest.update(b"mlsysforge:frozen-preference-batch:v1\0")
        digest.update(self.partition.encode("utf-8"))
        digest.update(self.manifest_sha256.encode("ascii"))
        digest.update(self.tensors_sha256.encode("ascii"))
        digest.update(json.dumps(record_indices, separators=(",", ":")).encode("ascii"))
        for tensor in (input_ids, attention_mask, labels):
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.contiguous().numpy().tobytes())
        return SyntheticPreferenceBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            scored_tokens=scored_tokens,
            non_padding_tokens=non_padding_tokens,
            fingerprint=digest.hexdigest(),
            source="frozen_tldr",
            record_indices=record_indices,
            data_partition=self.partition,
            manifest_sha256=self.manifest_sha256,
            tensors_sha256=self.tensors_sha256,
        )


def _normalized_sha256(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field} must be exactly 64 hexadecimal characters")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_file(root: Path, name: str) -> Path:
    candidate = root / name
    if candidate.is_symlink():
        raise ValueError(f"frozen data file cannot be a symlink: {name}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != root or not resolved.is_file():
        raise ValueError(f"frozen data file must be a regular direct child: {name}")
    return resolved


def load_frozen_preference_dataset(
    data_path: Path,
    *,
    expected_partition: str,
    expected_manifest_sha256: str,
    expected_tensors_sha256: str,
    ignore_index: int = OFFICIAL_TRL_IGNORE_INDEX,
) -> FrozenPreferenceDataset:
    """Hash, safely deserialize, and structurally validate a frozen partition."""

    root = data_path.expanduser().resolve(strict=True)
    if ignore_index != OFFICIAL_TRL_IGNORE_INDEX:
        raise ValueError("official TRL frozen labels require ignore_index=-100")
    if not root.is_dir():
        raise ValueError(f"frozen data path is not a directory: {root}")
    if expected_partition not in {"public", "private"}:
        raise ValueError("expected data partition must be 'public' or 'private'")
    expected_manifest = _normalized_sha256(
        expected_manifest_sha256, field="expected manifest sha256"
    )
    expected_tensors = _normalized_sha256(
        expected_tensors_sha256, field="expected tensors sha256"
    )
    manifest_path = _artifact_file(root, "manifest.json")
    tensors_path = _artifact_file(root, "tensors.safetensors")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("frozen data manifest exceeds the 1 MiB safety bound")

    actual_manifest = _sha256_file(manifest_path)
    if not hmac.compare_digest(actual_manifest, expected_manifest):
        raise ValueError(
            f"frozen manifest sha256 mismatch: {actual_manifest} != {expected_manifest}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen data manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("frozen data manifest must be a schema_version 1 object")
    if manifest.get("partition") != expected_partition:
        raise ValueError(
            f"frozen data partition mismatch: {manifest.get('partition')!r} != "
            f"{expected_partition!r}"
        )

    tensors_entry = manifest.get("artifacts", {}).get("tensors", {})
    if not isinstance(tensors_entry, dict) or tensors_entry.get("path") != "tensors.safetensors":
        raise ValueError("manifest must name the direct tensors.safetensors artifact")
    actual_tensors = _sha256_file(tensors_path)
    manifest_tensors = tensors_entry.get("sha256")
    if not isinstance(manifest_tensors, str):
        raise ValueError("manifest tensors sha256 is missing")
    manifest_tensors = _normalized_sha256(manifest_tensors, field="manifest tensors sha256")
    if not hmac.compare_digest(actual_tensors, expected_tensors) or not hmac.compare_digest(
        actual_tensors, manifest_tensors
    ):
        raise ValueError(
            "frozen tensors sha256 differs from the expected and/or manifest value"
        )
    if tensors_entry.get("file_size_bytes") != tensors_path.stat().st_size:
        raise ValueError("frozen tensors file size differs from manifest")

    record_count = manifest.get("record_count")
    sequence_count = manifest.get("sequence_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count <= 0
        or sequence_count != 2 * record_count
    ):
        raise ValueError("manifest record/sequence counts are invalid")
    record_ids = manifest.get("record_ids")
    if (
        not isinstance(record_ids, list)
        or len(record_ids) != record_count
        or any(not isinstance(value, str) or not value for value in record_ids)
        or len(set(record_ids)) != record_count
    ):
        raise ValueError("manifest record IDs must be unique and match record_count")

    provenance = manifest.get("provenance", {})
    preprocessing = provenance.get("preprocessing", {}) if isinstance(provenance, dict) else {}
    tokenizer = provenance.get("tokenizer", {}) if isinstance(provenance, dict) else {}
    max_length = preprocessing.get("max_length") if isinstance(preprocessing, dict) else None
    if max_length not in {512, 1024, 2048}:
        raise ValueError("frozen data max_length is outside the V4 workload contract")
    expected_preprocessing = {
        "ignore_index": ignore_index,
        "padding_side": "right",
        "padding_tokens_masked": True,
        "prompt_tokens_masked": True,
        "row_order": "all_chosen_then_all_rejected",
    }
    if not isinstance(preprocessing, dict) or any(
        preprocessing.get(key) != value for key, value in expected_preprocessing.items()
    ):
        raise ValueError("frozen data preprocessing contract differs from the task contract")
    if not isinstance(tokenizer, dict) or (
        tokenizer.get("repo_id") != MODEL_REPO_ID
        or tokenizer.get("revision") != MODEL_REVISION
    ):
        raise ValueError("frozen data tokenizer is not the pinned Qwen3 tokenizer")

    # ``safe_open`` only accepts the non-executable safetensors format.  Clone
    # while the context is open so no later access depends on the mmap handle.
    try:
        from safetensors import safe_open

        with safe_open(tensors_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != {"input_ids", "attention_mask", "labels"}:
                raise ValueError(f"unexpected frozen tensor keys: {sorted(keys)}")
            input_ids = handle.get_tensor("input_ids").clone()
            attention_mask = handle.get_tensor("attention_mask").clone()
            labels = handle.get_tensor("labels").clone()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("failed to safely deserialize frozen tensors") from exc

    expected_shape = (2 * record_count, max_length)
    if (
        tuple(input_ids.shape) != expected_shape
        or tuple(attention_mask.shape) != expected_shape
        or tuple(labels.shape) != expected_shape
    ):
        raise ValueError(f"frozen tensors must all have shape {expected_shape}")
    if input_ids.dtype != torch.int32 or labels.dtype != torch.int32:
        raise ValueError("frozen input_ids and labels must use int32 storage")
    if attention_mask.dtype != torch.uint8:
        raise ValueError("frozen attention_mask must use uint8 storage")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("frozen attention_mask must be binary")
    if not bool((attention_mask[:, 1:] <= attention_mask[:, :-1]).all()):
        raise ValueError("frozen attention_mask must use contiguous right padding")
    if bool(attention_mask.sum(dim=1).eq(0).any()):
        raise ValueError("every frozen sequence must contain at least one non-padding token")
    vocab_size = int(EXPECTED_CONFIG["vocab_size"])
    if not bool(((input_ids >= 0) & (input_ids < vocab_size)).all()):
        raise ValueError("frozen input_ids contain a token outside the pinned vocabulary")
    valid_labels = labels.eq(ignore_index) | ((labels >= 0) & (labels < vocab_size))
    if not bool(valid_labels.all()):
        raise ValueError("frozen labels contain an invalid token or ignore value")
    scored = labels.ne(ignore_index)
    if bool((scored & attention_mask.eq(0)).any()):
        raise ValueError("frozen labels score a padding token")
    if not bool((labels[scored] == input_ids[scored]).all()):
        raise ValueError("frozen scored labels must equal their input token IDs")
    if bool(labels[:, 1:].eq(ignore_index).all(dim=1).any()):
        raise ValueError("every frozen sequence must have a predicted completion token")

    # Validate one contiguous scored completion and a shared prompt per pair.
    for row in range(2 * record_count):
        positions = torch.nonzero(scored[row], as_tuple=False).flatten()
        expected_positions = torch.arange(
            int(positions[0]), int(positions[-1]) + 1, dtype=positions.dtype
        )
        if not torch.equal(positions, expected_positions):
            raise ValueError(f"frozen row {row} does not have one contiguous completion")
    for record_index in range(record_count):
        rejected_index = record_index + record_count
        chosen_start = int(torch.nonzero(scored[record_index], as_tuple=False)[0])
        rejected_start = int(torch.nonzero(scored[rejected_index], as_tuple=False)[0])
        if chosen_start != rejected_start or chosen_start < 1:
            raise ValueError(f"frozen pair {record_index} does not share a prompt boundary")
        if not torch.equal(
            input_ids[record_index, :chosen_start],
            input_ids[rejected_index, :rejected_start],
        ):
            raise ValueError(f"frozen pair {record_index} does not share prompt tokens")

    task_non_padding_tokens = int(attention_mask.sum().item())
    task_scored_tokens = int(labels[:, 1:].ne(ignore_index).sum().item())
    if manifest.get("non_padding_tokens") != task_non_padding_tokens:
        raise ValueError("task-owned non-padding count differs from manifest")
    if manifest.get("scored_completion_tokens_including_eos") != task_scored_tokens:
        raise ValueError("task-owned scored-token count differs from manifest")
    return FrozenPreferenceDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        partition=expected_partition,
        manifest_sha256=actual_manifest,
        tensors_sha256=actual_tensors,
        record_count=record_count,
        sequence_length=max_length,
        non_padding_tokens=task_non_padding_tokens,
        scored_tokens=task_scored_tokens,
    )


def validate_local_snapshot(model_path: Path) -> dict[str, object]:
    """Fail closed on anything other than the pinned Qwen3-1.7B artifact."""

    root = model_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"model path is not a directory: {root}")
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError("pinned snapshot is missing config or safetensors index")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(
                f"unexpected model config {key}: {config.get(key)!r} != {expected!r}"
            )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    total_size = index.get("metadata", {}).get("total_size")
    if total_size != MODEL_TOTAL_SIZE:
        raise ValueError(
            f"unexpected safetensors logical size: {total_size!r} != {MODEL_TOTAL_SIZE}"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index has no weight_map")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and (root / name).is_file() for name in shard_names):
        raise ValueError("one or more indexed safetensors shards are missing")
    physical_bytes = sum((root / name).stat().st_size for name in shard_names)
    if physical_bytes < MODEL_TOTAL_SIZE:
        raise ValueError("indexed safetensors shards appear truncated")
    return {
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "resolved_path": str(root),
        "logical_weight_bytes": MODEL_TOTAL_SIZE,
        "physical_weight_bytes": physical_bytes,
        "shards": shard_names,
    }


def load_pinned_model(model_path: Path) -> tuple[nn.Module, dict[str, object]]:
    """Load the exact local model in BF16 with every parameter trainable."""

    artifact = validate_local_snapshot(model_path)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        artifact["resolved_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if model.__class__.__name__ != "Qwen3ForCausalLM":
        raise TypeError(f"unexpected loaded model class: {model.__class__.__name__}")
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("Qwen3 input embedding and lm_head weight are not tied")
    if not model.lm_head.weight.requires_grad:
        raise ValueError("tied lm_head weight is not trainable")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("public smoke requires full-parameter training")
    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    if unique_parameters != MODEL_UNIQUE_PARAMETERS:
        raise ValueError(
            f"unexpected unique parameter count: {unique_parameters} != "
            f"{MODEL_UNIQUE_PARAMETERS}"
        )
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("all loaded model parameters must be BF16")
    model.config.use_cache = False
    model.train()
    artifact["unique_parameters"] = unique_parameters
    artifact["parameter_dtype"] = "torch.bfloat16"
    artifact["tied_lm_head"] = True
    return model, artifact


def load_probe_objective(
    kind: str,
    lab_root: Path,
    contract: ObjectiveContract,
    requested_version: int | None = None,
) -> tuple[
    nn.Module,
    int | None,
    ModelOptimizer | None,
    Qwen3ModelPatch | None,
]:
    """Select either the official TRL baseline or the versioned candidate."""

    if kind == "reference":
        return build_official_trl_objective(contract), None, None, None
    if kind == "candidate":
        loaded = load_objective(
            lab_root,
            contract,
            requested_version=requested_version,
        )
        return loaded.module, loaded.version, loaded.optimize_model, loaded.model_patch
    raise ValueError(f"unknown objective kind: {kind!r}")


class TrainStepModule(nn.Module):
    """The immutable full-model boundary wrapped by FSDP.

    Qwen's output projection is deliberately not invoked.  The final hidden
    states and the trainable tied ``lm_head.weight`` are passed to the objective
    ABI, allowing a candidate to fuse projection and ORPO without owning any
    part of the backbone or optimizer.
    """

    def __init__(self, model: nn.Module, objective: nn.Module):
        super().__init__()
        self.model = model
        self.objective = objective

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> ORPOResult:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [2B, T]")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        if input_ids.shape[0] == 0 or input_ids.shape[0] % 2:
            raise ValueError("chosen-first preference batch must have shape [2B, T]")
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state[:, :-1, :]
        shifted_labels, nll_target = derive_aligned_orpo_targets(
            input_ids,
            attention_mask,
            labels,
        )
        result = self.objective(
            self.model.lm_head.weight,
            hidden_states,
            shifted_labels,
            self.model.lm_head.bias,
            nll_target,
        )
        return normalize_result(result)


def derive_aligned_orpo_targets(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive TRL's distinct OR and chosen-NLL next-token targets.

    ``completion_labels`` keeps prompt tokens masked for the chosen/rejected
    odds-ratio term.  Official decoder-only ORPO derives NLL labels from the
    complete input sequence and masks only padding, so ``nll_target`` retains
    prompt and completion tokens.  Both outputs are already shifted to align
    with the hidden states consumed by the objective ABI.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [2B, T]")
    if attention_mask.shape != input_ids.shape or completion_labels.shape != input_ids.shape:
        raise ValueError("attention_mask and completion_labels must match input_ids")
    if input_ids.shape[1] < 2:
        raise ValueError("ORPO next-token targets require sequence length >= 2")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention_mask must be binary")

    or_labels = completion_labels[:, 1:].contiguous().clone()
    shifted_attention = attention_mask[:, 1:]
    or_labels.masked_fill_(shifted_attention.eq(0), OFFICIAL_TRL_IGNORE_INDEX)
    nll_target = input_ids[:, 1:].contiguous().clone()
    nll_target.masked_fill_(shifted_attention.eq(0), OFFICIAL_TRL_IGNORE_INDEX)
    return or_labels, nll_target


def make_synthetic_preference_batch(
    *,
    seed: int,
    rank: int,
    step: int,
    pair_batch: int,
    sequence_length: int,
    prompt_length: int,
    vocab_size: int = EXPECTED_CONFIG["vocab_size"],
    ignore_index: int = OFFICIAL_TRL_IGNORE_INDEX,
) -> SyntheticPreferenceBatch:
    """Create a reproducible chosen-first batch without tokenizer or I/O noise."""

    if rank not in (0, 1):
        raise ValueError("the public probe only supports ranks 0 and 1")
    if ignore_index != OFFICIAL_TRL_IGNORE_INDEX:
        raise ValueError("official TRL synthetic labels require ignore_index=-100")
    if pair_batch < 1:
        raise ValueError("pair_batch must be positive")
    if sequence_length < 8 or not 1 < prompt_length < sequence_length - 2:
        raise ValueError("invalid prompt/sequence lengths")
    if vocab_size != EXPECTED_CONFIG["vocab_size"]:
        raise ValueError("synthetic batch must use the pinned model vocabulary")

    # CPU generation makes the byte stream independent of CUDA RNG/runtime.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1_000_003 * int(rank) + 9_973 * int(step))
    high = vocab_size - 1024  # keep synthetic IDs away from reserved special IDs
    prompts = torch.randint(
        32, high, (pair_batch, prompt_length), generator=generator, dtype=torch.long
    )
    continuation = sequence_length - prompt_length
    chosen_tail = torch.randint(
        32, high, (pair_batch, continuation), generator=generator, dtype=torch.long
    )
    rejected_tail = torch.randint(
        32, high, (pair_batch, continuation), generator=generator, dtype=torch.long
    )
    chosen = torch.cat((prompts, chosen_tail), dim=1)
    rejected = torch.cat((prompts.clone(), rejected_tail), dim=1)
    input_ids = torch.cat((chosen, rejected), dim=0)  # chosen rows must be first
    attention_mask = torch.ones_like(input_ids)

    # Deterministic right padding exercises non-padding token accounting.  Both
    # members of a pair receive the same valid length.
    max_padding = min(7, sequence_length - prompt_length - 2)
    if max_padding > 0:
        pair_padding = torch.tensor(
            [((seed + rank + step + index) % (max_padding + 1)) for index in range(pair_batch)],
            dtype=torch.long,
        )
        padding = torch.cat((pair_padding, pair_padding), dim=0)
        for row, count in enumerate(padding.tolist()):
            if count:
                attention_mask[row, -count:] = 0
                input_ids[row, -count:] = 0

    labels = input_ids.clone()
    labels[:, :prompt_length] = ignore_index
    labels.masked_fill_(attention_mask.eq(0), ignore_index)
    if bool(labels[:, 1:].eq(ignore_index).all(dim=1).any()):
        raise AssertionError("synthetic sequence has no scored completion token")

    digest = hashlib.sha256()
    for tensor in (input_ids, attention_mask, labels):
        digest.update(tensor.contiguous().numpy().tobytes())
    return SyntheticPreferenceBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        scored_tokens=int(labels[:, 1:].ne(ignore_index).sum().item()),
        non_padding_tokens=int(attention_mask.sum().item()),
        fingerprint=digest.hexdigest(),
        source="deterministic_synthetic_pretokenized_preference",
    )
