"""The e7 edge input is bound by its recipe, not by the digest of one host's bytes.

Every deployment captures its own `e7_wan13b_grad_b29_t1_s8192.pt` at build time from
`fixtures/recipe/`, so the runtime cannot pin a SHA-256.  What it pins instead is
checked here: the manifest baked next to the capture must describe those exact bytes
and must name the `RECIPE.json` shipped beside the runtime, and every recorded tensor
must meet the recipe's shape, dtype, finiteness and sanity bands.  Each of those is
exercised on synthetic tensors, so the module needs no GPU and no real capture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import TASK_ROOT, load_module

torch = pytest.importorskip("torch")

runtime = load_module("attn_qat_runtime_recipe", "runtime/attn_qat_runtime.py")

RECIPE_PATH = TASK_ROOT / "fixtures/recipe/RECIPE.json"
RECIPE = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
RECIPE_SHA256 = hashlib.sha256(RECIPE_PATH.read_bytes()).hexdigest()


def _tensor_at(rms: float, abs_max: float, shape):
    """A bf16 tensor with (approximately) the requested RMS and absolute maximum."""
    flat = torch.full((int(torch.tensor(shape).prod()),), rms, dtype=torch.float32)
    flat[0] = abs_max
    # Rescale the tail so the realised RMS lands back on the request.
    target = rms * rms * flat.numel()
    tail = target - abs_max * abs_max
    if tail > 0:
        flat[1:] = (tail / (flat.numel() - 1)) ** 0.5
    return flat.reshape(shape).to(torch.bfloat16)


def _reference_payload():
    """A payload whose five tensors sit in the middle of their recipe bands."""
    payload = {}
    for name in runtime.FIXTURE_VALIDATED_TENSORS:
        reference = RECIPE["reference_capture"]["tensor_stats"][name]
        payload[name] = _tensor_at(
            reference["rms"], reference["abs_max"], runtime.FIXTURE_SHAPE
        )
    return payload


@pytest.fixture(scope="module")
def capture_dir(tmp_path_factory) -> Path:
    """One reused directory for every case.

    Each case rewrites the capture in place: a full-geometry payload is 126 MB, and a
    fresh directory per case would write two gigabytes to the temporary filesystem for
    no added coverage.
    """
    return tmp_path_factory.mktemp("e7-capture")


def _write_capture(directory: Path, payload: dict, *, recipe_sha256: str | None = None,
                   recipe_id: str | None = None, identity: str | None = None,
                   corrupt: bool = False) -> Path:
    fixture = directory / runtime.FIXTURE_NAME
    torch.save(payload, fixture)
    if corrupt:
        with fixture.open("ab") as handle:
            handle.write(b"\\0")
    manifest = {
        "schema_version": 1,
        "identity": runtime.FIXTURE_IDENTITY if identity is None else identity,
        "recipe": {
            "id": runtime.FIXTURE_RECIPE_ID if recipe_id is None else recipe_id,
            "sha256": RECIPE_SHA256 if recipe_sha256 is None else recipe_sha256,
            "path": "fixtures/recipe/RECIPE.json",
        },
        "fixture": {
            "name": runtime.FIXTURE_NAME,
            "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest() if not corrupt
            else hashlib.sha256(b"a different capture").hexdigest(),
            "bytes": fixture.stat().st_size,
        },
    }
    (directory / runtime.FIXTURE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return fixture


def test_the_runtime_pins_no_capture_digest() -> None:
    """The whole point of the change: no byte digest anywhere in the binding."""
    assert not hasattr(runtime, "FIXTURE_SHA256")
    source = (TASK_ROOT / "runtime/attn_qat_runtime.py").read_text(encoding="utf-8")
    assert "58a7773f" not in source
    assert runtime.FIXTURE_IDENTITY == "per-deployment-capture"
    assert runtime.FIXTURE_VALIDATED_TENSORS == ("q", "k", "v", "o", "do")
    assert runtime.FIXTURE_TENSORS == ("q", "k", "v", "do")


def test_the_shipped_recipe_is_the_one_the_runtime_expects() -> None:
    assert runtime._fixture_recipe_path() == RECIPE_PATH
    recipe = runtime._load_fixture_recipe()
    assert recipe["recipe_id"] == runtime.FIXTURE_RECIPE_ID
    assert recipe["fixture_name"] == runtime.FIXTURE_NAME
    assert tuple(recipe["capture"]["shape"]) == runtime.FIXTURE_SHAPE
    assert tuple(recipe["capture"]["tensors"]) == runtime.FIXTURE_VALIDATED_TENSORS
    assert recipe["model"]["repo_id"] == "Wan-AI/Wan2.1-T2V-1.3B"
    assert len(recipe["model"]["revision"]) == 40
    assert len(recipe["upstream"]["commit"]) == 40


def test_every_band_brackets_the_reference_capture() -> None:
    """The bands must contain the H100 reference and must not be unbounded."""
    for name in runtime.FIXTURE_VALIDATED_TENSORS:
        band = RECIPE["sanity_bands"][name]
        reference = RECIPE["reference_capture"]["tensor_stats"][name]
        assert band["rms_min"] < reference["rms"] < band["rms_max"]
        assert band["abs_max_min"] < reference["abs_max"] < band["abs_max_max"]
        assert band["finite_fraction_min"] == 1.0
        # Widened by two on the RMS and four on the magnitude, not by an order more.
        assert band["rms_max"] / band["rms_min"] <= 4.01
        assert band["abs_max_max"] / band["abs_max_min"] <= 16.01


def test_a_capture_that_matches_its_manifest_and_recipe_loads(capture_dir: Path,
                                                              monkeypatch) -> None:
    _write_capture(capture_dir, _reference_payload())
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    tensors = runtime._load_fixture(torch, torch.device("cpu"))
    assert set(tensors) == set(runtime.FIXTURE_TENSORS)
    for tensor in tensors.values():
        # Stored [B, S, H, D]; handed to the case as [B, H, S, D].
        assert tuple(tensor.shape) == (1, 12, 8192, 128)
        assert tensor.dtype == torch.bfloat16
        assert tensor.is_contiguous()


def test_bytes_that_do_not_match_the_manifest_fail_closed(capture_dir: Path,
                                                          monkeypatch) -> None:
    _write_capture(capture_dir, _reference_payload(), corrupt=True)
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match="does not match its capture manifest"):
        runtime._fixture_path()


def test_a_capture_from_another_recipe_fails_closed(capture_dir: Path, monkeypatch) -> None:
    _write_capture(capture_dir, _reference_payload(), recipe_sha256="0" * 64)
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match="not produced by this runtime's recipe"):
        runtime._fixture_path()


def test_a_capture_from_another_recipe_id_fails_closed(capture_dir: Path, monkeypatch) -> None:
    _write_capture(capture_dir, _reference_payload(), recipe_id="something-else-v1")
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match="names a different recipe"):
        runtime._fixture_path()


def test_a_manifest_with_another_identity_fails_closed(capture_dir: Path, monkeypatch) -> None:
    _write_capture(capture_dir, _reference_payload(), identity="host-private-blob")
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match="different identity"):
        runtime._fixture_path()


def test_a_symlinked_capture_fails_closed(capture_dir: Path, tmp_path: Path,
                                          monkeypatch) -> None:
    _write_capture(capture_dir, _reference_payload())
    (tmp_path / runtime.FIXTURE_NAME).symlink_to(capture_dir / runtime.FIXTURE_NAME)
    (tmp_path / runtime.FIXTURE_MANIFEST_NAME).write_bytes(
        (capture_dir / runtime.FIXTURE_MANIFEST_NAME).read_bytes()
    )
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        runtime._fixture_path()


@pytest.mark.parametrize("name", ["q", "k", "v", "o", "do"])
def test_a_tensor_outside_its_band_fails_closed(name: str, capture_dir: Path,
                                                monkeypatch) -> None:
    """A capture of the wrong block, sigma, prompt or model lands outside the bands."""
    payload = _reference_payload()
    band = RECIPE["sanity_bands"][name]
    payload[name] = _tensor_at(
        band["rms_max"] * 4.0, band["abs_max_max"] * 4.0, runtime.FIXTURE_SHAPE
    )
    _write_capture(capture_dir, payload)
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match=f"e7 tensor {name} fails its recipe"):
        runtime._load_fixture(torch, torch.device("cpu"))


def test_a_non_finite_capture_fails_closed(capture_dir: Path, monkeypatch) -> None:
    payload = _reference_payload()
    payload["v"] = payload["v"].clone()
    payload["v"][0, 0, 0, 0] = float("nan")
    _write_capture(capture_dir, payload)
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match="e7 tensor v is not finite"):
        runtime._load_fixture(torch, torch.device("cpu"))


def test_the_wrong_geometry_fails_closed(capture_dir: Path, monkeypatch) -> None:
    payload = _reference_payload()
    payload["q"] = payload["q"][:, :4096].contiguous()
    _write_capture(capture_dir, payload)
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match="bad e7 geometry"):
        runtime._load_fixture(torch, torch.device("cpu"))


def test_the_wrong_dtype_fails_closed(capture_dir: Path, monkeypatch) -> None:
    payload = _reference_payload()
    payload["k"] = payload["k"].float()
    _write_capture(capture_dir, payload)
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match="bad e7 dtype"):
        runtime._load_fixture(torch, torch.device("cpu"))


def test_a_payload_missing_the_orientation_output_fails_closed(capture_dir: Path,
                                                               monkeypatch) -> None:
    payload = _reference_payload()
    payload.pop("o")
    _write_capture(capture_dir, payload)
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(capture_dir))
    with pytest.raises(RuntimeError, match="bad e7 payload"):
        runtime._load_fixture(torch, torch.device("cpu"))


def test_the_recipe_path_is_not_environment_controlled(monkeypatch) -> None:
    """The recipe anchors the binding, so it is resolved from the module, never from env."""
    monkeypatch.setenv("ATTN_QAT_RECIPE_PATH", "/tmp/not-the-recipe.json")
    monkeypatch.setenv("ATTN_QAT_RECIPE_DIR", "/tmp")
    assert runtime._fixture_recipe_path() == RECIPE_PATH
    source = (TASK_ROOT / "runtime/attn_qat_runtime.py").read_text(encoding="utf-8")
    assert "ATTN_QAT_RECIPE" not in source


def test_the_capture_script_stays_self_contained() -> None:
    """The recipe may depend on the pinned upstream checkout and on nothing else."""
    text = (TASK_ROOT / "fixtures/recipe/capture_e7_wan_grad.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("iteration_docs", "edgebench", "/home/", "lab/captures",
                      "attn_qat_runtime"):
        assert forbidden not in text, forbidden
    assert "from wan.configs import WAN_CONFIGS" in text
    assert "sys.path.insert(0, str(args.upstream.resolve()))" in text


def test_the_build_script_needs_an_explicit_accelerator() -> None:
    text = (TASK_ROOT / "fixtures/recipe/build_fixture.sh").read_text(encoding="utf-8")
    assert "FIXTURE_GPU" in text
    assert 'NVIDIA_VISIBLE_DEVICES=${fixture_gpu}' in text
    assert "hf download" in text
    assert RECIPE["model"]["repo_id"] not in text  # read from the recipe, not hard-coded


def test_the_fixture_fetcher_is_gone() -> None:
    assert not (TASK_ROOT / "scripts/fetch_fixtures.sh").exists()
    for relative in ("scripts/validate_layout.py", "scripts/build_images.sh",
                     "portable/resource-manifest.json",
                     "portable/build-resource-manifest.json"):
        assert "fetch_fixtures" not in (TASK_ROOT / relative).read_text(encoding="utf-8")
