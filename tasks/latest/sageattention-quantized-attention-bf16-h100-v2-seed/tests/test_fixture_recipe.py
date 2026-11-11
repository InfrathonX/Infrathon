"""The e8 edge input is bound by its recipe, never by one host's bytes.

The reference capture is a public asset (``assets/fetch.sh``) and any
deployment can capture its own copy with ``fixtures/recipe/build_fixture.sh``;
either way the capture manifest travels next to it, so these tests cover the
binding the runtime enforces at load time -- file against manifest, manifest against the
``RECIPE.json`` shipped in the runtime build context -- plus the recipe's own
identity and sanity bands.  No Torch and no GPU are needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import load_module

runtime = load_module(
    "qattn_runtime_fixture_recipe", "runtime/quant_attention_runtime.py"
)

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "fixtures/recipe/RECIPE.json"
RECIPE = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
RECIPE_SHA256 = hashlib.sha256(RECIPE_PATH.read_bytes()).hexdigest()
PAYLOAD = b"this deployment's own capture bytes"


@pytest.fixture(autouse=True)
def _empty_asset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the layout check's asset fallback at an empty root of this test's own."""
    root = tmp_path / "asset-root"
    root.mkdir()
    monkeypatch.setenv("INFRATHON_ASSETS_ROOT", str(root))
    return root


def write_capture(directory: Path, payload: bytes = PAYLOAD, **overrides) -> Path:
    fixture = directory / runtime.FIXTURE_NAME
    fixture.write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "fixture": runtime.FIXTURE_NAME,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "recipe_id": runtime.RECIPE_ID,
        "recipe_sha256": RECIPE_SHA256,
    }
    manifest.update(overrides)
    (directory / runtime.FIXTURE_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return fixture


def test_the_runtime_no_longer_pins_the_capture_bytes() -> None:
    assert not hasattr(runtime, "FIXTURE_SHA256")
    source = (ROOT / "runtime/quant_attention_runtime.py").read_text(encoding="utf-8")
    assert "FIXTURE_SHA256" not in source
    layout = (ROOT / "scripts/validate_layout.py").read_text(encoding="utf-8")
    assert "FIXTURE_SHA256" not in layout


def test_recipe_identity_is_the_runtime_identity() -> None:
    recipe = runtime.load_recipe()
    assert recipe["recipe_id"] == runtime.RECIPE_ID == RECIPE["recipe_id"]
    assert recipe["identity"] == "per-deployment-capture"
    assert recipe["fixture"]["name"] == runtime.FIXTURE_NAME
    assert recipe["fixture"]["manifest"] == runtime.FIXTURE_MANIFEST_NAME
    assert recipe["tensors"]["layout"] == runtime.FIXTURE_LAYOUT
    assert recipe["tensors"]["dtype"] == "torch.bfloat16"
    assert tuple(recipe["tensors"]["shape"]) == (1, 8192, 12, 128)


def test_recipe_pins_public_sources_and_the_capture_point() -> None:
    assert RECIPE["model"]["repo"] == "Wan-AI/Wan2.1-T2V-1.3B"
    assert len(RECIPE["model"]["revision"]) == 40
    assert all(len(digest) == 64 for digest in RECIPE["model"]["files"].values())
    assert RECIPE["upstream"]["repo"].startswith("https://github.com/Wan-Video/Wan2.1")
    assert len(RECIPE["upstream"]["commit"]) == 40
    capture = RECIPE["capture"]
    assert (capture["block_index"], capture["step_index"]) == (15, 0)
    assert capture["cfg_branch"] == "cond"
    assert capture["seed"] == 4100 and capture["sampling_steps"] == 50
    assert capture["timestep_value"] == 999.0
    assert capture["post_rope_qk"] is True and capture["qk_norm"] is True


def test_sanity_bands_bracket_the_reference_capture() -> None:
    bands = RECIPE["sanity_bands"]
    assert bands["finite_fraction_min"] == 1.0
    assert set(bands["tensors"]) == {"q", "k", "v"}
    for name, band in bands["tensors"].items():
        for key in ("rms", "abs_max"):
            reference = band[f"{key}_reference"]
            low, high = band[f"{key}_min"], band[f"{key}_max"]
            assert low < reference < high, (name, key)
            # Generous but not vacuous: at least a factor of two either way,
            # at most a factor of four.
            assert 2.0 <= reference / low <= 4.0, (name, key)
            assert 2.0 <= high / reference <= 4.0, (name, key)


def test_matching_capture_and_manifest_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_capture(tmp_path)
    monkeypatch.setenv("SAGE_QATTN_FIXTURE_DIR", str(tmp_path))
    assert runtime._fixture_path() == fixture


def test_absent_fixture_directory_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAGE_QATTN_FIXTURE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="SAGE_QATTN_FIXTURE_DIR"):
        runtime._fixture_path()


def test_capture_without_its_manifest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / runtime.FIXTURE_NAME).write_bytes(PAYLOAD)
    monkeypatch.setenv("SAGE_QATTN_FIXTURE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="e8 capture is missing"):
        runtime._fixture_path()


def test_capture_that_its_manifest_does_not_describe_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_capture(tmp_path)
    fixture.write_bytes(PAYLOAD + b"tampered")
    monkeypatch.setenv("SAGE_QATTN_FIXTURE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="manifest SHA-256"):
        runtime._fixture_path()


def test_capture_from_a_different_recipe_revision_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_capture(tmp_path, recipe_sha256="0" * 64)
    monkeypatch.setenv("SAGE_QATTN_FIXTURE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="shipped RECIPE.json"):
        runtime._fixture_path()


def test_capture_from_a_different_recipe_id_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_capture(tmp_path, recipe_id="some-other-recipe-v9")
    monkeypatch.setenv("SAGE_QATTN_FIXTURE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="manifest identity differs"):
        runtime._fixture_path()


def test_manifest_naming_a_different_fixture_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_capture(tmp_path, fixture="something_else.pt")
    monkeypatch.setenv("SAGE_QATTN_FIXTURE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="manifest identity differs"):
        runtime._fixture_path()


def test_a_foreign_recipe_in_the_build_context_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foreign = dict(RECIPE, recipe_id="some-other-recipe-v9")
    path = tmp_path / "RECIPE.json"
    path.write_text(json.dumps(foreign), encoding="utf-8")
    monkeypatch.setenv("SAGE_QATTN_RECIPE_PATH", str(path))
    with pytest.raises(RuntimeError, match="recipe identity differs"):
        runtime.load_recipe()


def _layout_errors(root: Path) -> list[str]:
    """Run validate_layout's recipe/capture binding check against one tree."""
    pytest.importorskip("yaml")
    module = load_module("qattn_validate_layout_recipe", "scripts/validate_layout.py")
    errors: list[str] = []
    module._check_fixture_recipe(root, errors)
    return errors


def _fake_tree(tmp_path: Path, **overrides) -> Path:
    """A minimal task tree: the real recipe plus a stand-in capture pair.

    The binding check only hashes the capture, so a few bytes stand in for the
    75 MB tensor file and the whole test stays offline and instant.
    """
    root = tmp_path / "sageattention-quantized-attention-bf16-h100-v2-seed"
    (root / "fixtures/recipe").mkdir(parents=True)
    (root / "fixtures/recipe/RECIPE.json").write_bytes(RECIPE_PATH.read_bytes())
    write_capture(root / "fixtures", **overrides)
    return root


def test_validate_layout_accepts_a_matching_capture(tmp_path: Path) -> None:
    assert _layout_errors(_fake_tree(tmp_path)) == []


def test_validate_layout_reports_an_incomplete_capture(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path)
    (root / "fixtures" / runtime.FIXTURE_NAME).unlink()
    errors = _layout_errors(root)
    assert len(errors) == 1
    assert "e8 capture is incomplete" in errors[0]
    assert "assets/fetch.sh" in errors[0]
    assert "fixtures/recipe/build_fixture.sh" in errors[0]


def test_validate_layout_has_nothing_to_bind_without_a_capture(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path)
    for name in (runtime.FIXTURE_NAME, runtime.FIXTURE_MANIFEST_NAME):
        (root / "fixtures" / name).unlink()
    assert _layout_errors(root) == []


def test_validate_layout_binds_the_fetched_reference_capture(
    tmp_path: Path, _empty_asset_root: Path
) -> None:
    """Without a capture in fixtures/, the asset under the asset root is checked."""
    root = _fake_tree(tmp_path)
    public = _empty_asset_root / root.name / "public"
    public.mkdir(parents=True)
    for name in (runtime.FIXTURE_NAME, runtime.FIXTURE_MANIFEST_NAME):
        (public / name).write_bytes((root / "fixtures" / name).read_bytes())
        (root / "fixtures" / name).unlink()
    assert _layout_errors(root) == []
    (public / runtime.FIXTURE_NAME).write_bytes(PAYLOAD + b"tampered")
    assert any("manifest SHA-256" in e for e in _layout_errors(root))


def test_validate_layout_rejects_a_capture_from_another_recipe(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path, recipe_sha256="0" * 64)
    assert any("does not match fixtures/recipe/RECIPE.json" in e for e in _layout_errors(root))


def test_validate_layout_rejects_a_tampered_capture(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path)
    (root / "fixtures" / runtime.FIXTURE_NAME).write_bytes(PAYLOAD + b"tampered")
    assert any("manifest SHA-256" in e for e in _layout_errors(root))


def test_build_images_binds_the_capture_to_the_recipe() -> None:
    """The build refuses a capture that a different RECIPE.json produced.

    It also has to reach the capture at all on a fresh checkout, so layout
    validation runs after the per-deployment inputs exist, not before.
    """
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    for fragment in (
        '"$(jq -er \'.recipe_sha256\' "${fixture_manifest_source}")" == "${recipe_sha256}"',
        '"$(jq -er \'.sha256\' "${fixture_manifest_source}")" == "${fixture_sha256}"',
        '"$(jq -er \'.recipe_id\' "${fixture_manifest_source}")" == "${recipe_id}"',
        '"$(jq -er \'.fixture\' "${fixture_manifest_source}")" == "${fixture_name}"',
        'FIXTURE_FORCE=1 "${MIGRATION_ROOT}/fixtures/recipe/build_fixture.sh"',
    ):
        assert fragment in build, fragment
    capture_block = build.index("fixtures/recipe/build_fixture.sh")
    layout_gate = build.index('scripts/validate_layout.py" "${MIGRATION_ROOT}"')
    first_build = build.index('docker build')
    assert capture_block < layout_gate < first_build
    # A reused capture lives outside the task tree, so the layout gate has to be
    # pointed at the pair being baked rather than at fixtures/.
    assert '--fixture "${fixture_source}"' in build
    assert '--fixture-manifest "${fixture_manifest_source}"' in build


def test_validate_layout_accepts_a_capture_held_outside_the_task_tree(
    tmp_path: Path,
) -> None:
    """FIXTURE_PATH / FIXTURE_MANIFEST_PATH reuse, as build_images.sh does it."""
    root = _fake_tree(tmp_path)
    elsewhere = tmp_path / "another-deployment"
    elsewhere.mkdir()
    for name in (runtime.FIXTURE_NAME, runtime.FIXTURE_MANIFEST_NAME):
        (elsewhere / name).write_bytes((root / "fixtures" / name).read_bytes())
        (root / "fixtures" / name).unlink()
    assert _layout_errors(root) == []  # nothing in fixtures/ or the asset root
    pytest.importorskip("yaml")
    module = load_module("qattn_validate_layout_recipe", "scripts/validate_layout.py")
    errors: list[str] = []
    module._check_fixture_recipe(
        root,
        errors,
        elsewhere / runtime.FIXTURE_NAME,
        elsewhere / runtime.FIXTURE_MANIFEST_NAME,
    )
    assert errors == []


def test_capture_script_is_self_contained() -> None:
    source = (ROOT / "fixtures/recipe/capture_e8_wan_qkv.py").read_text(encoding="utf-8")
    for forbidden in ("iteration_docs", "edgebench_format_task", "wan_dit_cache_runtime"):
        assert forbidden not in source
    # Every location is an argument; nothing is read from an absolute host path.
    for required in ("--recipe", "--wan-src", "--checkpoint", "--output-dir"):
        assert required in source
