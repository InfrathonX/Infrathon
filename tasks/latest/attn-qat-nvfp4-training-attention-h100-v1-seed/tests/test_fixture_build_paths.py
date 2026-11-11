"""The two build paths for the recipe-bound e7 capture, checked without a GPU.

`scripts/build_images.sh` reaches a capture two ways: it takes one from
`fixtures/recipe/` when the tree is cold, or it stages the pair named by
`FIXTURE_PATH` / `FIXTURE_MANIFEST_PATH`.  Both were untested, and both have an
ordering obligation that only shows up when they are exercised:

* the layout must validate on a cold tree, where no capture exists yet, and again
  once one does -- the second pass is the one that judges the pair about to be
  baked into the images;
* a supplied pair must be judged *before* it is copied into the task tree, or a
  capture from another recipe lands in `fixtures/` and the next build-freshness
  assertion trips on it.

These run on the real task tree plus a synthetic one assembled in a temporary
directory, with hand-written capture bytes.  No GPU, no Torch, no Docker.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from conftest import TASK_ROOT, load_module

layout = load_module("attn_qat_validate_layout", "scripts/validate_layout.py")

RECIPE_PATH = TASK_ROOT / "fixtures/recipe/RECIPE.json"
RECIPE_SHA256 = hashlib.sha256(RECIPE_PATH.read_bytes()).hexdigest()
# Everything `_check_fixture_recipe` reads besides the capture pair itself.
SCAFFOLD = (
    "fixtures/recipe/RECIPE.json",
    "scripts/build_images.sh",
    "images/work/Dockerfile",
    "images/judge/Dockerfile",
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A directory carrying only what the capture checks read, and no capture."""
    root = tmp_path / layout.TASK_DIRECTORY
    for relative in SCAFFOLD:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TASK_ROOT / relative, destination)
    return root


def write_capture(directory: Path, *, body: bytes = b"capture bytes",
                  declared_sha256: str | None = None,
                  recipe_sha256: str | None = None,
                  recipe_id: str | None = None,
                  identity: str | None = None,
                  with_manifest: bool = True) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fixture = directory / layout.FIXTURE_NAME
    fixture.write_bytes(body)
    manifest_path = directory / layout.FIXTURE_MANIFEST_NAME
    if with_manifest:
        manifest = {
            "schema_version": 1,
            "identity": layout.FIXTURE_IDENTITY if identity is None else identity,
            "recipe": {
                "id": layout.FIXTURE_RECIPE_ID if recipe_id is None else recipe_id,
                "sha256": RECIPE_SHA256 if recipe_sha256 is None else recipe_sha256,
            },
            "fixture": {
                "name": layout.FIXTURE_NAME,
                "sha256": hashlib.sha256(body).hexdigest() if declared_sha256 is None
                else declared_sha256,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return fixture, manifest_path


def capture_errors(root: Path, **kwargs) -> list[str]:
    errors: list[str] = []
    layout._check_fixture_recipe(root, errors, **kwargs)
    return errors


# ------------------------------------------------------------------ cold tree ---
def test_a_cold_tree_validates(tree: Path) -> None:
    """build_images.sh runs the first layout pass before it has a capture."""
    assert capture_errors(tree) == []


def test_the_real_task_tree_validates_with_or_without_a_capture() -> None:
    errors: list[str] = []
    layout._check_fixture_recipe(TASK_ROOT, errors)
    assert errors == []


# ------------------------------------------------------- a staged capture pair ---
def test_a_staged_pair_that_matches_the_recipe_validates(tree: Path) -> None:
    write_capture(tree / "fixtures")
    assert capture_errors(tree) == []


def test_a_staged_capture_without_its_manifest_is_rejected(tree: Path) -> None:
    write_capture(tree / "fixtures", with_manifest=False)
    assert any("without its manifest" in error for error in capture_errors(tree))


def test_a_staged_manifest_without_its_capture_is_rejected(tree: Path) -> None:
    """Half a pair is worse than none: the manifest describes bytes that are not there."""
    write_capture(tree / "fixtures")
    (tree / "fixtures" / layout.FIXTURE_NAME).unlink()
    assert any("without the capture it describes" in error
               for error in capture_errors(tree))


def test_staged_bytes_that_do_not_match_the_manifest_are_rejected(tree: Path) -> None:
    write_capture(tree / "fixtures", declared_sha256="0" * 64)
    assert any("does not match its manifest" in error for error in capture_errors(tree))


def test_a_staged_pair_from_another_recipe_is_rejected(tree: Path) -> None:
    write_capture(tree / "fixtures", recipe_sha256="0" * 64)
    assert any("shipped RECIPE.json" in error for error in capture_errors(tree))


def test_a_staged_pair_from_another_recipe_id_is_rejected(tree: Path) -> None:
    write_capture(tree / "fixtures", recipe_id="some-other-recipe-v1")
    assert any("different recipe" in error for error in capture_errors(tree))


def test_a_staged_pair_with_another_identity_is_rejected(tree: Path) -> None:
    write_capture(tree / "fixtures", identity="host-private-blob")
    assert any("different identity" in error for error in capture_errors(tree))


def test_a_symlinked_staged_capture_is_rejected(tree: Path, tmp_path: Path) -> None:
    fixture, manifest_path = write_capture(tmp_path / "elsewhere")
    staged = tree / "fixtures"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / layout.FIXTURE_NAME).symlink_to(fixture)
    (staged / layout.FIXTURE_MANIFEST_NAME).write_bytes(manifest_path.read_bytes())
    assert any("regular files" in error for error in capture_errors(tree))


# ------------------------------------------------------ a supplied capture pair ---
def test_a_supplied_pair_outside_the_task_tree_validates(tree: Path,
                                                         tmp_path: Path) -> None:
    """The reuse path names a capture this deployment already holds elsewhere."""
    fixture, manifest_path = write_capture(tmp_path / "deployment-capture")
    assert capture_errors(tree, fixture=fixture, fixture_manifest=manifest_path) == []
    # Nothing was staged by validating it.
    assert not (tree / "fixtures" / layout.FIXTURE_NAME).exists()


def test_a_supplied_pair_from_another_recipe_is_rejected(tree: Path,
                                                         tmp_path: Path) -> None:
    """The bytes and their digest are honest; only the recipe binding is wrong."""
    fixture, manifest_path = write_capture(tmp_path / "wrong-recipe",
                                           recipe_sha256="0" * 64)
    errors = capture_errors(tree, fixture=fixture, fixture_manifest=manifest_path)
    assert any("shipped RECIPE.json" in error for error in errors)


def test_a_supplied_capture_that_does_not_exist_is_rejected(tree: Path,
                                                            tmp_path: Path) -> None:
    errors = capture_errors(tree, fixture=tmp_path / layout.FIXTURE_NAME,
                            fixture_manifest=tmp_path / layout.FIXTURE_MANIFEST_NAME)
    assert any("does not exist" in error for error in errors)


def test_a_supplied_capture_needs_its_manifest_named_too(tree: Path,
                                                         tmp_path: Path) -> None:
    fixture, _ = write_capture(tmp_path / "half")
    assert any("named together" in error for error in capture_errors(tree, fixture=fixture))


def test_the_command_line_carries_both_halves() -> None:
    errors = layout.main([str(TASK_ROOT), "--fixture", str(TASK_ROOT / "README.md")])
    assert errors == 1


# ------------------------------------------------------------ the build ordering ---
def test_build_images_validates_the_layout_on_both_sides_of_the_capture() -> None:
    errors: list[str] = []
    layout._check_build_order(TASK_ROOT, errors)
    assert errors == []


def _mutated_build_order(tmp_path: Path, old: str, new: str) -> list[str]:
    root = tmp_path / layout.TASK_DIRECTORY / "scripts"
    root.mkdir(parents=True)
    text = (TASK_ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    (root / "build_images.sh").write_text(text.replace(old, new, 1), encoding="utf-8")
    errors: list[str] = []
    layout._check_build_order(root.parent, errors)
    return errors


def test_dropping_the_post_capture_pass_is_caught(tmp_path: Path) -> None:
    errors = _mutated_build_order(
        tmp_path,
        '# The first layout pass ran on a tree that legitimately carried no capture; run it\n'
        '# again now that one is in place, so the pair that is about to be baked into both\n'
        '# images is validated by the same rule a reviewer would apply to a shipped tree.\n'
        '"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"\n',
        "",
    )
    assert any("three times" in error for error in errors)


def test_staging_a_supplied_capture_before_judging_it_is_caught(tmp_path: Path) -> None:
    errors = _mutated_build_order(
        tmp_path,
        '  "${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}" \\\n'
        '    --fixture "${FIXTURE_PATH}" --fixture-manifest "${FIXTURE_MANIFEST_PATH}"\n'
        '  install -m 0644 "${FIXTURE_PATH}" "${fixture_destination}"\n',
        '  install -m 0644 "${FIXTURE_PATH}" "${fixture_destination}"\n'
        '  "${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}" \\\n'
        '    --fixture "${FIXTURE_PATH}" --fixture-manifest "${FIXTURE_MANIFEST_PATH}"\n',
    )
    assert any("before it is staged" in error for error in errors)
