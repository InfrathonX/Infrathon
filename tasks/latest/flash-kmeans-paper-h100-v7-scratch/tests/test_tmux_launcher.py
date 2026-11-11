from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB = PROJECT_ROOT / "scripts" / "lib.sh"


def _detached_launchers() -> tuple[tuple[Path, str], ...]:
    """Each scripts/launch_<name>_tmux.sh whose scripts/run_<name>.sh is a run
    wrapper (maintainer files; a released tree may carry none)."""
    spec = importlib.util.spec_from_file_location(
        "flash_kmeans_validate_layout_for_tmux", PROJECT_ROOT / "scripts" / "validate_layout.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrappers = module.run_wrappers(PROJECT_ROOT)
    found = []
    for path in sorted((PROJECT_ROOT / "scripts").glob("launch_*_tmux.sh")):
        runner = "run_" + path.name[len("launch_"):-len("_tmux.sh")] + ".sh"
        if runner in wrappers:
            found.append((path, runner))
    return tuple(found)


LAUNCHERS = _detached_launchers()

FORWARDED_ENV = (
    "CODEX_AUTH_JSON_PATH",
    "MODEL",
    "REASONING_EFFORT",
    "CODEX_CLI_VERSION",
    "TASK_TIMEOUT",
    "EVAL_INTERVAL",
    "WORK_GPU",
    "JUDGE_GPU",
    "JUDGE_PORT",
    "MLSYSFORGE_ROOT",
    "SFORGE_BIN",
    "TASKS_DIR",
    "LOGS_DIR",
    "ARTIFACTS_DIR",
)


@dataclass(frozen=True)
class LauncherResult:
    process: subprocess.CompletedProcess[str]
    pane_rc: bytes | None
    launcher_exit: bytes | None
    launcher_log: bytes | None
    unexpected_log_artifacts: tuple[str, ...]
    marker: str


@unittest.skipUnless(LAUNCHERS, "no maintainer tmux launchers in this tree")
class TmuxLauncherTests(unittest.TestCase):
    def run_launcher(
        self,
        runner_rc: int,
        launcher: Path,
        runner_name: str,
    ) -> LauncherResult:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            migration_root = temp_root / "migration"
            scripts_dir = migration_root / "scripts"
            logs_dir = temp_root / "logs"
            fake_bin = temp_root / "bin"
            scripts_dir.mkdir(parents=True)
            logs_dir.mkdir()
            fake_bin.mkdir()

            shutil.copy2(launcher, scripts_dir / launcher.name)
            shutil.copy2(LIB, scripts_dir / LIB.name)

            marker = f"tmux-launcher-runner-marker-{runner_rc}"
            (scripts_dir / runner_name).write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\n' {marker!r}
                    exit {runner_rc}
                    """
                ),
                encoding="utf-8",
            )

            pane_rc_path = temp_root / "pane.rc"
            fake_tmux = fake_bin / "tmux"
            fake_tmux.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -u

                    case "${1:-}" in
                      has-session)
                        exit 1
                        ;;
                      new-session)
                        pane_command="${!#}"
                        (
                          /bin/bash -c "${pane_command}"
                          pane_rc="$?"
                          pane_rc_tmp="${FAKE_TMUX_PANE_RC_PATH}.tmp.$$"
                          printf '%s\\n' "${pane_rc}" > "${pane_rc_tmp}"
                          mv -f "${pane_rc_tmp}" "${FAKE_TMUX_PANE_RC_PATH}"
                        ) </dev/null >/dev/null 2>&1 &
                        exit 0
                        ;;
                      *)
                        echo "unexpected fake tmux invocation: $*" >&2
                        exit 64
                        ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            fake_tmux.chmod(0o755)

            run_id = f"launcher-test-{runner_rc}"
            env = os.environ.copy()
            for name in FORWARDED_ENV:
                env.pop(name, None)
            env.update(
                {
                    "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                    "RUN_ID": run_id,
                    "RUN_STAMP": "test-stamp",
                    "TMUX_SESSION": f"launcher-test-session-{runner_rc}",
                    "LOGS_DIR": str(logs_dir),
                    "FAKE_TMUX_PANE_RC_PATH": str(pane_rc_path),
                }
            )

            completed = subprocess.run(
                ["/bin/bash", str(scripts_dir / launcher.name)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            launcher_exit = logs_dir / f"launch-{run_id}.exit"
            launcher_log = logs_dir / f"launch-{run_id}.log"
            deadline = time.monotonic() + 5.0
            while (
                not launcher_exit.is_file() or not pane_rc_path.is_file()
            ) and time.monotonic() < deadline:
                time.sleep(0.01)
            expected_log_names = {launcher_exit.name, launcher_log.name}
            unexpected = tuple(
                sorted(
                    path.name
                    for path in logs_dir.iterdir()
                    if path.name not in expected_log_names
                )
            )
            return LauncherResult(
                process=completed,
                pane_rc=pane_rc_path.read_bytes() if pane_rc_path.is_file() else None,
                launcher_exit=(
                    launcher_exit.read_bytes() if launcher_exit.is_file() else None
                ),
                launcher_log=(
                    launcher_log.read_bytes() if launcher_log.is_file() else None
                ),
                unexpected_log_artifacts=unexpected,
                marker=marker,
            )

    def assert_launcher_result(
        self,
        runner_rc: int,
        launcher: Path,
        runner_name: str,
    ) -> None:
        result = self.run_launcher(runner_rc, launcher, runner_name)
        expected_rc = f"{runner_rc}\n".encode()
        expected_log = f"{result.marker}\n".encode()

        self.assertEqual(result.process.returncode, 0, result.process.stderr)
        self.assertEqual(result.pane_rc, expected_rc)
        self.assertEqual(result.launcher_exit, expected_rc)
        self.assertEqual(result.unexpected_log_artifacts, ())
        self.assertEqual(result.launcher_log, expected_log)

    def test_detached_runner_failure_is_preserved_in_terminal_status(self) -> None:
        for launcher, runner_name in LAUNCHERS:
            with self.subTest(launcher=launcher.name):
                self.assert_launcher_result(23, launcher, runner_name)

    def test_detached_runner_success_is_preserved_in_terminal_status(self) -> None:
        for launcher, runner_name in LAUNCHERS:
            with self.subTest(launcher=launcher.name):
                self.assert_launcher_result(0, launcher, runner_name)

    def test_rejects_unsafe_run_id_before_creating_paths_or_tmux(self) -> None:
        for launcher, runner_name in LAUNCHERS:
            with self.subTest(launcher=launcher.name), tempfile.TemporaryDirectory() as tmp:
                temp_root = Path(tmp)
                migration_root = temp_root / "migration"
                scripts_dir = migration_root / "scripts"
                logs_dir = temp_root / "logs"
                fake_bin = temp_root / "bin"
                scripts_dir.mkdir(parents=True)
                logs_dir.mkdir()
                fake_bin.mkdir()
                shutil.copy2(launcher, scripts_dir / launcher.name)
                shutil.copy2(LIB, scripts_dir / LIB.name)
                (scripts_dir / runner_name).write_text(
                    "#!/usr/bin/env bash\nexit 0\n",
                    encoding="utf-8",
                )

                tmux_marker = temp_root / "tmux-invoked"
                fake_tmux = fake_bin / "tmux"
                fake_tmux.write_text(
                    textwrap.dedent(
                        """\
                        #!/usr/bin/env bash
                        : > "${FAKE_TMUX_MARKER}"
                        exit 0
                        """
                    ),
                    encoding="utf-8",
                )
                fake_tmux.chmod(0o755)

                env = os.environ.copy()
                for name in FORWARDED_ENV:
                    env.pop(name, None)
                env.update(
                    {
                        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                        "RUN_ID": "x/../../escaped",
                        "RUN_STAMP": "test-stamp",
                        "TMUX_SESSION": "safe-test-session",
                        "LOGS_DIR": str(logs_dir),
                        "FAKE_TMUX_MARKER": str(tmux_marker),
                    }
                )
                completed = subprocess.run(
                    ["/bin/bash", str(scripts_dir / launcher.name)],
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("unsafe run id", completed.stderr)
                self.assertFalse(tmux_marker.exists())
                self.assertEqual(tuple(logs_dir.iterdir()), ())
                self.assertFalse((temp_root / "escaped.log").exists())
                self.assertFalse((temp_root / "escaped.exit").exists())


if __name__ == "__main__":
    unittest.main()
