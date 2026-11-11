"""Execute the remote log reader locally to verify unsafe-file handling."""
import os
import subprocess
from types import SimpleNamespace

import pytest

from sforge.pyromind import native_diagnostics


def test_fifo_log_is_rejected_without_waiting_for_a_writer(tmp_path, monkeypatch):
    logs = tmp_path / 'logs'
    logs.mkdir()
    os.mkfifo(logs / 'stderr.log')
    monkeypatch.setattr(native_diagnostics, 'torchrun_log_directory',
                        lambda handle, namespace: str(logs))

    def execute(handle, command, **kwargs):
        # O_RDONLY alone blocks on the FIFO and raises TimeoutExpired here;
        # O_NONBLOCK lets the actual regular-file check reject it immediately.
        result = subprocess.run(command, capture_output=True, text=True, timeout=2)
        return SimpleNamespace(exit_code=result.returncode, timed_out=False,
                               output=result.stdout)

    backend = SimpleNamespace(exec_run_with_timeout=execute)
    with pytest.raises(RuntimeError, match='diagnostic log collection failed'):
        native_diagnostics.collect_torchrun_logs(backend, 'sb-test', tmp_path, namespace='test')
    assert not (tmp_path / 'native-torchrun-logs.json').exists()
