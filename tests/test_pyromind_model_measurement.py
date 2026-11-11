"""Actual cross-process admission failures; no SDK or model generation."""
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from sforge.pyromind.model_measurement import ModelMeasurementGate, PendingModelMeasurement


def arm(gate):
    gate.arm(run_id='unit-run',leg_id='unit-leg',measurement_profile_sha256='a' * 64)


def test_crashed_controller_leaves_durable_admission_fence(tmp_path):
    source = '''
import os,sys
sys.path.insert(0,sys.argv[1])
from sforge.pyromind.model_measurement import ModelMeasurementGate
gate=ModelMeasurementGate(sys.argv[2])
with gate.exclusive():
    gate.arm(run_id='crashed-run',leg_id='leg',measurement_profile_sha256='a'*64)
    os._exit(0)
'''
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable,'-I','-c',source,str(repository),str(tmp_path)],
                            capture_output=True,timeout=10)
    assert result.returncode == 0, result.stderr.decode()
    assert (tmp_path / 'measurement-inflight.json').is_file()
    # The OS flock vanished, but a different Controller still cannot generate.
    with pytest.raises(PendingModelMeasurement,match='Prior shared model cleanup'):
        with ModelMeasurementGate(tmp_path).exclusive():
            pytest.fail('Crash cannot silently admit another measurement')


def test_failed_cleanup_blocks_other_owner_until_exact_retry(tmp_path):
    first = ModelMeasurementGate(tmp_path)
    with first.exclusive():
        arm(first)
    other = ModelMeasurementGate(tmp_path)
    with pytest.raises(PendingModelMeasurement):
        with other.exclusive():
            pytest.fail('Unconfirmed cleanup must retain admission')
    with first.exclusive():
        first.confirm_cleaned()
    assert not (tmp_path / 'measurement-inflight.json').exists()
    with other.exclusive():
        pass


def test_cleanup_worker_only_signals_while_measurement_owner_holds_flock(tmp_path):
    gate = ModelMeasurementGate(tmp_path)
    with gate.exclusive():
        arm(gate)
        worker = threading.Thread(target=gate.confirm_cleaned)
        worker.start();worker.join(5)
        assert not worker.is_alive()
        assert (tmp_path / 'measurement-inflight.json').exists()
    assert not (tmp_path / 'measurement-inflight.json').exists()


def test_fence_change_cannot_be_deleted_by_prior_owner(tmp_path):
    gate = ModelMeasurementGate(tmp_path)
    with pytest.raises(PendingModelMeasurement,match='ownership changed'):
        with gate.exclusive():
            arm(gate)
            file = tmp_path / 'measurement-inflight.json'
            changed = json.loads(file.read_text());changed['leg_id'] = 'other-leg'
            file.write_text(json.dumps(changed))
            gate.confirm_cleaned()
    assert (tmp_path / 'measurement-inflight.json').exists()


@pytest.mark.parametrize('kind',['invalid_json','public_file','symlink','hardlink'])
def test_unsafe_fence_never_admits_generation(tmp_path,kind):
    file = tmp_path / 'measurement-inflight.json'
    file.write_text('{}');file.chmod(0o600)
    if kind == 'invalid_json':file.write_text('not JSON')
    elif kind == 'public_file':file.chmod(0o644)
    else:
        target=tmp_path / 'other.json';file.rename(target)
        if kind == 'symlink':file.symlink_to(target)
        else:file.hardlink_to(target)
    with pytest.raises((PendingModelMeasurement,OSError)):
        with ModelMeasurementGate(tmp_path).exclusive():
            pytest.fail('Unsafe ownership metadata cannot admit a model request')


@pytest.mark.parametrize('mode', [0o777, 0o775, 0o755])
def test_shared_model_gate_rejects_existing_public_directory(tmp_path, mode):
    tmp_path.chmod(mode)
    with pytest.raises(PendingModelMeasurement, match='private, canonical'):
        with ModelMeasurementGate(tmp_path).exclusive():
            pytest.fail('A writable parent can replace otherwise private ownership files')
    assert tmp_path.stat().st_mode & 0o777 == mode


def test_shared_model_gate_rejects_ancestor_alias(tmp_path):
    private = tmp_path / 'real' / 'model';private.mkdir(parents=True, mode=0o700)
    alias = tmp_path / 'alias';alias.symlink_to(tmp_path / 'real', target_is_directory=True)
    with pytest.raises(PendingModelMeasurement, match='private, canonical'):
        with ModelMeasurementGate(alias / 'model').exclusive():
            pytest.fail('Model admission must use the actual private service directory')


def test_shared_model_gate_normalizes_ordinary_parent_relative_cli_path(tmp_path):
    gate = ModelMeasurementGate(tmp_path / 'launch' / '..' / 'model')
    with gate.exclusive():
        arm(gate)
        assert (tmp_path / 'model' / 'measurement-inflight.json').is_file()
        gate.confirm_cleaned()
    assert gate.directory == (tmp_path / 'model').resolve()
