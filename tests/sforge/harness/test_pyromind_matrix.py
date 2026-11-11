import json
import hashlib
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest

from sforge.pyromind.matrix import (
    confirmed_cleanup, live_external_gpu, process_identity, reservation, reserved_gpu,
    atomic_json, validate_manifest, provider_probe_matches, controller_command,
    startup_retry_allowed,
    health_record,
    external_local_health,
    deployment_at_capacity,
    append_manifest_slots,
)


def test_restart_adds_new_task_without_resetting_live_or_fenced_ownership():
    live = {'slot': 'attn-qwen', 'status': 'running', 'pid': 123,
            'artifact_root': '/existing', 'reserved_gpu': 2, 'agent_started_at': 100}
    fenced = {'slot': 'flux-glm', 'status': 'cleanup-unconfirmed', 'reserved_gpu': 4}
    state = {'slots': [live.copy(), fenced.copy()]}
    manifest = {'slots': [{'slot': 'attn-qwen', 'deployment': 'changed-is-not-applied'},
                          {'slot': 'eagle-qwen', 'deployment': 'eagle', 'model_config': '/qwen'}]}
    append_manifest_slots(state, manifest)
    assert state['slots'][:2] == [live, fenced]
    assert reserved_gpu(state['slots']) == 6
    assert state['slots'][2]['status'] == 'queued'
    assert state['slots'][2]['model_config'] == '/qwen'
    append_manifest_slots(state, manifest)
    assert len(state['slots']) == 3


def test_restart_rejects_duplicate_persisted_slot_ownership():
    with pytest.raises(ValueError, match='duplicate slot ownership'):
        append_manifest_slots({'slots': [{'slot': 'one'}, {'slot': 'one'}]}, {'slots': []})


def test_gpu_reservation_accounts_for_pending_and_unconfirmed_cleanup():
    entries = [{'status': status, 'reserved_gpu': count} for status, count in
               [('launching', 2), ('starting', 2), ('running', 4), ('finalizing', 4),
                ('cleanup-unconfirmed', 2), ('queued', 4), ('completed', 4)]]
    assert reserved_gpu(entries) == 14


def test_cleanup_requires_runner_exact_id_confirmation():
    assert not confirmed_cleanup(None)
    assert not confirmed_cleanup({'status': 'failed'})
    assert not confirmed_cleanup({'remaining_sandbox_ids': ['sb-active'], 'cleanup_errors': []})
    assert not confirmed_cleanup({'remaining_sandbox_ids': [], 'cleanup_errors': ['GETFailed']})
    assert not confirmed_cleanup({'remaining_sandbox_ids': []})
    assert not confirmed_cleanup({'status':'agent-running','remaining_sandbox_ids': [], 'cleanup_errors': []})
    assert confirmed_cleanup({'status':'completed','remaining_sandbox_ids': [], 'cleanup_errors': []})


def test_external_allocations_count_pending_and_do_not_double_count_owned():
    class Sandbox(SimpleNamespace):
        def model_dump(self):
            return self.__dict__
    client = SimpleNamespace(list=lambda: [
        Sandbox(name='sf-owned-worker', resources={'gpu':'1'}, status='Running'),
        Sandbox(name='sf-owned-judge', resources={'gpu':'1'}, status='Pending'),
        Sandbox(name='other-pending', resources={'gpu':'2'}, status='Pending'),
        Sandbox(name='other-running', resources={'gpu':'3'}, status='Running'),
        Sandbox(name='other-cpu', resources={'cpu':'4'}, status='Running'),
    ])
    assert live_external_gpu(client, [{'status':'running', 'sandbox_name_prefix':'sf-owned'}]) == 5
    assert live_external_gpu(client, []) == 7


def test_count_worker_and_judge_independently():
    deployment = SimpleNamespace(resources_for_role=lambda role: {'gpu': 2 if role=='worker' else 3})
    assert reservation({}, deployment) == 5


def test_nvlink_task_limit_preserves_judge_pair_without_counting_local_fallback():
    entry={'deployment':'flux'}
    slots=[{'deployment':'flux','status':'external-local'},
           {'deployment':'flash','status':'running'},
           {'deployment':'flux','status':'starting'}]
    assert deployment_at_capacity(entry,slots,{'flux':1})
    assert not deployment_at_capacity(entry,slots,{'flux':2})
    assert not deployment_at_capacity(entry,slots[:-1],{'flux':1})
    assert deployment_at_capacity(entry,[{'deployment':'flux','status':'cleanup-unconfirmed'}],{'flux':1})


def test_missing_process_is_not_alive():
    assert process_identity(999999999) is None


def test_local_fallback_is_observed_without_remote_gpu_reservation(tmp_path):
    state=tmp_path/'local-run.json'
    entry={'slot':'flux-qwen-local','execution':'external-local','status':'external-local',
           'local_state_path':str(state),'requested_model':'qwen3.8-flash'}
    assert reserved_gpu([entry])==0
    assert external_local_health(entry)['local_status']=='preparing'
    state.write_text(json.dumps({'status':'running','run_id':'local-one','deadline_at':time.time()+43200}))
    result=external_local_health(entry)
    assert result['local_status']=='running' and result['local_run_id']=='local-one'
    assert result['remaining_seconds']>43190


def test_monitor_counts_interactive_submissions_separately_from_auto_judges(tmp_path):
    run_root=tmp_path/'run-one'
    run_root.mkdir()
    (run_root/'history.json').write_text(json.dumps({
        'agent_submissions':2, 'auto_submissions':1,
        'entries':[{'kind':'interactive'},{'kind':'interactive'},{'kind':'final'}],
    }))
    record=health_record({'slot':'task-model','status':'running','artifact_root':str(tmp_path)},
                         {'run_id':'run-one','agent_progress':{'status':'running'}})
    assert record['submissions']==2
    assert record['agent_status']=='running'


def test_retry_startup_only_after_cleanup_and_before_budget_starts():
    failed={'status':'failed','remaining_sandbox_ids':[],'cleanup_errors':[],
            'agent_started':False,'error':'Sandbox sb-abc123 entered failed'}
    assert startup_retry_allowed(failed,1)
    assert not startup_retry_allowed(failed,3)
    assert not startup_retry_allowed({**failed,'agent_started':True},1)
    assert not startup_retry_allowed({**failed,'remaining_sandbox_ids':['sb-active']},1)
    assert not startup_retry_allowed({**failed,'error':'Invalid candidate'},1)


def test_manifest_rejects_duplicate_slots_and_outside_source_paths():
    manifest={'schema_version':1,'experiment':'test','max_gpu':28,'hours_each':12,
              'slots':[{'slot':'attn-qwen','deployment':'deploy/pyromind/attnqat/deployment.json'}]}
    validate_manifest(manifest)
    with pytest.raises(ValueError):validate_manifest({**manifest,'max_gpu':True})
    with pytest.raises(ValueError):validate_manifest({**manifest,'max_gpu':29})
    with pytest.raises(ValueError):validate_manifest({**manifest,'slots':manifest['slots']*2})
    with pytest.raises(ValueError):validate_manifest({**manifest,'slots':[{'slot':'attn-qwen','deployment':'../outside.json'}]})


def test_probe_binds_effort_config_and_runtime(monkeypatch,tmp_path):
    monkeypatch.setattr('sforge.harness.agent_runtime.load_agent_runtime',
        lambda _:SimpleNamespace(version='0.159.3',tree_sha256='runtime-sha'))
    path=tmp_path/'model.env';path.write_text('MODEL=exact-model\n')
    model={'MODEL':'exact-model','REASONING_EFFORT':'max','SFORGE_AGENT_RUNTIME_DIR':'/unused'}
    probe={'status':'passed','model':'exact-model','reasoning_effort':'max',
           'model_config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'runtime_version':'0.159.3','runtime_tree_sha256':'runtime-sha'}
    assert provider_probe_matches(probe,path,model)
    for field in ('model','reasoning_effort','runtime_version','runtime_tree_sha256','model_config_sha256'):
        assert not provider_probe_matches({**probe,field:'stale'},path,model)


@pytest.mark.parametrize('acknowledged',[False,True])
def test_gate_cannot_start_controller_before_durable_ack(tmp_path,acknowledged):
    marker=tmp_path/'started'
    ack=tmp_path/'ack.json'
    gate=tmp_path/'gate.json'
    atomic_json(gate,{'nonce':'fixed','ack_path':str(ack),'expires_at':time.time()+.25,
                    'command':[sys.executable,'-c','from pathlib import Path;Path('+repr(str(marker))+').touch()']})
    if acknowledged:atomic_json(ack,{'nonce':'fixed'})
    result=subprocess.run([sys.executable,'-m','sforge.pyromind.launch_gate',str(gate)],capture_output=True,timeout=5)
    assert marker.exists()==acknowledged
    assert (result.returncode==0)==acknowledged


def test_every_slot_runs_the_manifest_reviewer_unless_it_names_its_own():
    from pathlib import Path
    manifest={'schema_version':1,'experiment':'test','max_gpu':28,'hours_each':12,
              'review_model_config':'/private/review.env',
              'slots':[{'slot':'eagle3-example','deployment':'deploy/pyromind/eagle3/deployment.json',
                        'assets_file':'deploy/pyromind/eagle3/assets.json'},
                       {'slot':'eagle3-opus','deployment':'deploy/pyromind/eagle3/deployment.json',
                        'review_model_config':'/private/other-review.env'}]}
    validate_manifest(manifest)
    repo=Path('/repo')
    first=controller_command(repo,repo/'deploy/pyromind/eagle3/deployment.json',Path('/private/example.env'),
                             Path('/out/runs'),manifest['slots'][0],manifest)
    assert first[first.index('--review-model-config')+1]=='/private/review.env'
    assert first[first.index('--assets-file')+1]=='/repo/deploy/pyromind/eagle3/assets.json'
    assert first[first.index('--hours')+1]=='12'
    second=controller_command(repo,repo/'deploy/pyromind/eagle3/deployment.json',Path('/private/opus.env'),
                              Path('/out/runs'),manifest['slots'][1],manifest)
    assert second[second.index('--review-model-config')+1]=='/private/other-review.env'
    with pytest.raises(ValueError):
        validate_manifest({**manifest,'review_model_config':'relative/review.env'})
    plain={k:v for k,v in manifest.items() if k!='review_model_config'}
    assert '--review-model-config' not in controller_command(repo,repo/'d.json',Path('/m.env'),
        Path('/out'),{'slot':'x','deployment':'d.json'},plain)
