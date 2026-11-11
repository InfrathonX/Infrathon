from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.harness.backend.pyromind_backend import PyroMindBackend, PyroMindHandle, SandboxPlacementRejected
from sforge.pyromind.cli import safe_extract


def client_with_result(code=0, output="ok", error=""):
    client = Mock()
    client.exec_command.return_value = SimpleNamespace(
        output=output, stderr="", returncode=code, exception_info=error,
    )
    return client


def test_actual_sdk_nonzero_command_is_not_an_infrastructure_error():
    client = client_with_result(7, "failed test", "command exited with code 7")
    backend = PyroMindBackend(client)
    result = backend.exec_run(PyroMindHandle("sb-one", "worker"), "exit 7")
    assert result.exit_code == 7
    assert result.output == "failed test"


@pytest.mark.pyromind_sdk
def test_controller_declared_service_port_is_forwarded_to_sdk():
    client = Mock()
    client.create.return_value = SimpleNamespace(id='sb-model', name='model')
    backend = PyroMindBackend(client, port_mappings=[{'container_port': 18000, 'name': 'vllm'}])
    backend.create_container('image@sha256:' + 'a' * 64, 'model')
    request = client.create.call_args.args[0]
    assert len(request.port_mappings) == 1
    assert request.port_mappings[0].container_port == 18000
    assert request.port_mappings[0].name == 'vllm'
    assert request.port_mappings[0].protocol == 'TCP'


@pytest.mark.pyromind_sdk
def test_placement_retry_preserves_request_and_caller_handle_after_exact_cleanup(tmp_path):
    import json
    from pyromind_sdk.client.base import PyroMindAPIError
    client=client_with_result(output='0')
    client.create.side_effect=[SimpleNamespace(id='sb-one',name='sf-owned-judge-one'),
                               SimpleNamespace(id='sb-two',name='sf-owned-judge-two')]
    client.get_sandbox.side_effect=[SimpleNamespace(status='Running'),SimpleNamespace(status='Running'),
        PyroMindAPIError('deleted',status_code=404),SimpleNamespace(status='Running')]
    admitted=[]
    def validate(handle):
        if handle.id=='sb-one':raise SandboxPlacementRejected('nvlink-pair-required',{'links':['PXB','PXB']})
        admitted.append(handle.id)
    backend=PyroMindBackend(client,on_started=validate,event_path=tmp_path/'events.jsonl',
        gpu_count=2,gpu_card='H100',volume_mounts=[{'host_path':'/workspace/frozen-version','mount_path':'/candidate','read_only':True}],
        port_mappings=[{'container_port':18000,'name':'service'}])
    handle=backend.create_container('image@sha256:'+'a'*64,'sf-owned-judge',
        environment={'SFORGE_TOKEN':'local-test-bearer'},cpu_limit=48,mem_limit='96Gi')
    backend.start_container(handle)
    assert handle.id=='sb-two' and admitted==['sb-two']
    assert handle.environment=={'SFORGE_TOKEN':'local-test-bearer'}
    first,second=[call.args[0] for call in client.create.call_args_list]
    assert first.model_dump(exclude={'name'})==second.model_dump(exclude={'name'})
    calls=[call[0] for call in client.method_calls]
    assert calls.index('delete')<len(calls)-1-calls[::-1].index('create')
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [event['id'] for event in events if event['event']=='placement_cleanup_confirmed']==['sb-one']
    assert len({event['allocation_sha256'] for event in events if event['event']=='created'})==1
    assert 'local-test-bearer' not in (tmp_path/'events.jsonl').read_text()


@pytest.mark.parametrize('confirmation',['still-present',403])
@pytest.mark.pyromind_sdk
def test_placement_retry_never_allocates_again_without_independent_404(confirmation):
    from pyromind_sdk.client.base import PyroMindAPIError
    client=client_with_result(output='0')
    client.create.return_value=SimpleNamespace(id='sb-one',name='sf-owned-judge-one')
    final=SimpleNamespace(status='Running') if confirmation=='still-present' else PyroMindAPIError('forbidden',status_code=403)
    client.get_sandbox.side_effect=[SimpleNamespace(status='Running'),SimpleNamespace(status='Running'),final]
    def reject(_):raise SandboxPlacementRejected('nvlink-pair-required',{})
    backend=PyroMindBackend(client,on_started=reject)
    handle=backend.create_container('image','judge')
    with pytest.raises((RuntimeError,PyroMindAPIError)):backend.start_container(handle)
    assert client.create.call_count==1


@pytest.mark.pyromind_sdk
def test_placement_retry_is_bounded_and_does_not_retry_ordinary_task_failures():
    from pyromind_sdk.client.base import PyroMindAPIError
    client=client_with_result(output='0')
    client.create.side_effect=[SimpleNamespace(id='sb-'+str(i),name='sf-owned-'+str(i)) for i in range(3)]
    client.get_sandbox.side_effect=[value for _ in range(3) for value in
        [SimpleNamespace(status='Running'),SimpleNamespace(status='Running'),PyroMindAPIError('deleted',status_code=404)]]
    def reject(_):raise SandboxPlacementRejected('nvlink-pair-required',{})
    backend=PyroMindBackend(client,on_started=reject)
    handle=backend.create_container('image','judge')
    with pytest.raises(SandboxPlacementRejected):backend.start_container(handle)
    assert client.create.call_count==3 and client.delete.call_count==3
    plain=client_with_result(output='0')
    plain.create.return_value=SimpleNamespace(id='sb-ordinary',name='judge')
    plain.get_sandbox.return_value.status='Running'
    def task_failure(_):raise RuntimeError('candidate test failed')
    ordinary=PyroMindBackend(plain,on_started=task_failure)
    with pytest.raises(RuntimeError,match='candidate test failed'):
        ordinary.start_container(ordinary.create_container('image','judge'))
    assert plain.create.call_count==1 and plain.delete.call_count==0


@pytest.mark.pyromind_sdk
def test_cancel_during_rejected_cleanup_prevents_reallocation():
    from pyromind_sdk.client.base import PyroMindAPIError
    client=client_with_result(output='0')
    client.create.return_value=SimpleNamespace(id='sb-one',name='judge')
    client.get_sandbox.side_effect=[SimpleNamespace(status='Running'),SimpleNamespace(status='Running'),
                                   PyroMindAPIError('deleted',status_code=404)]
    cancelled=False
    def reject(_):
        nonlocal cancelled
        cancelled=True
        raise SandboxPlacementRejected('nvlink-pair-required',{})
    backend=PyroMindBackend(client,on_started=reject)
    backend.startup_cancelled=lambda:cancelled
    with pytest.raises(RuntimeError,match='stopping'):
        backend.start_container(backend.create_container('image','judge'))
    assert client.create.call_count==1 and client.delete.call_count==1


@pytest.mark.pyromind_sdk
def test_successful_admission_cannot_extend_total_startup_deadline(monkeypatch):
    clock=[0.0]
    monkeypatch.setattr('sforge.harness.backend.pyromind_backend.time.monotonic',lambda:clock[0])
    client=client_with_result(output='0')
    client.create.return_value=SimpleNamespace(id='sb-one',name='judge')
    client.get_sandbox.return_value.status='Running'
    def slow_admission(_):clock[0]=301
    backend=PyroMindBackend(client,on_started=slow_admission)
    handle=backend.create_container('image','judge')
    with pytest.raises(TimeoutError,match='admission exceeded'):backend.start_container(handle)
    assert client.create.call_count==1


class AcceptedButLostCreate:
    def __init__(self, journal, *, replacement=False):
        self.journal, self.replacement = journal, replacement
        self.requests, self.active, self.deleted = [], {}, []
        self.visible = False
        self.after_create = None

    def create(self, request):
        import json
        # This assertion runs at dispatch, before any accepted remote resource.
        events = [json.loads(line) for line in self.journal.read_text().splitlines()]
        assert events[-1]['event'] == 'creation_intent' and events[-1]['name'] == request.name
        self.requests.append(request)
        identifier = 'sb-created-' + str(len(self.requests))
        sandbox = SimpleNamespace(id=identifier, name=request.name, status='Running')
        self.active[identifier] = sandbox
        if self.after_create:
            self.after_create()
        if not self.replacement or len(self.requests) > 1:
            raise TimeoutError('Accepted create response was lost')
        return sandbox

    def get_sandbox(self, identifier):
        from pyromind_sdk.client.base import PyroMindAPIError
        if identifier not in self.active:
            raise PyroMindAPIError('absent', status_code=404)
        return self.active[identifier]

    def list(self):
        return list(self.active.values()) if self.visible else []

    def exec_command(self, *args, **kwargs):
        return SimpleNamespace(output='0', stderr='', returncode=0)

    def pause(self, identifier):
        self.active[identifier].status = 'Paused'

    def delete(self, identifier):
        self.deleted.append(identifier)
        self.active.pop(identifier)


@pytest.mark.parametrize('replacement', [False, True])
@pytest.mark.pyromind_sdk
def test_unknown_initial_and_replacement_creates_survive_empty_list_and_controller_restart(tmp_path, replacement):
    import json
    journal = tmp_path / 'events.jsonl'
    client = AcceptedButLostCreate(journal, replacement=replacement)
    def reject(_):
        raise SandboxPlacementRejected('nvlink-pair-required', {'links': ['PXB', 'PXB']})
    backend = PyroMindBackend(client, event_path=journal, on_started=reject)
    if replacement:
        handle = backend.create_container('image', 'sf-owned-judge')
        with pytest.raises(TimeoutError):
            backend.start_container(handle)
        assert handle.id == 'sb-created-1'
        backend.cleanup_container(handle)
    else:
        with pytest.raises(TimeoutError):
            backend.create_container('image', 'sf-owned-judge')
    name = client.requests[-1].name
    assert backend.unresolved_creation_names() == [name]
    receipt = backend.reconcile_creation_intents()
    assert receipt['confirmed_absent_by_get_404'] is False
    assert receipt['unresolved_creation_names'] == [name]
    assert 'unresolved-create:' + name in receipt['cleanup_errors']
    restarted = PyroMindBackend(client, event_path=journal)
    assert restarted.unresolved_creation_names() == [name]
    assert restarted.reconcile_creation_intents()['confirmed_absent_by_get_404'] is False
    client.visible = True
    cleaned = restarted.reconcile_creation_intents()
    assert cleaned == {'remaining_sandbox_ids': [], 'unresolved_creation_names': [],
                       'cleanup_errors': [], 'confirmed_absent_by_get_404': True}
    assert client.active == {}
    assert backend.unresolved_creation_names() == []
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    resolved = [event for event in events if event['event'] == 'creation_resolved' and event['name'] == name]
    assert len(resolved) == 1 and resolved[0]['id'] == client.deleted[-1]
    assert journal.stat().st_mode & 0o777 == 0o600


@pytest.mark.pyromind_sdk
def test_creation_intent_write_failure_prevents_dispatch(tmp_path, monkeypatch):
    client = Mock()
    backend = PyroMindBackend(client, event_path=tmp_path / 'events.jsonl')
    def fail(*args, **kwargs):
        raise OSError('ownership fsync fault')
    monkeypatch.setattr(backend, '_creation_event', fail)
    with pytest.raises(OSError):
        backend.create_container('image', 'judge')
    client.create.assert_not_called()


@pytest.mark.pyromind_sdk
def test_accepted_sid_remains_owned_when_resolution_journal_write_fails(tmp_path, monkeypatch):
    journal = tmp_path / 'events.jsonl'
    client = AcceptedButLostCreate(journal, replacement=True)
    backend = PyroMindBackend(client, event_path=journal)
    original = backend._creation_event
    def fail_resolution(event, *args, **kwargs):
        if event == 'creation_resolved':
            raise OSError('accepted SID fsync fault')
        return original(event, *args, **kwargs)
    monkeypatch.setattr(backend, '_creation_event', fail_resolution)
    with pytest.raises(OSError):
        backend.create_container('image', 'judge')
    # No LIST visibility is needed: the exact accepted SID is retained in memory.
    assert backend.unresolved_creation_names() == []
    receipt = backend.reconcile_creation_intents()
    assert receipt['confirmed_absent_by_get_404'] is True and client.active == {}
    assert PyroMindBackend(client, event_path=journal).unresolved_creation_names() == []


@pytest.mark.pyromind_sdk
def test_unknown_creation_never_adopts_a_duplicate_or_foreign_name(tmp_path):
    client = AcceptedButLostCreate(tmp_path / 'events.jsonl')
    backend = PyroMindBackend(client, event_path=client.journal)
    with pytest.raises(TimeoutError):
        backend.create_container('image', 'judge')
    original = next(iter(client.active.values()))
    client.list = lambda: [original, SimpleNamespace(id='sb-foreign', name=original.name)]
    assert backend.reconcile_creation_intents()['unresolved_creation_names'] == [original.name]
    assert client.deleted == []


@pytest.mark.pyromind_sdk
def test_unknown_create_cleanup_requires_get404_after_successful_delete(tmp_path):
    client = AcceptedButLostCreate(tmp_path / 'events.jsonl')
    backend = PyroMindBackend(client, event_path=client.journal)
    with pytest.raises(TimeoutError):
        backend.create_container('image', 'judge')
    client.visible = True
    client.delete = lambda identifier: client.deleted.append(identifier)
    receipt = backend.reconcile_creation_intents()
    assert receipt['confirmed_absent_by_get_404'] is False
    assert receipt['remaining_sandbox_ids'] == ['sb-created-1']
    assert receipt['cleanup_errors'] and client.active


@pytest.mark.pyromind_sdk
def test_create_cleanup_rejects_foreign_sid_name_from_stale_list(tmp_path):
    client = AcceptedButLostCreate(tmp_path / 'events.jsonl')
    backend = PyroMindBackend(client, event_path=client.journal)
    with pytest.raises(TimeoutError):
        backend.create_container('image', 'judge')
    name = client.requests[0].name
    client.list = lambda: [SimpleNamespace(id='sb-foreign', name=name)]
    client.get_sandbox = lambda _: SimpleNamespace(id='sb-foreign', name='foreign-owner', status='Running')
    receipt = backend.reconcile_creation_intents()
    assert receipt['confirmed_absent_by_get_404'] is False
    assert receipt['remaining_sandbox_ids'] == ['sb-foreign']
    assert client.deleted == []


@pytest.mark.pyromind_sdk
def test_late_cancelled_replacement_records_and_cleans_exact_returned_sid(tmp_path):
    import threading
    client = AcceptedButLostCreate(tmp_path / 'events.jsonl', replacement=True)
    cancelled = threading.Event()
    def reject(_):
        raise SandboxPlacementRejected('nvlink-pair-required', {})
    backend = PyroMindBackend(client, event_path=client.journal, on_started=reject)
    backend.startup_cancelled = cancelled.is_set
    handle = backend.create_container('image', 'judge')
    original_create = client.create
    def accepted(request):
        client.after_create = cancelled.set
        try:
            original_create(request)
        except TimeoutError:
            return list(client.active.values())[-1]
    client.create = accepted
    with pytest.raises(RuntimeError, match='stopping'):
        backend.start_container(handle)
    assert handle.id == 'sb-created-2' and client.active == {}
    assert backend.unresolved_creation_names() == []
    assert backend.reconcile_creation_intents()['confirmed_absent_by_get_404'] is True


@pytest.mark.pyromind_sdk
def test_invalid_service_port_fails_before_any_sandbox_is_created():
    client = Mock()
    backend = PyroMindBackend(client, port_mappings=[{'container_port': 70000}])
    with pytest.raises(ValueError):
        backend.create_container('image', 'model')
    client.create.assert_not_called()


@pytest.mark.pyromind_sdk
def test_legacy_backend_exposes_no_additional_service_ports():
    client = Mock()
    client.create.return_value = SimpleNamespace(id='sb-worker', name='worker')
    PyroMindBackend(client).create_container('image', 'worker')
    assert client.create.call_args.args[0].port_mappings is None


def test_unknown_exit_status_is_an_infrastructure_error():
    backend = PyroMindBackend(client_with_result(-1, error="pod not ready"))
    with pytest.raises(RuntimeError, match="pod not ready"):
        backend.exec_run(PyroMindHandle("sb-one", "worker"), "true")


@pytest.mark.parametrize("option", [
    {"network_mode": "none"}, {"cap_drop": ["ALL"]},
    {"extra_hosts": {"private": "1.2.3.4"}}, {"user": "nobody"},
    {"annotations": {"binding": "lease"}}, {"command": "arbitrary command"},
    {"accelerator_allocation": object()},
])
@pytest.mark.pyromind_sdk
def test_unsupported_guarantees_fail_before_creating_a_resource(option):
    client = Mock()
    backend = PyroMindBackend(client)
    with pytest.raises(NotImplementedError):
        backend.create_container("image", "worker", **option)
    client.create.assert_not_called()


def test_only_explicit_exec_environment_is_forwarded(monkeypatch):
    monkeypatch.setenv("PYROMIND_API_KEY", "controller-only-secret")
    client = client_with_result()
    backend = PyroMindBackend(client)
    handle = PyroMindHandle("sb-one", "worker", {"SFORGE_TOKEN": "session-token"})
    backend.exec_run(handle, ["printf", "%s", "$(touch /bad)"], environment={"DATA": "a b"})
    argv = client.exec_command.call_args.args[1]
    assert "controller-only-secret" not in str(argv)
    assert "SFORGE_TOKEN=session-token" in argv
    assert argv[-1] == "printf %s '$(touch /bad)'"


def test_timeout_does_not_report_success_and_uses_process_group_timeout(tmp_path):
    client = client_with_result()
    client.exec_command.side_effect = [
        SimpleNamespace(returncode=0, output="", stderr="", exception_info=""),
        SimpleNamespace(returncode=0, output="124", stderr="", exception_info=""),
    ]
    client.read_file.return_value = b"partial"
    backend = PyroMindBackend(client)
    result = backend.exec_run_with_timeout(PyroMindHandle("sb-one", "judge"), "sleep 99", 1,
                                          log_file=tmp_path / "exec.log")
    assert result.timed_out and result.exit_code == 124
    assert "timeout --signal=TERM --kill-after=5 1" in client.exec_command.call_args_list[0].args[1]
    assert all(call.kwargs["timeout"] == 10 for call in client.exec_command.call_args_list)
    assert (tmp_path / "exec.log").read_text() == "partial"


@pytest.mark.pyromind_sdk
def test_cleanup_pauses_then_deletes_only_exact_id():
    client = Mock()
    client.get_sandbox.return_value.status = "Running"
    backend = PyroMindBackend(client)
    backend.cleanup_container(PyroMindHandle("sb-one", "worker"))
    assert [call[0] for call in client.method_calls] == ["get_sandbox", "pause", "delete"]
    client.delete.assert_called_once_with("sb-one")


@pytest.mark.parametrize("missing_at", ["get_sandbox", "pause", "delete"])
@pytest.mark.pyromind_sdk
def test_cleanup_is_idempotent_when_concurrent_removal_returns_404(missing_at, tmp_path):
    import json
    from pyromind_sdk.client.base import PyroMindAPIError
    client = Mock()
    client.get_sandbox.return_value.status = "Running"
    getattr(client, missing_at).side_effect = PyroMindAPIError("missing exact sandbox", status_code=404)
    events = tmp_path / "events.jsonl"
    PyroMindBackend(client, event_path=events).cleanup_container(PyroMindHandle("sb-exact", "judge"))
    event = json.loads(events.read_text())
    assert event["event"] == "deleted" and event["id"] == "sb-exact"
    assert all(call.args[0] == "sb-exact" for call in client.method_calls)
    if missing_at == "get_sandbox":
        client.pause.assert_not_called()
        client.delete.assert_not_called()
    else:
        client.delete.assert_called_once_with("sb-exact")


@pytest.mark.parametrize("failing_at", ["get_sandbox", "pause", "delete"])
@pytest.mark.parametrize("status_code", [403, 409, 500, None])
@pytest.mark.pyromind_sdk
def test_cleanup_never_swallows_non404_failures(failing_at, status_code, tmp_path):
    from pyromind_sdk.client.base import PyroMindAPIError
    client = Mock()
    client.get_sandbox.return_value.status = "Running"
    getattr(client, failing_at).side_effect = PyroMindAPIError("cleanup failed", status_code=status_code)
    events = tmp_path / "events.jsonl"
    with pytest.raises(PyroMindAPIError, match="cleanup failed"):
        PyroMindBackend(client, event_path=events).cleanup_container(PyroMindHandle("sb-exact", "judge"))
    assert not events.exists()


def test_artifact_extraction_rejects_symlinks_and_traversal(tmp_path):
    import io
    import tarfile
    for name, kind in [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
            archive.addfile(entry)
        with pytest.raises(ValueError, match="Unsafe"):
            safe_extract(buffer.getvalue(), tmp_path / "export")


@pytest.mark.pyromind_sdk
def test_resource_shape_is_checked_before_api_call():
    client = Mock()
    for cpu, memory in ((0, "8Gi"), (4, "16"), (4, "0Gi"), (2.5, "8Gi")):
        with pytest.raises(ValueError, match="whole number"):
            PyroMindBackend(client).create_container("image", "worker", cpu_limit=cpu, mem_limit=memory)
    client.create.assert_not_called()


@pytest.mark.pyromind_sdk
def test_cpu_and_memory_are_requested_independently():
    client = Mock()
    client.create.return_value = SimpleNamespace(id="sb-0123456789ab", name="worker")
    PyroMindBackend(client).create_container("image", "worker", cpu_limit=8, mem_limit="64Gi")
    resources = client.create.call_args.args[0].resources.model_dump()
    assert (resources["cpu"], resources["memory"]) == ("8", "64Gi")


@pytest.mark.pyromind_sdk
def test_file_upload_uses_the_installed_sdk_api(tmp_path):
    from unittest.mock import create_autospec
    from pyromind_sdk.client.sandbox import SandboxClient
    client = create_autospec(SandboxClient, instance=True)
    client.exec_command.return_value = SimpleNamespace(output="", stderr="", returncode=0)
    source = tmp_path / "payload.txt"
    source.write_text("hello")
    backend = PyroMindBackend(client)
    backend.copy_to_container(PyroMindHandle("sb-one", "worker"), source, Path("/tmp/payload.txt"))
    client.write_file.assert_called_once_with("sb-one", "/tmp/payload.txt", source)


def test_cwd_and_argv_are_quoted_without_sdk_cwd_wrapper(tmp_path):
    import subprocess
    work = tmp_path / "space and ' quote"
    work.mkdir()
    client = Mock()
    def execute(identifier, command, **kwargs):
        assert "cwd" not in kwargs
        result = subprocess.run(command, text=True, capture_output=True)
        return SimpleNamespace(output=result.stdout, stderr=result.stderr,
                               returncode=result.returncode, exception_info="")
    client.exec_command.side_effect = execute
    result = PyroMindBackend(client).exec_run(
        PyroMindHandle("sb-one", "worker"), ["pwd"], workdir=str(work))
    assert result.exit_code == 0
    assert result.output.strip() == str(work)


@pytest.mark.pyromind_sdk
def test_explicit_h100_request_is_encoded_in_actual_sdk_model(tmp_path):
    import json
    client = Mock()
    client.create.return_value = SimpleNamespace(id="sb-h100", name="worker")
    events = tmp_path / "events.jsonl"
    backend = PyroMindBackend(client, image_map={"task-work": {
        "image": "registry/work@sha256:abc", "gpu_count": 1, "gpu_card": "H100",
    }}, event_path=events)
    handle = backend.create_container("task-work", "worker", cpu_limit=16, mem_limit="32Gi",
                                      environment={"SFORGE_TOKEN": "private-session"})
    request = client.create.call_args.args[0]
    assert request.image == "registry/work@sha256:abc"
    assert request.resources.model_dump() == {
        "cpu": "16", "memory": "32Gi", "gpu": "1", "gpu_card": "H100",
    }
    assert handle.id == "sb-h100"
    created = [json.loads(line) for line in events.read_text().splitlines()
               if json.loads(line)['event'] == 'created']
    assert len(created) == 1 and created[0]["gpu_card"] == "H100"
    assert "private-session" not in events.read_text()


@pytest.mark.pyromind_sdk
def test_gpu_defaults_and_per_image_cpu_override(monkeypatch):
    monkeypatch.setenv("SFORGE_PYROMIND_GPU_COUNT", "2")
    monkeypatch.setenv("SFORGE_PYROMIND_GPU_CARD", "H100")
    client = Mock()
    client.create.return_value = SimpleNamespace(id="sb-resource", name="resource")
    backend = PyroMindBackend(client, image_map={"cpu": {"image": "cpu", "gpu_count": 0}})
    backend.create_container("gpu", "gpu")
    assert client.create.call_args.args[0].resources.gpu == "2"
    backend.create_container("cpu", "cpu")
    assert client.create.call_args.args[0].resources.gpu is None
    assert client.create.call_args.args[0].resources.gpu_card is None
    backend = PyroMindBackend(client, gpu_count=1, gpu_card="L40S")
    backend.create_container("gpu", "gpu")
    assert client.create.call_args.args[0].resources.gpu == "1"
    assert client.create.call_args.args[0].resources.gpu_card == "L40S"


@pytest.mark.parametrize("count,card", [(-1, "H100"), (True, "H100"), (1.5, "H100"),
                                      (1, None), (1, " ")])
@pytest.mark.pyromind_sdk
def test_invalid_gpu_request_creates_no_resource(count, card):
    client = Mock()
    backend = PyroMindBackend(client, gpu_count=count, gpu_card=card)
    with pytest.raises(ValueError, match="gpu_"):
        backend.create_container("image", "worker")
    client.create.assert_not_called()


@pytest.mark.parametrize("secret_name", ["PYROMIND_API_KEY", "PYROMIND_STORAGE_SECRET_KEY"])
@pytest.mark.pyromind_sdk
def test_platform_credentials_cannot_be_forwarded_even_explicitly(secret_name):
    client = Mock()
    backend = PyroMindBackend(client)
    with pytest.raises(ValueError, match="stay on the controller"):
        backend.create_container("image", "worker", environment={secret_name: "local-only"})
    with pytest.raises(ValueError, match="stay on the controller"):
        backend.exec_run(PyroMindHandle("sb-one", "worker"), "true",
                         environment={secret_name: "local-only"})
    client.create.assert_not_called()
    client.exec_command.assert_not_called()


def test_platform_key_cannot_be_forwarded_under_an_alias(monkeypatch):
    monkeypatch.setenv("PYROMIND_API_KEY", "platform-local-only-key")
    client = Mock()
    with pytest.raises(ValueError, match="stay on the controller"):
        PyroMindBackend(client).exec_run(PyroMindHandle("sb-one", "worker"), "true",
                                       environment={"UNRELATED": "platform-local-only-key"})
    client.exec_command.assert_not_called()


def test_long_deadline_uses_short_api_calls_without_truncating_remote_timeout():
    client = client_with_result()
    client.exec_command.side_effect = [
        SimpleNamespace(returncode=0, output="", stderr="", exception_info=""),
        SimpleNamespace(returncode=0, output="0", stderr="", exception_info=""),
    ]
    client.read_file.return_value = b"finished"
    result = PyroMindBackend(client).exec_run_with_timeout(
        PyroMindHandle("sb-one", "worker"), "agent run", timeout=21600)
    assert result.exit_code == 0 and not result.timed_out
    assert "timeout --signal=TERM --kill-after=5 21600" in client.exec_command.call_args_list[0].args[1]
    assert all(call.kwargs["timeout"] == 10 for call in client.exec_command.call_args_list)


@pytest.mark.parametrize("timeout", [None, False, 0, -1, float("inf"), float("nan"), "21600"])
def test_unbounded_or_invalid_timeout_rejected_before_launch(timeout):
    client = Mock()
    with pytest.raises(ValueError, match="finite timeout"):
        PyroMindBackend(client).exec_run_with_timeout(
            PyroMindHandle("sb-one", "worker"), "true", timeout=timeout)
    client.exec_command.assert_not_called()


def test_transport_error_attempts_group_cleanup_and_redacts_env():
    secret = "private-model-api-token"
    client = client_with_result()
    client.exec_command.side_effect = [
        RuntimeError(f"request failed with DEEPSEEK_API_KEY={secret}"),
        SimpleNamespace(returncode=0, output="", stderr="", exception_info=""),
    ]
    backend = PyroMindBackend(client)
    with pytest.raises(RuntimeError) as caught:
        backend.exec_run_with_timeout(PyroMindHandle("sb-one", "worker"), "agent run", 21600,
                                      environment={"DEEPSEEK_API_KEY": secret})
    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "kill -TERM" in client.exec_command.call_args_list[1].args[1][-1]
    assert "kill -KILL" in client.exec_command.call_args_list[1].args[1][-1]
    client.delete_file.assert_called_once()


def test_unreachable_job_keeps_metadata_and_does_not_mask_original_failure():
    client = client_with_result()
    client.exec_command.side_effect = [
        SimpleNamespace(returncode=0, output="", stderr="", exception_info=""),
        RuntimeError("poll failed"), RuntimeError("cleanup failed"),
    ]
    with pytest.raises(RuntimeError, match="poll failed"):
        PyroMindBackend(client).exec_run_with_timeout(
            PyroMindHandle("sb-one", "worker"), "agent run", 21600)
    client.delete_file.assert_not_called()


def local_process_client():
    """Exercise the actual generated shell and timeout without a cloud sandbox."""
    import shutil
    import subprocess
    client = Mock()
    def execute(identifier, command, **kwargs):
        result = subprocess.run(command, shell=isinstance(command, str),
                                executable="/bin/bash" if isinstance(command, str) else None,
                                text=True, capture_output=True, timeout=kwargs["timeout"])
        return SimpleNamespace(output=result.stdout, stderr=result.stderr,
                               returncode=result.returncode, exception_info="")
    client.exec_command.side_effect = execute
    client.read_file.side_effect = lambda identifier, path: Path(path).read_bytes()
    client.delete_file.side_effect = lambda identifier, path, **kwargs: shutil.rmtree(path)
    return client


def assert_process_exited(pid):
    import time
    for _ in range(20):
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split(") ", 1)[1].startswith("Z"):
            return
        time.sleep(0.1)
    pytest.fail(f"process {pid} survived process-group cleanup")


def test_actual_timeout_kills_descendants_and_preserves_partial_output(tmp_path):
    import shlex
    child_file = tmp_path / "child.pid"
    client = local_process_client()
    result = PyroMindBackend(client).exec_run_with_timeout(
        PyroMindHandle("sb-local", "judge"),
        f"sleep 90 & echo $! >{shlex.quote(str(child_file))}; echo partial-output; wait", 1)
    assert result.exit_code == 124 and result.timed_out
    assert "partial-output" in result.output
    assert_process_exited(int(child_file.read_text()))
    assert not Path(client.delete_file.call_args.args[1]).exists()


def test_successful_command_cannot_leave_background_descendants(tmp_path):
    import shlex
    child_file = tmp_path / "child.pid"
    result = PyroMindBackend(local_process_client()).exec_run_with_timeout(
        PyroMindHandle("sb-local", "worker"),
        f"sleep 90 & echo $! >{shlex.quote(str(child_file))}; echo done", 60)
    assert result.exit_code == 0 and not result.timed_out
    assert result.output == "done\n"
    assert_process_exited(int(child_file.read_text()))


def test_shutdown_cancels_long_command_group_and_cleans_metadata(tmp_path):
    import shlex
    import threading
    import time
    child_file = tmp_path / "child.pid"
    shutdown = threading.Event()
    def request_shutdown():
        deadline = time.monotonic() + 10
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        shutdown.set()
    thread = threading.Thread(target=request_shutdown, daemon=True)
    thread.start()
    client = local_process_client()
    result = PyroMindBackend(client).exec_run_with_timeout(
        PyroMindHandle("sb-local", "worker"),
        f"sleep 90 & echo $! >{shlex.quote(str(child_file))}; echo started; wait", 21600,
        shutdown_event=shutdown)
    thread.join(timeout=1)
    assert result.exit_code == 130 and not result.timed_out
    assert "started" in result.output
    assert_process_exited(int(child_file.read_text()))
    assert not Path(client.delete_file.call_args.args[1]).exists()
