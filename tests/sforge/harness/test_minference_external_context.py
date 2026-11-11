"""Task-scoped host metadata handoff; no container/GPU/qualification fixtures."""
from types import SimpleNamespace
import pytest
from sforge.harness.run_evaluation import merge_minference_external_context

PROTOCOL='minference-ruler-quality-v3'
VALUES={'EXTERNAL_ASSET_BINDINGS_SHA256':'a'*64,'EXTERNAL_ASSET_RECEIPT_SHA256':'b'*64}

def test_trusted_on_started_handle_hashes_are_copied_without_extra_fields():
    values=dict(VALUES);context={'EXISTING_HOST_BINDING':'kept'}
    merge_minference_external_context(context,SimpleNamespace(minference_external_context=values),PROTOCOL)
    assert context=={'EXISTING_HOST_BINDING':'kept',**VALUES}
    values['EXTERNAL_ASSET_BINDINGS_SHA256']='c'*64
    assert context['EXTERNAL_ASSET_BINDINGS_SHA256']=='a'*64

def test_other_tasks_are_unmodified_without_native_handle_metadata():
    context={'value':'unchanged'}
    merge_minference_external_context(context,SimpleNamespace(),None)
    merge_minference_external_context(context,SimpleNamespace(),'another-task-bootstrap')
    assert context=={'value':'unchanged'}

@pytest.mark.parametrize('values',[None,{}, {'EXTERNAL_ASSET_BINDINGS_SHA256':'a'*64},
    {**VALUES,'extra':'c'*64},{**VALUES,'EXTERNAL_ASSET_RECEIPT_SHA256':True},
    {**VALUES,'EXTERNAL_ASSET_RECEIPT_SHA256':'sha256:'+'b'*64},
    {**VALUES,'EXTERNAL_ASSET_RECEIPT_SHA256':'B'*64}])
def test_missing_extra_or_invalid_identity_rejected_without_context_mutation(values):
    context={'host':'original'}
    with pytest.raises(ValueError):merge_minference_external_context(context,
        SimpleNamespace(minference_external_context=values),PROTOCOL)
    assert context=={'host':'original'}

def test_conflicting_host_hash_rejected_before_bootstrap():
    context={'EXTERNAL_ASSET_RECEIPT_SHA256':'c'*64}
    with pytest.raises(ValueError,match='conflicts'):merge_minference_external_context(context,
        SimpleNamespace(minference_external_context=VALUES),PROTOCOL)
    assert context=={'EXTERNAL_ASSET_RECEIPT_SHA256':'c'*64}
