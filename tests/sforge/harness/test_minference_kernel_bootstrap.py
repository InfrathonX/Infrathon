import copy
from types import SimpleNamespace
import pytest
from sforge.harness.minference_kernel_bootstrap import validate_receipt
G='GPU-6e5ab7c8-39bf-df56-533a-729fb2bcc77b';I='sha256:'+'a'*64;C='b'*64
def receipt():return {'Id':C,'Image':I,'State':{'Running':True},'Config':{'User':'root'},'HostConfig':{'NetworkMode':'none','Privileged':False,'Devices':[],'CapAdd':[],'DeviceRequests':[{'Driver':'nvidia','Count':0,'Capabilities':[['gpu']],'DeviceIDs':[G]}]}}
def allocation():return SimpleNamespace(vendor='nvidia',device_ids=(G,))
def test_actual_one_gpu():assert validate_receipt(receipt(),C,I,allocation())['Id']==C
@pytest.mark.parametrize('damage',['four-gpu','wrong-driver','implicit-allocation','requested-tag','wrong-container','privileged','network','worker-user','wrong-allocation','stopped'])
def test_not_a_relaxed_quad_receipt(damage):
 r=receipt();a=allocation()
 if damage=='four-gpu':r['HostConfig']['DeviceRequests'][0]['DeviceIDs']=[G]*4
 if damage=='wrong-driver':r['HostConfig']['DeviceRequests'][0]['Driver']='amd'
 if damage=='implicit-allocation':r['HostConfig']['DeviceRequests'][0]['Count']=-1
 if damage=='requested-tag':r['Image']='tag:latest'
 if damage=='wrong-container':r['Id']='c'*64
 if damage=='privileged':r['HostConfig']['Privileged']=True
 if damage=='network':r['HostConfig']['NetworkMode']='bridge'
 if damage=='worker-user':r['Config']['User']='agent'
 if damage=='wrong-allocation':a.device_ids=('GPU-'+'c'*8+'-'+'c'*4+'-'+'c'*4+'-'+'c'*4+'-'+'c'*12,)
 if damage=='stopped':r['State']['Running']=False
 with pytest.raises(ValueError):validate_receipt(r,C,I,a)
