import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest
import requests

from sforge.harness.relay.model_proxy import ModelResponsesProxy, load_provider_key, pin_native_request


def test_bare_and_named_credential_files(tmp_path):
    path=tmp_path/'private.key'
    path.write_text('fake-ali-token\n'); path.chmod(0o600)
    assert load_provider_key(path)=='fake-ali-token'
    path.write_text('DASHSCOPE_API_KEY=fake-ali-token\n')
    assert load_provider_key(path)=='fake-ali-token'
    path.write_text('API_KEY=one\nOPENAI_API_KEY=two\n')
    with pytest.raises(ValueError):load_provider_key(path)


@pytest.mark.parametrize('stream',[False,True])
def test_native_relay_pin_auth_gzip_tools_receipt_and_error_redaction(tmp_path,stream):
    records=[]
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_POST(self):
            doc=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            records.append((self.path,self.headers.get('Authorization'),doc))
            if doc.get('input')=='error':
                self.send_response(400); self.end_headers()
                self.wfile.write(b'{"error":{"message":"upstream-secret private prompt"}}');return
            result={'object':'response','status':'completed','output':[],
                    'usage':{'input_tokens':2,'output_tokens':3,'secret':'upstream-secret'}}
            self.send_response(200);self.send_header('Content-Type','text/event-stream' if stream else 'application/json');self.end_headers()
            self.wfile.write(('data: '+json.dumps({'type':'response.completed','response':result})+'\n\n').encode()
                             if stream else json.dumps(result).encode())
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
    proxy=ModelResponsesProxy(api_key='upstream-secret',relay_token='worker-token',model='exact-model',
        upstream_base=f'http://127.0.0.1:{upstream.server_port}',log_file=tmp_path/'usage.jsonl',
        vendor='alicloud',port=0)
    proxy.start()
    try:
        url=f'http://127.0.0.1:{proxy.port}/v1/responses'
        assert requests.post(url,json={'input':'hi'},timeout=3).status_code==401
        doc={'model':'wrong','input':'private prompt','stream':stream,
             'tools':[{'type':'function','name':'exec_command','parameters':{'type':'object'}}]}
        r=requests.post(url,headers={'Authorization':'Bearer worker-token','Content-Encoding':'gzip'},
                        data=gzip.compress(json.dumps(doc).encode()),timeout=3)
        assert r.status_code==200
        assert records[0][0]=='/responses' and records[0][1]=='Bearer upstream-secret'
        assert records[0][2]['model']=='exact-model' and records[0][2]['reasoning']['effort']=='max'
        assert records[0][2]['tools']==doc['tools']
        r=requests.post(url,headers={'Authorization':'Bearer worker-token'},json={'input':'error'},timeout=3)
        assert r.status_code==400 and 'upstream-secret' not in r.text
        proxy.log.flush()
        text=proxy.log_file.read_text()
        assert 'private prompt' not in text and 'upstream-secret' not in text and 'worker-token' not in text
        assert json.loads(text.splitlines()[0])['usage']=={'input_tokens':2,'output_tokens':3}
    finally:
        proxy.stop();upstream.shutdown();upstream.server_close()


def test_openrouter_provider_pin_replaces_caller_routing():
    document={'model':'other','provider':{'only':['cheap'],'allow_fallbacks':True},
              'tools':[{'type':'web_search'}],'reasoning':{'effort':'low','max_tokens':5}}
    pinned=pin_native_request(document,'google/gemini-4-argon','max','openrouter','google-vertex/global')
    assert pinned['model']=='google/gemini-4-argon'
    assert pinned['provider']=={'only':['google-vertex/global'],'allow_fallbacks':False}
    assert pinned['reasoning']=={'effort':'max','enabled':True,'exclude':False}
    assert pinned['tools']==[{'type':'web_search'}]
    unpinned=pin_native_request(document,'google/gemini-4-argon','max','openrouter')
    assert 'provider' not in unpinned
    assert 'provider' not in pin_native_request({'input':'x'},'qwen','max','alicloud')


def test_provider_pin_is_openrouter_only(tmp_path):
    with pytest.raises(ValueError):
        ModelResponsesProxy(api_key='k',relay_token='t',model='m',upstream_base='http://x',
                            log_file=tmp_path/'u.jsonl',vendor='alicloud',provider='google-vertex/global')
