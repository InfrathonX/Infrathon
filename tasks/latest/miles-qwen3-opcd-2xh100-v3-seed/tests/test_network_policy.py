"""CPU-only checks of the Work network policy chosen by the task JSON's "internet" field."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

TASK = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load("opcd_network_policy_launch", TASK / "scripts/run_task.py")
entry = load("opcd_network_policy_entry", TASK / "runtime/sforge_entry.py")
WORK_NAME = f"sforge.run.{entry.TASK_ID}.test-run"
JUDGE_NAME = f"{entry.BENCH}.judge.{entry.TASK_ID}.submission"


def test_task_json_allows_internet_and_parser_defaults(tmp_path):
    assert json.loads(launch.TASK_JSON.read_text())["internet"] is True
    assert launch.task_allows_internet() is True
    for body, expected in (({}, True), ({"internet": False}, False)):
        path = tmp_path / "task.json"
        path.write_text(json.dumps(body))
        assert launch.task_allows_internet(path) is expected
    path.write_text(json.dumps({"internet": 1}))
    with pytest.raises(launch.NotReady, match="boolean"):
        launch.task_allows_internet(path)


class FakeDocker:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, check=True, capture=True):
        self.calls.append(args)
        stdout = ""
        if args[0] == "inspect":
            network = next(a for a in self.calls if a[:2] == ("network", "create"))[-1]
            stdout = json.dumps([{"NetworkSettings": {"Networks": {
                network: {"IPAddress": "172.31.0.2", "IPPrefixLen": 24}}}}])
        elif args[:2] == ("network", "inspect"):
            stdout = json.dumps([{"IPAM": {"Config": [{"Subnet": "172.31.0.0/24", "Gateway": "172.31.0.1"}]}}])
        return SimpleNamespace(stdout=stdout, returncode=0)

    def relay_program(self):
        run = next(a for a in self.calls if a[0] == "run")
        return list(run[run.index("python3") + 2:])


@pytest.fixture
def relay_env(monkeypatch):
    fake, probes = FakeDocker(), []
    monkeypatch.setattr(launch, "docker", fake)

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    monkeypatch.setattr(launch.urllib.request, "build_opener",
                        lambda *a: SimpleNamespace(open=lambda url, timeout: Response()))
    monkeypatch.setattr(launch, "check_connect_policy",
                        lambda ip, internet, gateway=None: probes.append((ip, internet, gateway)) or {"probed": internet})
    return fake, probes


CONFIGS = [{"MODE": "endpoint", "MODEL_API_BASE_URL": "http://10.1.2.3:8000/v1", "HARNESS_ROOT": "/harness"},
           {"MODE": "chatgpt", "HARNESS_ROOT": "/harness"}]


@pytest.mark.parametrize("cfg", CONFIGS, ids=["endpoint", "chatgpt"])
def test_internet_relay_opens_public_egress_and_gives_work_the_proxy(relay_env, cfg):
    fake, probes = relay_env
    with launch.isolated_relay(cfg, "r1", 18600, "work-image", internet=True) as relay:
        pass
    assert ("network", "create", "--internal", "--label", "opcd.run=r1", "opcd-r1") in fake.calls
    assert fake.relay_program()[-1] == "--allow-public-egress"
    assert probes == [("172.31.0.2", True, "172.31.0.1")]
    assert relay["public_egress"] is True and relay["connect_probes"] == {"probed": True}
    proxy_env = relay["work_proxy_env"]
    assert proxy_env["SFORGE_HTTP_PROXY"] == proxy_env["SFORGE_HTTPS_PROXY"] == "http://172.31.0.2:3128"
    direct = proxy_env["SFORGE_NO_PROXY"].split(",")
    for name in ("172.31.0.2", "opcd-relay", "localhost", "127.0.0.1", "::1", "opcd-work", "172.31.0.0/24"):
        assert name in direct
    assert relay["judge_url"] == "http://172.31.0.2:3130"


@pytest.mark.parametrize("cfg", CONFIGS, ids=["endpoint", "chatgpt"])
def test_offline_relay_keeps_the_allowlist_and_no_work_proxy(relay_env, cfg):
    fake, probes = relay_env
    with launch.isolated_relay(cfg, "r2", 18600, "work-image") as relay:
        pass
    assert "--allow-public-egress" not in fake.relay_program()
    assert probes == [("172.31.0.2", False, "172.31.0.1")]
    assert relay["public_egress"] is False and relay["work_proxy_env"] == {}


def test_connect_policy_probes_are_mode_dependent():
    status = lambda ip, authority: 200 if authority == "github.com:443" else 403
    observed = launch.check_connect_policy("172.31.0.2", True, status=status, gateway="172.31.0.1")
    assert observed["github.com:443"] == 200
    assert {"localhost:443", "host.docker.internal:443", "169.254.169.254:80", "github.com:22",
            "172.31.0.1:443"} <= set(observed)
    assert {v for k, v in observed.items() if k != "github.com:443"} == {403}
    with pytest.raises(launch.NotReady, match="172.31.0.1:443 returned 200"):
        launch.check_connect_policy("172.31.0.2", True, status=lambda ip, a: 200 if a != "github.com:22" and
                                    not a.startswith(("localhost", "host.docker", "169.254")) else 403,
                                    gateway="172.31.0.1")
    with pytest.raises(launch.NotReady, match="expected 403"):
        launch.check_connect_policy("172.31.0.2", False, status=status)
    with pytest.raises(launch.NotReady, match="expected 200"):
        launch.check_connect_policy("172.31.0.2", True, status=lambda ip, authority: 403)
    assert set(launch.check_connect_policy("172.31.0.2", False, status=lambda ip, a: 403).values()) == {403}


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("OPCD_GPU_UUIDS", "GPU-a,GPU-b")
    monkeypatch.setenv("OPCD_RUN_ID", "test-run")
    monkeypatch.setenv("OPCD_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("OPCD_HF_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("OPCD_MEGATRON_INIT", str(tmp_path / "megatron"))
    monkeypatch.setenv("SFORGE_AGENT_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("OPCD_INTERNAL_NETWORK", "opcd-test-run")
    created = {}

    def create(image, **kwargs):
        created[kwargs["name"]] = kwargs
        return SimpleNamespace(id="container-id", name=kwargs["name"])

    network = SimpleNamespace(attrs={"Internal": True}, reload=lambda: None)
    backend = SimpleNamespace(client=SimpleNamespace(containers=SimpleNamespace(create=create),
                                                     networks=SimpleNamespace(get=lambda name: network)))
    return backend, created


def profile_binds(kwargs):
    return {v["bind"] for v in kwargs["volumes"].values() if v["bind"].startswith("/etc/profile.d/")}


@pytest.mark.parametrize("internet", [True, False])
def test_work_shell_proxy_and_package_index_follow_the_task(configured, monkeypatch, internet):
    backend, created = configured
    monkeypatch.setenv("MLSYSFORGE_WORK_INTERNET", "true" if internet else "false")
    entry.create_container(backend, "work-image", WORK_NAME)
    entry.create_container(backend, "judge-image", JUDGE_NAME)
    work, judge = created[WORK_NAME], created[JUDGE_NAME]
    expected_profile = ("/etc/profile.d/zz-opcd-public-egress.sh" if internet
                        else "/etc/profile.d/zz-opcd-no-proxy.sh")
    assert profile_binds(work) == {expected_profile}
    source = next(s for s, v in work["volumes"].items() if v["bind"] == expected_profile)
    assert Path(source).name == ("public-egress.profile.sh" if internet else "no-proxy.profile.sh")
    assert ("unset HTTP_PROXY" in Path(source).read_text()) is not internet
    assert ("PIP_NO_INDEX" in work["environment"]) is not internet
    assert work["network"] == "opcd-test-run" and work["user"] == "agent"
    assert judge["environment"]["PIP_NO_INDEX"] == "1" and judge["network_mode"] == "none"
    assert profile_binds(judge) == set()
    for env in (work["environment"], judge["environment"]):
        assert env["HF_HUB_OFFLINE"] == "1"


@pytest.fixture
def installed(monkeypatch):
    from sforge.harness.agent.codex import CodexAgent
    from sforge.harness.backend.docker_backend import DockerBackend
    from sforge.harness import judge_server, run_agent
    seen = {"format": [], "prompt": []}
    monkeypatch.setattr(DockerBackend, "create_container", DockerBackend.create_container)
    monkeypatch.setattr(judge_server, "judge_submission", judge_server.judge_submission)
    monkeypatch.setattr(CodexAgent, "format_run_cmd",
                        lambda self, *a, **kw: seen["format"].append(kw["internet"]) or "codex exec --x")
    monkeypatch.setattr(run_agent, "generate_evolve_prompt",
                        lambda *a, **kw: seen["prompt"].append(kw["internet"]) or "prompt")
    entry.install()
    return seen, CodexAgent, run_agent


@pytest.mark.parametrize("flag,internet", [(None, True), ("true", True), ("false", False)])
def test_codex_search_and_prompt_follow_the_task(installed, monkeypatch, flag, internet):
    seen, codex, run_agent = installed
    if flag is None:
        monkeypatch.delenv("MLSYSFORGE_WORK_INTERNET", raising=False)
    else:
        monkeypatch.setenv("MLSYSFORGE_WORK_INTERNET", flag)
    command = codex.format_run_cmd(object(), "/tmp/prompt.md", internet=True)
    run_agent.generate_evolve_prompt("query", internet=True)
    assert seen == {"format": [internet], "prompt": [internet]}
    assert "features.shell_snapshot=false" in command


def test_invalid_work_internet_value_fails_closed(monkeypatch):
    monkeypatch.setenv("MLSYSFORGE_WORK_INTERNET", "1")
    with pytest.raises(RuntimeError, match="MLSYSFORGE_WORK_INTERNET"):
        entry.work_internet()


def test_brief_and_query_say_nothing_about_the_network():
    task = json.loads(launch.TASK_JSON.read_text())
    for text in ((TASK / "images/work/AGENT_TASK.md").read_text(), task["work"]["agent_query"]):
        lowered = text.lower()
        for phrase in ("internet", "offline", "web search", "no network access"):
            assert phrase not in lowered, phrase


def test_launcher_exports_the_task_flag_to_the_entry():
    source = (TASK / "scripts/run_task.py").read_text()
    assert '"MLSYSFORGE_WORK_INTERNET": "true" if args.internet else "false"' in source
    assert 'isolated_relay(cfg, args.run_id, port, ids["work"], internet=args.internet)' in source
