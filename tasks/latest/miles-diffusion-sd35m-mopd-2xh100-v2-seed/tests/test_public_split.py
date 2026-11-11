"""CPU-only checks that Work holds a small public workload and the scored workload stays with the Judge."""
import ast
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
JUDGE = REPO / "scripts/miles_diffusion_opd/judge_v2"


def contract(name):
    return json.loads((TASK / "runtime" / name).read_text())


def test_public_contract_is_small_and_otherwise_the_same_recipe():
    public, scored = contract("public_contract.json"), contract("contract.json")
    assert (public["rounds"], public["rows_per_round"]) == (8, 4)
    assert (scored["rounds"], scored["rows_per_round"]) == (64, 8)
    differ = {"rounds", "rows_per_round"}
    assert {k: v for k, v in public.items() if k not in differ} == {k: v for k, v in scored.items() if k not in differ}


def test_work_gets_the_public_contract_and_not_run_judge():
    build = (TASK / "scripts/build_images.sh").read_text()
    assert 'cp "${TASK}/runtime/public_contract.json" "${STAGE}/work/public/judge/contract.json"' in build
    assert '[[ "${f}" == run_judge.py ]] ||' in build
    assert "test ! -e /opt/opd_public/judge/run_judge.py" in (TASK / "images/work/Dockerfile").read_text()


def test_public_runner_does_not_import_the_judge_and_reports_no_score():
    public = (JUDGE / "run_public.py").read_text()
    assert "run_judge" not in public
    for word in ("provisional", "metric_v3", "kl_route_gate", "closure_gate"):
        assert word not in public, word


def test_public_helpers_are_the_judges_verbatim():
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}

    judge, common = functions(JUDGE / "run_judge.py"), functions(JUDGE / "public_common.py")
    assert common
    for name, dump in common.items():
        assert judge[name] == dump, name


def test_published_reference_holds_no_reference_results():
    published = contract("published_reference.json")
    assert set(published) == {"public_heldout", "public_kl", "public_task_quality", "note"}
    text = json.dumps(published)
    for word in ("T_seconds", "reference_seconds", "min_route_gain", "closure"):
        assert word not in text, word


def test_agent_text_does_not_state_the_scored_workload():
    prompt = json.loads((TASK / "tasks/miles_diffusion_sd35m_mopd_2xh100_v2.json").read_text())["work"]["agent_query"]
    texts = {"AGENT_TASK.md": (TASK / "images/work/AGENT_TASK.md").read_text(),
             "INTERFACE.md": (TASK / "docs/INTERFACE.md").read_text(), "agent_query": prompt,
             "starter": (TASK / "images/work/workdir/miles_lab/task_entry.py").read_text()}
    for name, text in texts.items():
        for phrase in ("64-round", "64 rounds", "6 rounds", "three hidden", "DiffusionOPD", "1,000", "provisional", "same two GPUs"):
            assert phrase not in text, (name, phrase)
    assert "only to get the pipeline working" in prompt and "sforge-submit" in prompt
    brief = texts["AGENT_TASK.md"]
    assert brief.index("## Score") < brief.index("## Submission evaluation policy") < brief.index("## Submission workflow")
