from __future__ import annotations

import json

import pytest

from sforge.harness import host_topology as ht
from sforge.harness.host_topology import (
    TopologyError,
    format_cpulist,
    parse_cpulist,
)


H100_TABLE = {
    "0000:46:00.0": (0, "0-3,32-35"),
    "0000:49:00.0": (0, "4-7,36-39"),
    "0000:4e:00.0": (0, "8-11,40-43"),
    "0000:4f:00.0": (0, "12-15,44-47"),
    "0000:c5:00.0": (1, "16-19,48-51"),
    "0000:c6:00.0": (1, "20-23,52-55"),
    "0000:c9:00.0": (1, "24-27,56-59"),
    "0000:d0:00.0": (1, "28-31,60-63"),
}
# sha256 of the canonical {pci: {node, cpus}} partition of the table above;
# the same value the design prototype printed on the real host.
H100_FINGERPRINT = (
    "sha256:3a51f52c156b199240f55ffd06408eedaf5f55c6c6789f180120acc4eb568bc8"
)


def _table(host, reserve: str = ""):
    topo = ht.read_topology(host.sysfs)
    return topo, ht.partition(topo, parse_cpulist(reserve))


def _cpus(table) -> dict[str, tuple[int | None, str]]:
    return {pci: (item.node, format_cpulist(item.cpus)) for pci, item in table.items()}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", frozenset()),
        ("  ", frozenset()),
        ("7", frozenset({7})),
        ("0-3,32-35", frozenset({0, 1, 2, 3, 32, 33, 34, 35})),
        (" 8-11,40-43\n", frozenset({8, 9, 10, 11, 40, 41, 42, 43})),
        ("3,1,2", frozenset({1, 2, 3})),
    ],
)
def test_cpulist_parse(text: str, expected: frozenset[int]) -> None:
    assert parse_cpulist(text) == expected


@pytest.mark.parametrize(
    "text", ["8-", "-3", "a", "1,,2", "1-2-3", "4-1", "1, 2", "0x3", "@0"]
)
def test_cpulist_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError):
        parse_cpulist(text)


@pytest.mark.parametrize(
    "cpus", [set(), {5}, {0, 1, 2, 3, 32, 33, 34, 35}, {1, 3, 5, 6, 7, 9}]
)
def test_cpulist_format_round_trip(cpus: set[int]) -> None:
    text = format_cpulist(cpus)
    assert parse_cpulist(text) == frozenset(cpus)
    assert format_cpulist(parse_cpulist(text)) == text


def test_format_is_canonical() -> None:
    assert format_cpulist({3, 0, 1, 2, 35, 32, 33, 34}) == "0-3,32-35"
    assert format_cpulist([9, 7]) == "7,9"
    assert ht.normalize_cpulist(" 8,9,10,11,40-43 ") == "8-11,40-43"


@pytest.mark.parametrize(
    ("bus_id", "expected"),
    [
        ("00000000:4E:00.0", "0000:4e:00.0"),
        ("0000:c5:00.0", "0000:c5:00.0"),
        ("4f:00.0", "0000:4f:00.0"),
    ],
)
def test_normalize_pci(bus_id: str, expected: str) -> None:
    assert ht.normalize_pci(bus_id) == expected


def test_reference_host_shape_gives_one_l3_group_per_gpu(fake_host) -> None:
    topo, table = _table(fake_host())

    assert _cpus(table) == H100_TABLE
    assert ht.fingerprint(table) == H100_FINGERPRINT
    for item in table.values():
        assert item.cores == 4
        assert item.l3_confined is True
        assert item.l3_groups == (format_cpulist(item.cpus),)
        assert item.node_spare == frozenset()
    # The NVIDIA bridge function and the non-NVIDIA display are not GPUs.
    assert [gpu.pci for gpu in topo.gpus] == sorted(H100_TABLE)


def test_l3_groups_are_ranked_by_lowest_cpu_not_cache_id(fake_host) -> None:
    _, table = _table(fake_host(shuffle_l3_ids=True))

    assert _cpus(table) == H100_TABLE


def test_reserve_keeps_whole_l3_mode_and_leaves_spare_cores(fake_host) -> None:
    _, table = _table(fake_host(), reserve="0,32")

    assert _cpus(table) == {
        **H100_TABLE,
        "0000:46:00.0": (0, "1-3,33-35"),
        "0000:49:00.0": (0, "4-6,36-38"),
        "0000:4e:00.0": (0, "8-10,40-42"),
        "0000:4f:00.0": (0, "12-14,44-46"),
    }
    node0 = [table[pci] for pci in sorted(table) if table[pci].node == 0]
    assert all(item.l3_confined and item.cores == 3 for item in node0)
    assert format_cpulist(node0[0].node_spare) == "7,11,15,39,43,47"


def test_one_l3_per_socket_is_cut_into_contiguous_chunks(fake_host) -> None:
    gpus = [
        ("0000:1a:00.0", 0), ("0000:3d:00.0", 0), ("0000:5e:00.0", 0), ("0000:6b:00.0", 0),
        ("0000:9a:00.0", 1), ("0000:b1:00.0", 1), ("0000:cb:00.0", 1), ("0000:db:00.0", 1),
    ]
    _, table = _table(fake_host(nodes=(28, 28), l3_cores=0, gpus=gpus))

    assert format_cpulist(table["0000:1a:00.0"].cpus) == "0-6,56-62"
    assert format_cpulist(table["0000:6b:00.0"].cpus) == "21-27,77-83"
    assert format_cpulist(table["0000:9a:00.0"].cpus) == "28-34,84-90"
    assert all(len(item.cpus) == 14 for item in table.values())
    # One shared L3 per socket: slices are disjoint but share the L3.
    assert not any(item.l3_confined for item in table.values())


def test_more_l3_groups_than_gpus_gives_each_gpu_whole_groups(fake_host) -> None:
    gpus = [("0000:17:00.0", 0), ("0000:3d:00.0", 0), ("0000:9b:00.0", 1), ("0000:bb:00.0", 1)]
    topo, table = _table(fake_host(nodes=(32, 32), l3_cores=8, gpus=gpus))

    assert format_cpulist(table["0000:17:00.0"].cpus) == "0-15,64-79"
    assert format_cpulist(table["0000:3d:00.0"].cpus) == "16-31,80-95"
    assert all(item.l3_confined and item.cores == 16 for item in table.values())
    # A container budget of 8 keeps 4 cores + SMT inside one L3 group.
    trimmed = ht.trim(topo, table["0000:3d:00.0"].cpus, 8)
    assert format_cpulist(trimmed) == "16-19,80-83"


def test_single_socket_without_numa_directory_sets_no_mems(fake_host) -> None:
    gpus = [("0000:01:00.0", -1), ("0000:41:00.0", -1)]
    topo, table = _table(fake_host(nodes=(24,), l3_cores=8, gpus=gpus, nonuma=True))

    assert all(item.node is None and item.cores == 12 for item in table.values())
    cpus, mems = ht.slice_for(table, ["0000:01:00.0"])
    assert len(cpus) == 24
    assert mems == frozenset()
    assert ht.nodes_of(topo, cpus) == frozenset()


def test_gpu_without_numa_node_on_a_two_node_host_is_refused(fake_host) -> None:
    host = fake_host(gpus=[("0000:46:00.0", -1)])

    with pytest.raises(TopologyError, match="explicit CPULIST@NODES"):
        _table(host)


def test_gpu_on_a_cpuless_node_is_refused(fake_host) -> None:
    host = fake_host(gpus=[("0000:46:00.0", 2)], cpuless_nodes=(2,))

    with pytest.raises(TopologyError, match="no online CPUs"):
        _table(host)


def test_more_gpus_than_free_cores_is_refused(fake_host) -> None:
    gpus = [("0000:01:00.0", 0), ("0000:02:00.0", 0), ("0000:03:00.0", 0)]
    host = fake_host(nodes=(2, 2), smt=1, l3_cores=0, gpus=gpus)

    with pytest.raises(TopologyError, match="2 free cores for 3 GPUs"):
        _table(host)


def test_offline_cpus_and_cgroup_effective_set_are_excluded(fake_host) -> None:
    topo, table = _table(fake_host(offline=(32,), cgroup_effective="1-63"))

    assert 0 not in topo.online and 32 not in topo.online
    # Core 0 (0,32) is gone entirely; node 0 keeps whole-L3 mode at 3 cores.
    assert format_cpulist(table["0000:46:00.0"].cpus) == "1-3,33-35"
    assert format_cpulist(table["0000:49:00.0"].cpus) == "4-6,36-38"


def test_no_gpus_gives_an_empty_table(fake_host) -> None:
    _, table = _table(fake_host(gpus=()))

    assert table == {}


def test_slice_for_unions_pairs_and_nodes(fake_host) -> None:
    _, table = _table(fake_host())

    cpus, mems = ht.slice_for(table, ["0000:46:00.0", "0000:49:00.0"])
    assert (format_cpulist(cpus), mems) == ("0-7,32-39", frozenset({0}))
    cpus, mems = ht.slice_for(table, ["0000:4f:00.0", "0000:c5:00.0"])
    assert mems == frozenset({0, 1})
    with pytest.raises(TopologyError, match="not an NVIDIA GPU"):
        ht.slice_for(table, ["0001:ff:00.0"])


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (8, "8-11,40-43"),
        (6, "8-10,40-42"),
        (5, "8-10,40-41"),
        (1, "8"),
        (99, "8-11,40-43"),
    ],
)
def test_trim_takes_whole_cores_first(fake_host, budget: int, expected: str) -> None:
    topo, table = _table(fake_host())

    assert format_cpulist(ht.trim(topo, table["0000:4e:00.0"].cpus, budget)) == expected


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (16, "0-7,32-39"),
        (8, "0-1,4-5,32-33,36-37"),
        (6, "0-1,4-5,32,36"),
        (5, "0-1,4,32,36"),
    ],
)
def test_trim_slices_splits_the_budget_per_gpu(
    fake_host, budget: int, expected: str
) -> None:
    topo, table = _table(fake_host())

    trimmed = ht.trim_slices(topo, table, ["0000:46:00.0", "0000:49:00.0"], budget)
    assert len(trimmed) == min(budget, 16)
    assert format_cpulist(trimmed) == expected


def test_policy_id_never_contains_cpu_ids() -> None:
    assert ht.policy_id("auto") == "auto:gpu-local-l3-v1:mems=bind"
    assert (
        ht.policy_id("auto", mems_mode="none", reserve_count=2, undersize="allow")
        == "auto:gpu-local-l3-v1:mems=none:reserve=2:undersize=allow"
    )
    assert (
        ht.policy_id("explicit", cpu_count=8, node_count=1)
        == "explicit:cpus=8:nodes=1:mems=bind"
    )
    assert (
        ht.policy_id("explicit", cpu_count=8, mems_mode="none")
        == "explicit:cpus=8:nodes=local:mems=none"
    )
    with pytest.raises(ValueError):
        ht.policy_id("off")


def test_validate_explicit_checks_online_nodes_and_locality(fake_host) -> None:
    topo = ht.read_topology(fake_host().sysfs)
    cpus = parse_cpulist("8-11,40-43")

    assert ht.validate_explicit(topo, cpus, None, "bind") == frozenset({0})
    assert ht.validate_explicit(topo, cpus, frozenset({0}), "bind") == frozenset({0})
    assert ht.validate_explicit(topo, cpus, None, "none") is None
    with pytest.raises(TopologyError, match="remote"):
        ht.validate_explicit(topo, parse_cpulist("8,16"), frozenset({0}), "bind")
    with pytest.raises(TopologyError, match="not online"):
        ht.validate_explicit(topo, parse_cpulist("8,64"), None, "bind")
    with pytest.raises(TopologyError, match="do not exist"):
        ht.validate_explicit(topo, cpus, frozenset({0, 3}), "bind")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ()),
        ("void", ()),
        ("NONE", ()),
        ("all", ("all",)),
        ("3", ("3",)),
        ("03,3", ("3",)),
        (
            "GPU-00000001-1111-2222-3333-000000000001, 2",
            ("GPU-00000001-1111-2222-3333-000000000001", "2"),
        ),
    ],
)
def test_parse_visible_devices(value: str, expected: tuple[str, ...]) -> None:
    assert ht.parse_visible_devices(value) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("MIG-00000001-1111-2222-3333-000000000001", "MIG"),
        ("gpu0", "unrecognized"),
        ("0,all", "unrecognized"),
    ],
)
def test_parse_visible_devices_rejects(value: str, message: str) -> None:
    with pytest.raises(TopologyError, match=message):
        ht.parse_visible_devices(value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(host, capsys, *argv: str) -> tuple[int, str, str]:
    code = ht.main(["--sysfs", str(host.sysfs), "--procfs", str(host.procfs), *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_table_prints_one_row_per_gpu(fake_host, capsys, monkeypatch) -> None:
    monkeypatch.delenv("SFORGE_CPUSET_RESERVE", raising=False)
    code, out, _ = _cli(fake_host(), capsys, "table")

    assert code == 0
    assert out.splitlines() == [
        "GPU0 0000:46:00.0 0-3,32-35 node0",
        "GPU1 0000:49:00.0 4-7,36-39 node0",
        "GPU2 0000:4e:00.0 8-11,40-43 node0",
        "GPU3 0000:4f:00.0 12-15,44-47 node0",
        "GPU4 0000:c5:00.0 16-19,48-51 node1",
        "GPU5 0000:c6:00.0 20-23,52-55 node1",
        "GPU6 0000:c9:00.0 24-27,56-59 node1",
        "GPU7 0000:d0:00.0 28-31,60-63 node1",
    ]
    assert "GPU-" not in out


def test_cli_table_json_and_reserve(fake_host, capsys) -> None:
    code, out, _ = _cli(fake_host(), capsys, "table", "--json", "--reserve", "0,32")

    payload = json.loads(out)
    assert code == 0
    assert payload["algorithm"] == "gpu-local-l3-v1"
    assert payload["reserve"] == "0,32"
    first = payload["gpus"][0]
    assert (first["gpu"], first["pci"], first["cpus"], first["cpu_count"]) == (
        "0", "0000:46:00.0", "1-3,33-35", 6,
    )
    assert first["node_spare"] == "7,11,15,39,43,47"


def test_cli_slice_formats(fake_host, capsys) -> None:
    host = fake_host()

    code, out, _ = _cli(host, capsys, "slice", "--gpu", "0000:4e:00.0", "--budget", "8")
    assert (code, out.strip()) == (0, "--cpuset-cpus 8-11,40-43 --cpuset-mems 0")

    code, out, _ = _cli(
        host, capsys, "slice", "--gpu", "0000:c5:00.0", "--gpu", "0000:c6:00.0",
        "--format", "cpulist",
    )
    assert (code, out.strip()) == (0, "16-23,48-55")

    code, out, _ = _cli(
        host, capsys, "slice", "--gpu", "0000:4e:00.0", "--mems", "none",
    )
    assert out.strip() == "--cpuset-cpus 8-11,40-43"

    code, out, _ = _cli(
        host, capsys, "slice", "--gpu", "0000:46:00.0", "--gpu", "0000:49:00.0",
        "--budget", "8", "--format", "json",
    )
    payload = json.loads(out)
    assert payload["slice_cpus"] == "0-7,32-39"
    assert payload["cpu_count"] == 8
    assert payload["trimmed_to_quota"] is True
    assert payload["cpuset_mems"] == "0"


def test_cli_slice_resolves_a_uuid_through_procfs(fake_host, capsys) -> None:
    host = fake_host()
    uuid = ht.read_procfs_gpus(host.procfs)["0000:c9:00.0"].uuid
    assert uuid is not None

    code, out, _ = _cli(host, capsys, "slice", "--gpu", uuid, "--budget", "8")
    assert (code, out.strip()) == (0, "--cpuset-cpus 24-27,56-59 --cpuset-mems 1")


def test_cli_slice_refuses_an_undersized_slice(fake_host, capsys) -> None:
    host = fake_host()

    code, _, err = _cli(host, capsys, "slice", "--gpu", "0000:4e:00.0", "--budget", "16")
    assert code == 1
    assert "SFORGE_CPUSET_UNDERSIZE=allow" in err

    code, out, _ = _cli(
        host, capsys, "slice", "--gpu", "0000:4e:00.0", "--budget", "16",
        "--undersize", "allow",
    )
    assert (code, out.strip()) == (0, "--cpuset-cpus 8-11,40-43 --cpuset-mems 0")


def test_cli_policy_id_reads_the_environment(fake_host, capsys, monkeypatch) -> None:
    host = fake_host()
    for name in (
        "SFORGE_CPUSET", "SFORGE_WORK_CPUSET", "SFORGE_JUDGE_CPUSET",
        "SFORGE_CPUSET_MEMS", "SFORGE_CPUSET_RESERVE", "SFORGE_CPUSET_UNDERSIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _cli(host, capsys, "policy-id", "--role", "judge") == (0, "", "")

    monkeypatch.setenv("SFORGE_CPUSET", "auto")
    monkeypatch.setenv("SFORGE_WORK_CPUSET", "8-11,40-43@0")
    assert _cli(host, capsys, "policy-id", "--role", "judge")[1] == (
        "auto:gpu-local-l3-v1:mems=bind\n"
    )
    assert _cli(host, capsys, "policy-id", "--role", "work")[1] == (
        "explicit:cpus=8:nodes=1:mems=bind\n"
    )
    assert _cli(host, capsys, "policy-id")[1] == "auto:gpu-local-l3-v1:mems=bind\n"

    monkeypatch.setenv("SFORGE_JUDGE_CPUSET", "off")
    assert _cli(host, capsys, "policy-id", "--role", "judge")[1] == ""

    monkeypatch.setenv("SFORGE_CPUSET", "garbage")
    code, _, err = _cli(host, capsys, "policy-id")
    assert code == 1 and "SFORGE_CPUSET must be off, auto or CPULIST" in err


def test_cli_check_validates_against_the_topology(fake_host, capsys, monkeypatch) -> None:
    host = fake_host()
    monkeypatch.delenv("SFORGE_WORK_CPUSET", raising=False)
    monkeypatch.delenv("SFORGE_JUDGE_CPUSET", raising=False)
    monkeypatch.delenv("SFORGE_CPUSET_MEMS", raising=False)
    monkeypatch.delenv("SFORGE_CPUSET_RESERVE", raising=False)
    monkeypatch.delenv("SFORGE_CPUSET_UNDERSIZE", raising=False)

    monkeypatch.delenv("SFORGE_CPUSET", raising=False)
    assert _cli(host, capsys, "check")[:2] == (0, "off\n")

    monkeypatch.setenv("SFORGE_CPUSET", "auto")
    code, out, _ = _cli(host, capsys, "check", "--gpu", "0000:4e:00.0")
    assert code == 0
    assert out.startswith("ok auto:gpu-local-l3-v1:mems=bind sha256:")
    # The preflight also shows the slice the GPU resolves to.
    assert out.endswith(" cpus=8-11,40-43 mems=0\n")

    monkeypatch.setenv("SFORGE_CPUSET_MEMS", "none")
    code, out, _ = _cli(host, capsys, "check", "--gpu", "0000:4e:00.0", "--gpu", "0000:4f:00.0")
    assert code == 0 and out.endswith(" cpus=8-15,40-47\n")
    monkeypatch.delenv("SFORGE_CPUSET_MEMS")

    monkeypatch.setenv("SFORGE_CPUSET", "8,16@0")
    code, _, err = _cli(host, capsys, "check")
    assert code == 1 and "remote" in err
