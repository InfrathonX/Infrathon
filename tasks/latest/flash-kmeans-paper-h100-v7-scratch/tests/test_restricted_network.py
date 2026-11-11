from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROXY_PATH = PROJECT_ROOT / "scripts" / "restricted_connect_proxy.py"


def _load_proxy():
    spec = importlib.util.spec_from_file_location(
        "flash_kmeans_restricted_proxy",
        PROXY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RestrictedNetworkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.proxy = _load_proxy()

    def test_connect_allowlist_is_model_auth_only(self) -> None:
        self.assertEqual(
            self.proxy.ALLOWED_CONNECT_HOSTS,
            frozenset(
                {
                    "api.openai.com",
                    "auth.openai.com",
                    "chatgpt.com",
                }
            ),
        )
        self.assertEqual(
            self.proxy.parse_connect_target("chatgpt.com:443"),
            ("chatgpt.com", 443),
        )

    def test_connect_target_rejects_web_literal_ip_and_other_ports(self) -> None:
        for target in (
            "github.com:443",
            "example.com:443",
            "1.1.1.1:443",
            "chatgpt.com:80",
            "chatgpt.com",
        ):
            with self.subTest(target=target):
                with self.assertRaises((ValueError, PermissionError)):
                    self.proxy.parse_connect_target(target)

    def test_public_egress_mode_widens_only_to_web_ports(self) -> None:
        self.assertEqual(
            self.proxy.parse_connect_target(
                "github.com:443", allow_public_egress=True
            ),
            ("github.com", 443),
        )
        with self.assertRaises(PermissionError):
            self.proxy.parse_connect_target(
                "github.com:22", allow_public_egress=True
            )
        self.assertFalse(self.proxy.is_public_address("172.17.0.1"))
        self.assertTrue(self.proxy.is_public_address("140.82.112.3"))

    def test_formal_launcher_uses_internal_network_entry_and_sidecar(self) -> None:
        launcher = (
            PROJECT_ROOT / "scripts" / "run_fixed_budget.sh"
        ).read_text(encoding="utf-8")
        entry = (
            PROJECT_ROOT / "scripts" / "sforge_isolated_entry.py"
        ).read_text(encoding="utf-8")
        self.assertIn("docker network create", launcher)
        self.assertIn("--internal", launcher)
        self.assertIn("restricted_connect_proxy.py", launcher)
        self.assertIn("SFORGE_ISOLATED_DOCKER_NETWORK", launcher)
        self.assertIn("direct_public_egress: false", launcher)
        self.assertIn('sidecar_public_egress="--allow-public-egress"', launcher)
        self.assertIn("Public-egress network: PASS", launcher)
        self.assertIn('export MLSYSFORGE_WORK_INTERNET="${task_internet}"', launcher)
        self.assertIn('WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"', entry)
        self.assertNotIn('kwargs["internet"] = True', entry)
        self.assertNotIn('kwargs["internet"] = False', entry)
        self.assertIn("CodexAgent.install_cmds", entry)
        self.assertIn('"network": network_name', entry)

