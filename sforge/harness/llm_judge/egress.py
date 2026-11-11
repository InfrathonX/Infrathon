"""Network for integrity-review containers.

``isolated`` (default): review containers join an internal Docker network
whose only other member is a small egress sidecar (``egress_proxy.py``).  The
sidecar tunnels HTTPS only to allowlisted model-API hosts and relays fixed
plain-HTTP model endpoints; nothing else is reachable.  The network's bridge
gets no host IPv4 address (``inhibit_ipv4``): an ordinary internal network
still lets containers reach host services through its gateway address.  This is the same
rootless pattern the task launchers use for Work containers, because this
host grants no passwordless iptables.

``bridge``: plain Docker bridge with open egress, for local debugging only.

The network and sidecar are created on first use and shared by every review
of one Judge server; ``close`` removes them.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger("sforge.llm_judge")

PROXY_SCRIPT = Path(__file__).with_name("egress_proxy.py")
PROXY_TARGET = "/opt/sforge-review/egress_proxy.py"
CONNECT_PORT = 3128
FIRST_FORWARD_PORT = 3131
CHATGPT_HOSTS = ("chatgpt.com", "auth.openai.com", "api.openai.com")
_DEFAULT_PORTS = {"http": 80, "https": 443}


class EgressError(RuntimeError):
    """The review network could not be prepared."""


@dataclass(frozen=True)
class EgressPlan:
    """What a review container must reach: CONNECT hosts and raw relays."""

    allow_hosts: tuple[str, ...] = ()
    forwards: tuple[tuple[str, int], ...] = ()

    @classmethod
    def for_endpoints(
        cls,
        urls: list[str],
        *,
        chatgpt: bool,
        extra_hosts: tuple[str, ...] = (),
    ) -> "EgressPlan":
        hosts: list[str] = list(CHATGPT_HOSTS) if chatgpt else []
        forwards: list[tuple[str, int]] = []
        for url in urls:
            parsed = urlparse(url)
            if not parsed.hostname:
                raise EgressError(f"model endpoint has no host: {url!r}")
            port = parsed.port or _DEFAULT_PORTS.get(parsed.scheme, 0)
            if parsed.scheme == "https" and port == 443:
                hosts.append(parsed.hostname.lower())
            elif parsed.scheme == "http" and port:
                forwards.append((parsed.hostname, port))
            else:
                raise EgressError(
                    "isolated review egress supports https://host (port 443) "
                    f"or plain http://host:port endpoints, not {url!r}"
                )
        hosts.extend(host.lower() for host in extra_hosts)
        return cls(
            allow_hosts=tuple(dict.fromkeys(hosts)),
            forwards=tuple(dict.fromkeys(forwards)),
        )


@dataclass(frozen=True)
class EgressEndpoint:
    """How a review container reaches its model."""

    network: str
    proxy_url: str | None = None
    forward_urls: dict[tuple[str, int], str] = field(default_factory=dict)

    def environment(self) -> dict[str, str]:
        if self.proxy_url is None:
            return {}
        proxy_host = urlparse(self.proxy_url).hostname or ""
        no_proxy = ",".join(item for item in (proxy_host, "localhost", "127.0.0.1") if item)
        return {
            "HTTP_PROXY": self.proxy_url,
            "HTTPS_PROXY": self.proxy_url,
            "http_proxy": self.proxy_url,
            "https_proxy": self.proxy_url,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }

    def rewrite(self, url: str) -> str:
        """Route a plain-HTTP model URL through its relay (others unchanged)."""

        parsed = urlparse(url)
        if parsed.scheme != "http" or not parsed.hostname:
            return url
        port = parsed.port or 80
        relay = self.forward_urls.get((parsed.hostname, port))
        if relay is None:
            return url
        relay_parsed = urlparse(relay)
        return urlunparse(parsed._replace(netloc=relay_parsed.netloc))


class ReviewEgress:
    def __init__(
        self,
        client,
        *,
        mode: str,
        plan: EgressPlan,
        sidecar_image_resolver,
        label: str | None = None,
    ) -> None:
        self.client = client
        self.mode = mode
        self.plan = plan
        self._image_resolver = sidecar_image_resolver
        self._label = label or uuid.uuid4().hex[:10]
        self._lock = threading.Lock()
        self._endpoint: EgressEndpoint | None = None
        self._network = None
        self._sidecar = None

    @property
    def names(self) -> tuple[str, str]:
        return (
            f"sforge-review-{self._label}-internal",
            f"sforge-review-{self._label}-egress",
        )

    def endpoint(self, image_hint: str) -> EgressEndpoint:
        if self.mode == "bridge":
            return EgressEndpoint(network="bridge")
        with self._lock:
            if self._endpoint is not None and self._sidecar_alive():
                return self._endpoint
            image = self._image_resolver(image_hint)
            if self._network is None:
                self._create_network()
            else:
                # Reviews may still be attached to the network: replace only
                # the dead sidecar, keep the network (and its handle).
                self._remove_sidecar()
            self._endpoint = self._start_sidecar(image)
            return self._endpoint

    def _sidecar_alive(self) -> bool:
        try:
            self._sidecar.reload()
        except Exception:
            return False
        return self._sidecar.status == "running"

    def _create_network(self) -> None:
        network_name, _ = self.names
        labels = {"sforge.role": "llm-judge-egress", "sforge.review_egress": self._label}
        self._network = self.client.networks.create(
            network_name,
            driver="bridge",
            internal=True,
            labels=labels,
            options={"com.docker.network.bridge.inhibit_ipv4": "true"},
        )

    def _start_sidecar(self, image: str) -> EgressEndpoint:
        network_name, sidecar_name = self.names
        labels = {"sforge.role": "llm-judge-egress", "sforge.review_egress": self._label}
        self._network.reload()
        subnets = [
            entry.get("Subnet")
            for entry in (self._network.attrs.get("IPAM") or {}).get("Config") or []
            if entry.get("Subnet")
        ]
        if not subnets:
            raise EgressError("review network has no subnet to restrict the sidecar to")
        command = ["--proxy-port", str(CONNECT_PORT), "--allow", ",".join(self.plan.allow_hosts)]
        for subnet in subnets:
            command += ["--allow-peer", subnet]
        ports: dict[tuple[str, int], int] = {}
        for index, (host, port) in enumerate(self.plan.forwards):
            listen = FIRST_FORWARD_PORT + index
            ports[(host, port)] = listen
            command += ["--forward", f"{listen}:{host}:{port}"]
        try:
            self._sidecar = self.client.containers.run(
                image,
                command=command,
                entrypoint=["python3", PROXY_TARGET],
                name=f"{sidecar_name}-{uuid.uuid4().hex[:6]}",
                detach=True,
                network="bridge",
                environment={"NVIDIA_VISIBLE_DEVICES": "void"},
                volumes={str(PROXY_SCRIPT): {"bind": PROXY_TARGET, "mode": "ro"}},
                user="65534:65534",
                mem_limit="256m",
                nano_cpus=int(0.5e9),
                labels=labels,
            )
            self._network.connect(self._sidecar, aliases=["review-egress"])
            address = self._wait_ready()
        except Exception as exc:
            self._remove_sidecar()
            raise EgressError(f"review egress sidecar failed: {exc}") from exc
        logger.info(
            "Review egress ready on %s (%s), CONNECT allow=%s, relays=%s",
            network_name,
            address,
            ",".join(self.plan.allow_hosts) or "-",
            ",".join(f"{h}:{p}" for h, p in self.plan.forwards) or "-",
        )
        return EgressEndpoint(
            network=network_name,
            proxy_url=f"http://{address}:{CONNECT_PORT}",
            forward_urls={
                key: f"http://{address}:{listen}" for key, listen in ports.items()
            },
        )

    def _wait_ready(self) -> str:
        network_name, _ = self.names
        deadline = time.monotonic() + 30
        probe = (
            "import socket,sys\n"
            f"socket.create_connection(('127.0.0.1', {CONNECT_PORT}), 2).close()\n"
        )
        while time.monotonic() < deadline:
            self._sidecar.reload()
            if self._sidecar.status not in {"created", "running"}:
                raise EgressError(
                    "sidecar exited: " + self._sidecar.logs(tail=20).decode("utf-8", "replace")
                )
            networks = self._sidecar.attrs.get("NetworkSettings", {}).get("Networks", {})
            address = (networks.get(network_name) or {}).get("IPAddress")
            if address and self._sidecar.status == "running":
                result = self._sidecar.exec_run(["python3", "-c", probe])
                if result.exit_code == 0:
                    return address
            time.sleep(0.5)
        raise EgressError("sidecar did not become ready within 30s")

    def _remove_sidecar(self) -> None:
        if self._sidecar is not None:
            try:
                self._sidecar.remove(force=True)
            except Exception:
                pass
            self._sidecar = None
        self._endpoint = None

    def close(self) -> None:
        """Remove the sidecar and the network; retry while review containers
        are still detaching, and keep the handle if it never succeeds."""

        with self._lock:
            self._remove_sidecar()
            if self._network is None:
                return
            for attempt in range(10):
                try:
                    self._network.remove()
                    self._network = None
                    return
                except Exception:
                    time.sleep(min(2.0, 0.2 * (attempt + 1)))
            logger.warning(
                "Review network %s still has endpoints; remove it manually "
                "(label sforge.review_egress=%s)",
                self.names[0],
                self._label,
            )
