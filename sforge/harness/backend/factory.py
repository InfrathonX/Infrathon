"""Backend factory."""

from __future__ import annotations

from sforge.harness.backend.base import ContainerBackend


def create_backend(
    name: str = "docker",
    *,
    docker_client=None,
    k8s_namespace: str = "default",
    k8s_node_selector: dict[str, str] | None = None,
    k8s_image_registry: str = "",
    k8s_kubeconfig: str | None = None,
) -> ContainerBackend:
    if name == "docker":
        from sforge.harness.backend.docker_backend import DockerBackend

        return DockerBackend(client=docker_client)
    elif name == "k8s":
        from sforge.harness.backend.k8s_backend import K8sBackend

        return K8sBackend(
            namespace=k8s_namespace,
            node_selector=k8s_node_selector,
            image_registry=k8s_image_registry,
            kubeconfig=k8s_kubeconfig,
        )
    elif name == "pyromind":
        from sforge.harness.backend.pyromind_backend import PyroMindBackend

        return PyroMindBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}. Must be 'docker', 'k8s' or 'pyromind'.")
