"""Host-side accelerator-idle observation for Work/Judge handoff.

Docker freezer state is deliberately not treated as evidence that CUDA has
released an accelerator.  The opt-in serial handoff path must obtain an
attestation from this interface after pausing Work and after destroying Judge.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import time
from dataclasses import dataclass
from typing import Protocol

from sforge.harness.backend import AcceleratorAllocation


class AcceleratorAttestationError(RuntimeError):
    """The observer could not prove that an allocation is idle."""


class AcceleratorBusyError(AcceleratorAttestationError):
    """Observed processes prevent idle attestation.

    ``recoverable`` may be true only when a stronger observer has attributed
    every reported process to the paused Work container.  The built-in NVML
    process-list observer cannot make that cgroup attribution and therefore
    always reports unknown ownership (false).
    """

    def __init__(self, message: str, *, recoverable: bool = False) -> None:
        super().__init__(message)
        self.recoverable = recoverable


@dataclass(frozen=True)
class AcceleratorIdleAttestation:
    """JSON-safe evidence from consecutive host NVML observations."""

    observer: str
    allocation_vendor: str
    device_ids: tuple[str, ...]
    phase: str
    sample_count: int
    observed_at: float
    observed_monotonic: float

    def to_dict(self) -> dict[str, object]:
        return {
            "observer": self.observer,
            "allocation_vendor": self.allocation_vendor,
            "device_ids": list(self.device_ids),
            "phase": self.phase,
            "sample_count": self.sample_count,
            "observed_at": self.observed_at,
            "observed_monotonic": self.observed_monotonic,
            "compute_process_count": 0,
            "graphics_process_count": 0,
            "accelerator_idle": True,
        }


class AcceleratorObserver(Protocol):
    """Trusted host observer required by the serial GPU coordinator."""

    def attest_idle(
        self,
        allocation: AcceleratorAllocation,
        *,
        phase: str,
    ) -> AcceleratorIdleAttestation:
        """Return evidence only after proving every exact device idle."""


class _NvmlProcessInfo(ctypes.Structure):
    # ABI used by nvmlDeviceGet*RunningProcesses_v2.  The first two fields are
    # stable; the GI/CI fields make the query safe for exact MIG handles.
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class NvmlAcceleratorObserver:
    """Real host NVML process/context observer loaded through ``ctypes``.

    The library is loaded lazily so legacy CPU-only SForge usage has no NVIDIA
    dependency.  An unavailable API, unknown return code, non-empty process
    list, or changing device identity fails closed.  Two observations are the
    minimum: one snapshot alone cannot establish a handoff boundary.
    """

    NVML_SUCCESS = 0
    NVML_ERROR_INSUFFICIENT_SIZE = 7

    def __init__(
        self,
        *,
        sample_count: int = 2,
        sample_interval_seconds: float = 0.2,
        idle_grace_seconds: float = 5.0,
        library: object | None = None,
    ) -> None:
        if isinstance(sample_count, bool) or sample_count < 2:
            raise ValueError("NVML idle attestation requires at least two samples")
        if sample_interval_seconds < 0:
            raise ValueError("NVML sample interval must be non-negative")
        if idle_grace_seconds < 0:
            raise ValueError("NVML idle grace must be non-negative")
        self._sample_count = sample_count
        self._sample_interval_seconds = sample_interval_seconds
        self._idle_grace_seconds = idle_grace_seconds
        self._library = library

    def attest_idle(
        self,
        allocation: AcceleratorAllocation,
        *,
        phase: str,
    ) -> AcceleratorIdleAttestation:
        if allocation.vendor != "nvidia":
            raise AcceleratorAttestationError(
                "NVML observer supports only NVIDIA allocations"
            )
        if not isinstance(phase, str) or not phase.strip():
            raise AcceleratorAttestationError(
                "attestation phase must be a non-empty string"
            )

        library = self._library or self._load_library()
        init = self._function(library, "nvmlInit_v2", [], ctypes.c_int)
        shutdown = self._function(library, "nvmlShutdown", [], ctypes.c_int)
        self._check(init(), "nvmlInit_v2")
        try:
            deadline = time.monotonic() + self._idle_grace_seconds
            consecutive_empty = 0
            while consecutive_empty < self._sample_count:
                busy_observation: tuple[str, tuple[int, ...], tuple[int, ...]] | None = None
                for device_id in allocation.device_ids:
                    compute, graphics = self._sample_device(library, device_id)
                    if compute or graphics:
                        busy_observation = (device_id, compute, graphics)
                        break
                if busy_observation is None:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                    if time.monotonic() >= deadline:
                        device_id, compute, graphics = busy_observation
                        raise AcceleratorBusyError(
                            "accelerator did not become idle within the "
                            "bounded grace after Docker lifecycle operation "
                            f"(device={device_id}, phase={phase}, "
                            f"compute_pids={compute}, graphics_pids={graphics}); "
                            "Docker pause does not release CUDA contexts",
                            recoverable=False,
                        )
                if consecutive_empty < self._sample_count:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 and busy_observation is not None:
                        continue
                    time.sleep(
                        min(self._sample_interval_seconds, max(0.0, remaining))
                    )
        finally:
            self._check(shutdown(), "nvmlShutdown")

        return AcceleratorIdleAttestation(
            observer="nvml-process-context-v1",
            allocation_vendor=allocation.vendor,
            device_ids=allocation.device_ids,
            phase=phase,
            sample_count=self._sample_count,
            observed_at=time.time(),
            observed_monotonic=time.monotonic(),
        )

    @staticmethod
    def _load_library() -> object:
        library_name = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        try:
            return ctypes.CDLL(library_name)
        except OSError as exc:
            raise AcceleratorAttestationError(
                "serial GPU handoff requires a working host NVML library"
            ) from exc

    @staticmethod
    def _function(
        library: object,
        name: str,
        argtypes: list[object],
        restype: object,
    ):
        try:
            function = getattr(library, name)
        except AttributeError as exc:
            raise AcceleratorAttestationError(
                f"host NVML is missing required API {name}"
            ) from exc
        function.argtypes = argtypes
        function.restype = restype
        return function

    def _sample_device(
        self,
        library: object,
        device_id: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        handle = ctypes.c_void_p()
        get_handle = self._function(
            library,
            "nvmlDeviceGetHandleByUUID",
            [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            ctypes.c_int,
        )
        self._check(
            get_handle(device_id.encode("ascii"), ctypes.byref(handle)),
            f"nvmlDeviceGetHandleByUUID({device_id})",
        )
        if not handle.value:
            raise AcceleratorAttestationError(
                f"NVML returned an empty handle for {device_id}"
            )
        compute = self._running_processes(
            library,
            "nvmlDeviceGetComputeRunningProcesses_v2",
            handle,
            device_id,
        )
        graphics = self._running_processes(
            library,
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            handle,
            device_id,
        )
        return compute, graphics

    def _running_processes(
        self,
        library: object,
        function_name: str,
        handle: ctypes.c_void_p,
        device_id: str,
    ) -> tuple[int, ...]:
        process_pointer = ctypes.POINTER(_NvmlProcessInfo)
        query = self._function(
            library,
            function_name,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), process_pointer],
            ctypes.c_int,
        )
        count = ctypes.c_uint(0)
        result = query(handle, ctypes.byref(count), None)
        if result == self.NVML_SUCCESS:
            return ()
        if result != self.NVML_ERROR_INSUFFICIENT_SIZE or count.value == 0:
            self._check(result, f"{function_name}({device_id})")
            raise AcceleratorAttestationError(
                f"{function_name} returned ambiguous process count for {device_id}"
            )
        values = (_NvmlProcessInfo * count.value)()
        result = query(handle, ctypes.byref(count), values)
        self._check(result, f"{function_name}({device_id})")
        return tuple(sorted({int(values[index].pid) for index in range(count.value)}))

    @classmethod
    def _check(cls, result: int, operation: str) -> None:
        if int(result) != cls.NVML_SUCCESS:
            raise AcceleratorAttestationError(
                f"{operation} failed with NVML status {int(result)}"
            )


def create_accelerator_observer(
    mode: str | None,
    *,
    idle_grace_seconds: float = 5.0,
) -> AcceleratorObserver | None:
    """Create only explicitly selected trusted observers."""

    if mode is None:
        return None
    if mode != "nvml":
        raise ValueError("accelerator observer must be exactly 'nvml'")
    return NvmlAcceleratorObserver(idle_grace_seconds=idle_grace_seconds)
