"""Trusted controller-side concurrent NVML peak sampling."""

from __future__ import annotations

import ctypes
import threading
from typing import Callable


class NvmlSamplerError(RuntimeError):
    pass


def fixed_shape_memory_stable(
    first: dict[str, int | float],
    repeat: dict[str, int | float],
    *,
    material_growth_bytes: int = 16 * 1024 * 1024,
) -> tuple[bool, dict[str, int]]:
    """Compare only identical, warmed request shapes for leak detection."""
    fields = (
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "nvml_peak_device_used_bytes",
    )
    deltas = {field: int(repeat[field]) - int(first[field]) for field in fields}
    return (
        all(delta <= material_growth_bytes for delta in deltas.values()),
        deltas,
    )


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def bind_single_visible_nvml_used_probe() -> Callable[[], int]:
    """Bind the sole DeviceRequest-isolated GPU before candidate import."""
    try:
        library = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError as exc:
        raise NvmlSamplerError("NVML library is unavailable") from exc
    library.nvmlInit_v2.restype = ctypes.c_int
    library.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    library.nvmlDeviceGetCount_v2.restype = ctypes.c_int
    library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    library.nvmlDeviceGetMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NvmlMemory),
    ]
    library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
    if library.nvmlInit_v2() != 0:
        raise NvmlSamplerError("NVML initialization failed")
    count = ctypes.c_uint()
    if library.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value != 1:
        raise NvmlSamplerError("Judge must expose exactly one NVML device")
    handle = ctypes.c_void_p()
    if library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
        raise NvmlSamplerError("NVML could not resolve the sole Judge device")

    def used_bytes() -> int:
        memory = _NvmlMemory()
        if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
            raise NvmlSamplerError("NVML memory query failed")
        value = int(memory.used)
        if value <= 0:
            raise NvmlSamplerError("NVML returned nonpositive used memory")
        return value

    return used_bytes


def nvml_driver_version() -> str:
    """Return the NVIDIA driver version, or "unavailable" when NVML is absent.

    The driver participates in the measurement-profile key, so both the
    reference producer and the Judge must derive it the same way, including
    the fallback: a profile is only ever compared against another profile.
    """
    try:
        library = ctypes.CDLL("libnvidia-ml.so.1")
        library.nvmlInit_v2.restype = ctypes.c_int
        library.nvmlSystemGetDriverVersion.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        library.nvmlSystemGetDriverVersion.restype = ctypes.c_int
        if library.nvmlInit_v2() != 0:
            return "unavailable"
        buffer = ctypes.create_string_buffer(96)
        if library.nvmlSystemGetDriverVersion(buffer, 96) != 0:
            return "unavailable"
    except OSError:
        return "unavailable"
    version = buffer.value.decode("ascii", "replace").strip()
    return version or "unavailable"


class PeakMemorySampler:
    """Sample concurrently across a complete candidate run window."""

    def __init__(self, probe: Callable[[], int], interval_seconds: float = 0.005):
        if interval_seconds <= 0:
            raise ValueError("NVML sampling interval must be positive")
        self._probe = probe
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._error: BaseException | None = None

    def _sample(self) -> None:
        try:
            value = int(self._probe())
            if value <= 0:
                raise NvmlSamplerError("NVML sampler observed nonpositive memory")
            self._peak = max(self._peak, value)
        except BaseException as exc:  # retained and raised by the trusted owner
            self._error = exc
            self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise NvmlSamplerError("NVML sampler cannot be reused")
        self._sample()
        if self._error is not None:
            raise NvmlSamplerError("initial NVML sample failed") from self._error
        self._thread = threading.Thread(
            target=self._loop,
            name="trusted-nvml-peak-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> int:
        if self._thread is None:
            raise NvmlSamplerError("NVML sampler was not started")
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval * 4))
        if self._thread.is_alive():
            raise NvmlSamplerError("NVML sampler did not stop")
        self._sample()
        if self._error is not None:
            raise NvmlSamplerError("concurrent NVML sampling failed") from self._error
        if self._peak <= 0:
            raise NvmlSamplerError("NVML sampler produced no peak")
        return self._peak
