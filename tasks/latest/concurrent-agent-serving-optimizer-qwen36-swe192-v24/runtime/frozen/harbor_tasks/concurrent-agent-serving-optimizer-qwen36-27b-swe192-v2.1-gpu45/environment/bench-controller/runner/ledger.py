from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


MEASUREMENT_WINDOW_REASON = "trusted_measurement_window_ended_v1"


class LedgerClosed(RuntimeError):
    """A trusted measurement boundary closed request admission."""

    def __init__(self, reason: str, cutoff_unix: float) -> None:
        self.reason = reason
        self.cutoff_unix = cutoff_unix
        super().__init__(f"{reason}:cutoff_unix={cutoff_unix:.6f}")


class Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._active: dict[str, dict[str, Any]] = {}
        self._closed_reason: str | None = None
        self._closed_unix: float | None = None
        self._cutoff_request_ids: tuple[str, ...] = ()

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)

    def begin(self, record: dict[str, Any]) -> None:
        """Register a client request before it can enter an untrusted gateway."""

        request_id = str(record.get("request_id") or "")
        if not request_id:
            raise ValueError("active ledger record has no request_id")
        active = {
            **record,
            "request_id": request_id,
            "started_monotonic": time.monotonic(),
            "cancel_reason": None,
            "cancel_callback": None,
        }
        with self._lock:
            if self._closed_reason is not None and self._closed_unix is not None:
                raise LedgerClosed(self._closed_reason, self._closed_unix)
            if request_id in self._active:
                raise ValueError(f"duplicate active request_id: {request_id}")
            self._active[request_id] = active

    def register_canceller(
        self,
        request_id: str,
        callback: Callable[[str], None],
    ) -> bool:
        """Attach a connection abort callback, closing a cancellation race.

        Returns ``False`` when a watchdog cancelled the request before the
        connection existed.  In that case the callback is invoked immediately.
        """

        reason: str | None
        with self._lock:
            active = self._active.get(request_id)
            if active is None:
                raise KeyError(f"request is not active: {request_id}")
            active["cancel_callback"] = callback
            raw_reason = active.get("cancel_reason")
            reason = str(raw_reason) if raw_reason is not None else None
        if reason is not None:
            callback(reason)
            return False
        return True

    def cancellation_reason(self, request_id: str) -> str | None:
        with self._lock:
            active = self._active.get(request_id)
            if active is None or active.get("cancel_reason") is None:
                return None
            return str(active["cancel_reason"])

    def complete(self, record: dict[str, Any]) -> None:
        request_id = str(record.get("request_id") or "")
        if not request_id:
            raise ValueError("completed ledger record has no request_id")
        with self._lock:
            if self._active.pop(request_id, None) is None:
                raise KeyError(f"request completed without an active record: {request_id}")
            self._records.append(record)

    def activity_snapshot(self, now: float | None = None) -> dict[str, Any]:
        observed = time.monotonic() if now is None else now
        with self._lock:
            active = list(self._active.values())
            completed = len(self._records)
        ages = [max(0.0, observed - float(item["started_monotonic"])) for item in active]
        return {
            "active_requests": len(active),
            "completed_requests": completed,
            "cancelled_active_requests": sum(
                item.get("cancel_reason") is not None for item in active
            ),
            "oldest_active_age_sec": max(ages) if ages else None,
        }

    def cancel_active(self, reason: str) -> dict[str, Any]:
        """Mark and actively disconnect every in-flight client request."""

        if not reason:
            raise ValueError("active-request cancellation reason is empty")
        callbacks: list[Callable[[str], None]] = []
        with self._lock:
            active_count = len(self._active)
            newly_cancelled = 0
            for active in self._active.values():
                if active.get("cancel_reason") is None:
                    active["cancel_reason"] = reason
                    newly_cancelled += 1
                callback = active.get("cancel_callback")
                if callable(callback):
                    callbacks.append(callback)
        errors = []
        for callback in callbacks:
            try:
                callback(reason)
            except Exception as exc:  # pragma: no cover - diagnostic safety net
                errors.append(f"{type(exc).__name__}: {exc}")
        return {
            "active_requests": active_count,
            "newly_cancelled_requests": newly_cancelled,
            "cancel_callbacks": len(callbacks),
            "cancel_errors": errors[:5],
        }

    def close_for_measurement(
        self,
        reason: str,
        cutoff_unix: float,
        *,
        now_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """Atomically close admission and freeze trusted cutoff request IDs.

        Either ``begin`` wins the ledger lock and the request is included in
        the frozen cutoff set, or this method wins and the later ``begin`` is
        rejected with :class:`LedgerClosed`.  Callbacks run outside the lock so
        a slow socket shutdown cannot block record completion.
        """

        if not reason:
            raise ValueError("measurement close reason is empty")
        if not isinstance(cutoff_unix, (int, float)) or isinstance(cutoff_unix, bool):
            raise ValueError("measurement cutoff must be numeric")
        observed_monotonic = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        callbacks: list[Callable[[str], None]] = []
        with self._lock:
            if self._closed_reason is not None:
                raise RuntimeError("measurement ledger is already closed")
            self._closed_reason = reason
            self._closed_unix = float(cutoff_unix)
            cutoff_ids = tuple(sorted(self._active))
            self._cutoff_request_ids = cutoff_ids
            ages = []
            for active in self._active.values():
                active["cancel_reason"] = reason
                ages.append(
                    max(
                        0.0,
                        observed_monotonic
                        - float(active.get("started_monotonic", observed_monotonic)),
                    )
                )
                callback = active.get("cancel_callback")
                if callable(callback):
                    callbacks.append(callback)
        errors = []
        for callback in callbacks:
            try:
                callback(reason)
            except Exception as exc:  # pragma: no cover - diagnostic safety net
                errors.append(f"{type(exc).__name__}: {exc}")
        return {
            "reason": reason,
            "cutoff_unix": float(cutoff_unix),
            "request_ids": list(cutoff_ids),
            "active_requests": len(cutoff_ids),
            "cancel_callbacks": len(callbacks),
            "cancel_errors": errors[:5],
            "oldest_active_age_sec": max(ages) if ages else None,
        }

    def measurement_state(self) -> dict[str, Any]:
        """Return controller-private cutoff state for tests and aggregation."""

        with self._lock:
            return {
                "closed": self._closed_reason is not None,
                "reason": self._closed_reason,
                "cutoff_unix": self._closed_unix,
                "request_ids": list(self._cutoff_request_ids),
            }

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.copy() for record in self._records]
