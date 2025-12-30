"""interlatent.api.latent_db

The public façade for interacting with a LatentDB instance.
It abstracts away storage details, statistics computation, and LLM-driven
explanations so users can treat a trained neural network like a chatty,
self‑documenting black box.

Design goals
------------
• **Simple import surface** – `from interlatent.api import LatentDB`
• **Storage‑agnostic** – pluggable back‑end selected from URI scheme.
• **Async‑friendly** – heavy tasks (stats, descriptions) can run in a
  background executor but fall back to sync for notebooks.
• **Typed** – Pydantic schemas enforce contract integrity across
  modules.
"""
from __future__ import annotations

import json
import importlib
import inspect
import textwrap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from pydantic import ValidationError

# Local imports — these modules live elsewhere in the repo; circular
# dependency risks are mitigated by importing lazily where needed.
from ..schema import ActivationEvent, Explanation, StatBlock  # type: ignore
from ..storage.base import StorageBackend  # type: ignore
from ..utils.logging import get_logger  # type: ignore

_LOG = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helper: backend dispatcher -------------------------------------------------
# ---------------------------------------------------------------------------


_SCHEME_TO_BACKEND = {
    "sqlite": "..storage.sqlite:SQLiteBackend",
    "file": "..storage.sqlite:SQLiteBackend",  # alias
    "dynamodb": "..storage.dynamo:DynamoBackend",
    "hdf5": "..storage.hdf5:HDF5Backend",
    "h5": "..storage.hdf5:HDF5Backend",
    "hdf5row": "..storage.hdf5_row:HDF5RowBackend",
    "hdf5v2": "..storage.hdf5_row:HDF5RowBackend",
}


def _resolve_backend(uri: str) -> StorageBackend:  # pragma: no cover
    """Instantiate a :class:`StorageBackend` based on the URI scheme.

    Parameters
    ----------
    uri:
        e.g. ``"sqlite:///my_runs/latents.db"`` or
        ``"dynamodb://latentdb-prod"``.
    """
    scheme = uri.split(":", 1)[0]
    dotted = _SCHEME_TO_BACKEND.get(scheme)
    if dotted is None:
        raise ValueError(f"Unknown storage scheme '{scheme}'. Registered: {list(_SCHEME_TO_BACKEND)}")

    mod_path, _, cls_name = dotted.partition(":")
    module = importlib.import_module(mod_path, package=__package__)
    backend_cls: type[StorageBackend] = getattr(module, cls_name)
    return backend_cls(uri)


# ---------------------------------------------------------------------------
# LatentDB facade ------------------------------------------------------------
# ---------------------------------------------------------------------------


class LatentDB:
    """High‑level object users interact with.

    Notes
    -----
    • Thread‑safe for independent short methods. Heavy compute is offloaded to
      an internal :class:`ThreadPoolExecutor` so notebooks remain snappy.
    • All public APIs raise *only* `ValueError`, `RuntimeError`, or
      `KeyError`—the rest are considered bugs.
    """

    _executor: ThreadPoolExecutor

    # ---------------------------------------------------------------------
    # Construction & lifecycle -------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(self, uri: str | Path, *, max_workers: int | None = 4):
        self._uri = str(uri)
        self._store: StorageBackend = _resolve_backend(self._uri)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        _LOG.info("LatentDB initialised with %s", uri)
        # Optional timing instrumentation (set LATENTDB_WRITE_LOG_INTERVAL=N to enable).
        self._write_count = 0
        self._write_accum = 0.0
        self._write_log_interval = int(os.environ.get("LATENTDB_WRITE_LOG_INTERVAL", "0") or 0)
        # Optional batching to cut round-trips (set LATENTDB_WRITE_BATCH_SIZE=N to enable).
        self._write_batch_size = int(os.environ.get("LATENTDB_WRITE_BATCH_SIZE", "0") or 400)
        self._write_flush_secs = float(os.environ.get("LATENTDB_WRITE_FLUSH_SECS", "0") or 0)
        self._write_buffer: list[ActivationEvent] = []
        self._last_flush = time.perf_counter()

    # ----------------------- context‑manager sugar -----------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D401
        self.close()

    def close(self):
        """Release underlying resources synchronously."""
        self.flush()
        self._executor.shutdown(wait=True)
        self._store.close()

    # ---------------------------------------------------------------------
    # Write path -----------------------------------------------------------
    # ---------------------------------------------------------------------

    def write_event(self, event: ActivationEvent) -> None:
        """Persist a single activation event.

        *Heavy JSON validation* happens here so downstream analysis can
        trust the schema.
        """
        t0 = time.perf_counter()
        try:
            event = ActivationEvent.parse_obj(event)  # idempotent if already validated
        except ValidationError as e:  # pragma: no cover
            raise ValueError(f"Invalid ActivationEvent: {e}") from None
        if self._write_batch_size > 0:
            self._write_buffer.append(event)
            elapsed = time.perf_counter() - self._last_flush
            if len(self._write_buffer) >= self._write_batch_size or (
                self._write_flush_secs and elapsed >= self._write_flush_secs
            ):
                self._flush_buffer()
        else:
            self._store.write_event(event)
        t1 = time.perf_counter()
        if self._write_log_interval:
            self._write_count += 1
            self._write_accum += t1 - t0
            if self._write_count % self._write_log_interval == 0:
                avg_ms = (self._write_accum / self._write_count) * 1e3
                _LOG.info(
                    "[latents] write_event count=%d avg=%.2fms last=%.2fms total=%.2fs",
                    self._write_count,
                    avg_ms,
                    (t1 - t0) * 1e3,
                    self._write_accum,
                )

    # ---------------------------------------------------------------------
    # Analysis -------------------------------------------------------------
    # ---------------------------------------------------------------------

    def compute_stats(self, *, min_count: int = 1, workers: int | None = None) -> None:
        """Compute/update :class:`StatBlock`s for all stored channels.

        This method *blocks* by default. For long‑running datasets you may
        pass ``workers=0`` and call :py:meth:`await_stats` later.
        """
        fut = self._executor.submit(
            self._store.compute_stats,
            min_count=min_count,        # ← pass as keyword
        )
        if workers is None or workers > 0:
            fut.result()

    def await_stats(self):
        """Convenience helper that blocks until all queued stat jobs finish."""
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=1)

    # ---------------------------------------------------------------------
    # Querying -------------------------------------------------------------
    # ---------------------------------------------------------------------

    def fetch_activations(
        self,
        *,
        layer: str,
        limit: int | None = None,
    ) -> list[ActivationEvent]:
        """
        Return ActivationEvent rows for *all channels* of the given layer,
        ordered by (step, channel).  Used by training datasets.
        """
        t0 = time.perf_counter()
        rows = self._store.fetch_activations(layer=layer, limit=limit)
        if self._write_log_interval:
            t1 = time.perf_counter()
            _LOG.info("[latents] fetch_activations layer=%s rows=%d time=%.2fms", layer, len(rows), (t1 - t0) * 1e3)
        return rows

    def iter_activations(self, *, layer: str, batch_size: int = 1000):
        """Yield activation batches for a layer (paged to reduce memory)."""
        t0 = time.perf_counter()
        total = 0
        for events in self._store.iter_activations(layer, batch_size):  # type: ignore[attr-defined]
            total += len(events)
            if self._write_log_interval:
                _LOG.info("[latents] iter_activations batch=%d total=%d layer=%s", len(events), total, layer)
            yield events
        if self._write_log_interval:
            t1 = time.perf_counter()
            _LOG.info("[latents] iter_activations layer=%s total=%d time=%.2fms", layer, total, (t1 - t0) * 1e3)

    def fetch_vectors(self, *, layer: str, limit: int | None = None):
        """Return dense activations and metadata for a layer, when supported by the backend."""
        return self._store.fetch_vectors(layer=layer, limit=limit)

    def get_block(self, *, run_id: str, layer: str, start: int, end: int):
        """Fast path: return (activations, index) for a contiguous step slice."""
        if hasattr(self._store, "get_block"):
            try:
                return self._store.get_block(run_id=run_id, layer=layer, start=start, end=end)  # type: ignore[attr-defined]
            except NotImplementedError:
                pass
        x, meta = self._store.fetch_vectors(layer=layer, limit=None)
        end = min(end, x.shape[0])
        if end <= start:
            return x[:0], {}
        sliced = x[start:end]
        out = {}
        for key, val in meta.items():
            if isinstance(val, list):
                out[key] = val[start:end]
            elif hasattr(val, "__getitem__"):
                out[key] = val[start:end]
            else:
                out[key] = val
        out["run_id"] = run_id
        return sliced, out

    def iter_events(
        self,
        *,
        run_id: str,
        layer: str,
        start: int = 0,
        end: int | None = None,
        channels: Sequence[int] | None = None,
    ):
        """Slow path: materialize ActivationEvent rows for a slice."""
        if hasattr(self._store, "iter_events"):
            yield from self._store.iter_events(  # type: ignore[attr-defined]
                run_id=run_id, layer=layer, start=start, end=end, channels=channels
            )
            return
        events = self._store.fetch_activations(layer=layer, limit=None)
        for ev in events:
            if ev.run_id != run_id:
                continue
            if ev.step < start or (end is not None and ev.step >= end):
                continue
            if channels is not None and ev.channel not in channels:
                continue
            yield ev


    def timeline(
        self,
        layer: str,
        channel: int,
        *,
        t0: float | None = None,
        t1: float | None = None,
        downsample: int = 1,
        as_array: bool = True,
    ) -> Any:
        """Return raw activation time‑series (optionally down‑sampled)."""
        events = self._store.fetch_events(layer, channel, t0, t1, downsample)
        return events if as_array else events.tolist()

    # ---------------------------------------------------------------------
    # Maintenance helpers --------------------------------------------------
    # ---------------------------------------------------------------------

    def find_dead(self, *, threshold: float = 1e-3) -> list[tuple[str, int]]:
        """Return channels whose mean absolute activation < *threshold*."""
        return [
            (sb.layer, sb.channel)
            for sb in self._store.iter_statblocks()
            if abs(sb.mean) < threshold
        ]
    
    def iter_statblocks(
        self,
        layer: str | None = None,
        channel: int | None = None,
    ) -> Iterable[StatBlock]:
        """Yield StatBlocks, optionally filtered by layer / channel."""
        yield from self._store.iter_statblocks(layer=layer, channel=channel)

    def flush(self) -> None:
        """Persist any buffered events immediately."""
        self._flush_buffer()
        self._store.flush()

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------
    # ------------------------------------------------------------------

    def _flush_buffer(self) -> None:
        if not self._write_buffer:
            return
        if hasattr(self._store, "write_events"):
            try:
                self._store.write_events(self._write_buffer)  # type: ignore[attr-defined]
            except Exception:
                # Fall back to single writes on error to avoid data loss.
                for ev in self._write_buffer:
                    self._store.write_event(ev)
        else:
            for ev in self._write_buffer:
                self._store.write_event(ev)
        self._write_buffer.clear()
        self._last_flush = time.perf_counter()

    # ---------------------------------------------------------------------
    # Magic methods --------------------------------------------------------
    # ---------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        sig = inspect.signature(self.__class__)
        params = ", ".join(f"{p}={getattr(self, p)}" for p in sig.parameters if hasattr(self, p))
        return f"<{self.__class__.__name__} {params}>"
