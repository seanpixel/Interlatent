"""interlatent.storage.base

Abstract persistence layer.
All concrete back‑ends—SQLite, DynamoDB, Postgres, etc.—inherit from
:class:`StorageBackend` and implement the same small surface area so the
rest of the library can stay blissfully ignorant of where bits live.

Design notes
------------
* Keep the interface *minimal but complete*—only methods required by
  `LatentDB` and training/LLM workers appear here.
* The contract is **sync**, not async.  Async back‑ends can wrap their
  I/O but must present blocking semantics here.
* Docs & unit tests will lock this API; breaking changes need version
  bumps.
"""
from __future__ import annotations

import abc
from typing import Iterable, List, Sequence, Tuple

from ..schema import ActivationEvent, Artifact, Explanation, StatBlock

__all__ = ["StorageBackend"]


class StorageBackend(abc.ABC):
    """Abstract base class for persistent storage."""

    # ---------------------------------------------------------------------
    # Construction / teardown --------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(self, uri: str):
        self._uri = uri

    # Concrete back‑ends must ensure tables exist when instantiated.

    # ------------------------------
    def close(self):  # noqa: D401 – not a context manager here
        """Optional: close DB connections / flush buffers."""
        pass

    # ---------------------------------------------------------------------
    # Write path -----------------------------------------------------------
    # ---------------------------------------------------------------------

    @abc.abstractmethod
    def write_event(self, ev: ActivationEvent) -> None:  # pragma: no cover
        """Persist an :class:`ActivationEvent`."""

    @abc.abstractmethod
    def write_statblock(self, sb: StatBlock) -> None:  # pragma: no cover
        """Insert or update a :class:`StatBlock`."""

    @abc.abstractmethod
    def write_explanation(self, ex: Explanation) -> None:  # pragma: no cover
        """Insert an :class:`Explanation` (new version row)."""

    @abc.abstractmethod
    def write_artifact(self, art: Artifact) -> None:  # pragma: no cover
        """Register a file/weights blob in the artifact catalogue."""

    # ---------------------------------------------------------------------
    # Read / query ---------------------------------------------------------
    # ---------------------------------------------------------------------

    @abc.abstractmethod
    def fetch_events(
        self,
        layer: str,
        channel: int,
        t0: float | None = None,
        t1: float | None = None,
        downsample: int = 1,
    ) -> Sequence[float]:  # pragma: no cover
        """Return flattened activation values satisfying the filter."""

    @abc.abstractmethod
    def fetch_activations(self, *, layer: str, limit: int | None = None) -> List[ActivationEvent]:
        """Return ActivationEvent rows for the given layer."""

    @abc.abstractmethod
    def fetch_vectors(self, *, layer: str, limit: int | None = None):
        """Return (x, meta) where x is a dense array of activations for the layer."""

    @abc.abstractmethod
    def unexplained(self, overwrite: bool) -> Iterable[StatBlock]:  # pragma: no cover
        """Yield StatBlocks needing a (new) explanation."""
    
    @abc.abstractmethod
    def iter_statblocks(self) -> Iterable[StatBlock]:  # pragma: no cover
        """Stream all :class:`StatBlock`s (used for pruning, etc.)."""

    # ---------------------------------------------------------------------
    # Analysis helpers -----------------------------------------------------
    # ---------------------------------------------------------------------

    @abc.abstractmethod
    def compute_stats(self, *, min_count: int = 1) -> None:  # pragma: no cover
        """Scan activations and store/update :class:`StatBlock`s."""

    # ---------------------------------------------------------------------
    # House‑keeping --------------------------------------------------------
    # ---------------------------------------------------------------------

    @abc.abstractmethod
    def iter_statblocks(
        self,
        layer: str | None = None,
        channel: int | None = None,
    ) -> Iterable[StatBlock]: ...

    @abc.abstractmethod
    def flush(self) -> None:
        """Force-commit any buffered writes to the underlying store."""

    # Optional: paged activation iteration (implement where supported).
    def iter_activations(self, layer: str, batch_size: int = 1000):
        """Yield batches of ActivationEvents for the given layer."""
        raise NotImplementedError

    # Optional: row-wise access for scalable backends.
    def get_block(self, *, run_id: str, layer: str, start: int, end: int):
        """Return (activations, index) for a contiguous step slice."""
        raise NotImplementedError

    def iter_events(
        self,
        *,
        run_id: str,
        layer: str,
        start: int = 0,
        end: int | None = None,
        channels: Sequence[int] | None = None,
    ):
        """Yield ActivationEvents for a slice, optionally filtered by channels."""
        raise NotImplementedError
