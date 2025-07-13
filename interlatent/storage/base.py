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
    def fetch_explanation(self, layer: str, channel: int) -> Explanation | None:  # pragma: no cover
        """Return most‑recent explanation or ``None`` if absent."""
        
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
    # Search utilities -----------------------------------------------------
    # ---------------------------------------------------------------------

    @abc.abstractmethod
    def search_explanations(self, query: str, k: int = 10) -> List[Explanation]:  # pragma: no cover
        """Naïve full‑text search; back‑ends may override with FTS/vectors."""

    # ---------------------------------------------------------------------
    # House‑keeping --------------------------------------------------------
    # ---------------------------------------------------------------------

    @abc.abstractmethod
    def prune_explanations(self, *, keep_most_recent: int = 3) -> None:  # pragma: no cover
        """Drop old explanation versions per channel, keeping *n* newest."""

    @abc.abstractmethod
    def iter_statblocks(
        self,
        layer: str | None = None,
        channel: int | None = None,
    ) -> Iterable[StatBlock]: ...

    @abc.abstractmethod
    def flush(self) -> None:
        """Force-commit any buffered writes to the underlying store."""
