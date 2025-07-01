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

import importlib
import inspect
import textwrap
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

    # ----------------------- context‑manager sugar -----------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D401
        self.close()

    def close(self):
        """Release underlying resources synchronously."""
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
        try:
            event = ActivationEvent.parse_obj(event)  # idempotent if already validated
        except ValidationError as e:  # pragma: no cover
            raise ValueError(f"Invalid ActivationEvent: {e}") from None
        self._store.write_event(event)

    # ---------------------------------------------------------------------
    # Analysis -------------------------------------------------------------
    # ---------------------------------------------------------------------

    def compute_stats(self, *, batch_size: int = 512, workers: int | None = None) -> None:
        """Compute/update :class:`StatBlock`s for all stored channels.

        This method *blocks* by default. For long‑running datasets you may
        pass ``workers=0`` and call :py:meth:`await_stats` later.
        """
        fut = self._executor.submit(self._store.compute_stats, batch_size)
        if workers is not None:
            # fire‑and‑forget when workers=0 (or negative); otherwise block
            if workers > 0:
                fut.result()
        else:
            fut.result()

    def await_stats(self):
        """Convenience helper that blocks until all queued stat jobs finish."""
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=1)

    # ---------------------------------------------------------------------
    # LLM explanations -----------------------------------------------------
    # ---------------------------------------------------------------------

    def autodescribe(self, *, llm: str = "openai", overwrite: bool = False, **kwargs: Any) -> None:
        """Generate or refresh textual explanations for channels without one.

        Parameters
        ----------
        llm:
            Backend identifier, e.g. ``"openai"`` or ``"local"``.
        overwrite:
            If ``True``, regenerate even if an explanation string exists.
        kwargs:
            Extra kwargs forwarded to the LLM backend (e.g. ``model="gpt-4o"``).
        """
        from ..explain import load_backend  # local import to avoid cycles

        backend = load_backend(llm, **kwargs)
        missing: list[StatBlock] = list(self._store.unexplained(overwrite))
        _LOG.info("Generating %d explanations via %s", len(missing), llm)

        for sb in missing:
            prompt = self._explain_prompt(sb)
            text = backend(prompt)
            self._store.write_explanation(Explanation.from_statblock(sb, text))

    # -- helper ------------------------------------------------------------
    @staticmethod
    def _explain_prompt(sb: StatBlock) -> str:
        template = textwrap.dedent(
            """
            You are analysing a neural‑network latent channel. Here are its
            summary statistics and correlations. Respond with a *concise*,
            lay‑readable blurb about what this channel detects.

            LAYER   : {layer}
            CHANNEL : {channel}
            MEAN    : {mean:.4f}
            STD     : {std:.4f}
            TOP‑CORR: {correlations}
            """
        )
        return template.format(
            layer=sb.layer,
            channel=sb.channel,
            mean=sb.mean,
            std=sb.std,
            correlations=sb.top_correlations[:5],
        )

    # ---------------------------------------------------------------------
    # Querying -------------------------------------------------------------
    # ---------------------------------------------------------------------

    def describe(self, layer: str, channel: int, *, as_dict: bool = False) -> str | dict[str, Any]:
        """Return the human‑readable description for a latent channel."""
        expl = self._store.fetch_explanation(layer, channel)
        if expl is None:
            raise KeyError(f"No explanation for {layer}:{channel}")
        return expl.dict() if as_dict else expl.text

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
    # Chat‑style free‑text QA ---------------------------------------------
    # ---------------------------------------------------------------------

    def chat(self, prompt: str, *, llm: str = "openai", **kwargs: Any) -> str:
        """Answer an arbitrary question about stored activations.

        For now this is a thin wrapper that stuffs the entire prompt and
        perhaps some retrieved docs into an LLM. Future versions may use a
        proper RAG setup.
        """
        from ..explain import load_backend  # local import to avoid cycles

        # naive retrieval: top‑k explanations whose text matches any word
        # in the prompt. Replace with vector search later.
        docs = self._store.search_explanations(prompt, k=20)
        context = "\n\n".join(d.text for d in docs)
        meta = f"Context docs (k={len(docs)}). Answer the user question concisely."
        full_prompt = f"{meta}\n\n{context}\n\nQuestion: {prompt}\nAnswer:"

        backend = load_backend(llm, **kwargs)
        return backend(full_prompt)

    # ---------------------------------------------------------------------
    # Maintenance helpers --------------------------------------------------
    # ---------------------------------------------------------------------

    def prune_explanations(self, *, keep_most_recent: int = 3):
        """Drop old explanation drafts, keeping only the newest *n* per channel."""
        self._store.prune_explanations(keep_most_recent)

    def find_dead(self, *, threshold: float = 1e-3) -> list[tuple[str, int]]:
        """Return channels whose mean absolute activation < *threshold*."""
        return [
            (sb.layer, sb.channel)
            for sb in self._store.iter_statblocks()
            if abs(sb.mean) < threshold
        ]

    # ---------------------------------------------------------------------
    # Magic methods --------------------------------------------------------
    # ---------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        sig = inspect.signature(self.__class__)
        params = ", ".join(f"{p}={getattr(self, p)}" for p in sig.parameters if hasattr(self, p))
        return f"<{self.__class__.__name__} {params}>"
