"""interlatent.schema

Shared data objects that flow through the Interlatent pipeline.
Every bit that crosses module boundaries or hits persistent storage
_validates_ against one of these Pydantic models.  Think of them as the
contract binding Collector ↔ Trainer ↔ LLM ↔ UI.

Tables & Lineage
----------------
1. **runs**           – metadata for a replay / simulation episode
2. **activations**    – many per‑run tensor snapshots (`ActivationEvent`)
3. **stats**          – aggregate properties of a channel (`StatBlock`)
4. **explanations**   – human‑readable blurbs (`Explanation`)
5. **artifacts**      – model files (e.g. trained transcoders)
"""
from __future__ import annotations

import datetime as _dt
import uuid as _uuid_mod
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Utilities -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _now() -> str:
    """Return current UTC time in ISO‑8601 with trailing Z."""
    return _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _uuid() -> str:  # noqa: D401 – function not method
    return _uuid_mod.uuid4().hex


# ---------------------------------------------------------------------------
# 0. RunInfo -----------------------------------------------------------------
# ---------------------------------------------------------------------------

class RunInfo(BaseModel):
    """Metadata about a single simulation/game episode."""

    run_id: str = Field(default_factory=_uuid, description="Primary key shared by all events in this run.")
    env_name: str = Field(..., description="Gym environment or dataset identifier.")
    start_time: str = Field(default_factory=_now)
    tags: Dict[str, Any] = Field(default_factory=dict, description="User‑supplied arbitrary labels (seed, difficulty, …).")

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# 1. ActivationEvent ---------------------------------------------------------
# ---------------------------------------------------------------------------

class ActivationEvent(BaseModel):
    """Flattened activation tensor captured at a single forward step."""

    # Composite primary key → (run_id, layer, channel, step)
    run_id: str = Field(...)
    step: int = Field(..., ge=0, description="Timestep or frame index within the run.")
    layer: str = Field(...)
    channel: int = Field(..., ge=0)
    prompt: str | None = Field(None, description="Source prompt text for this activation slice.")
    prompt_index: int | None = Field(None, ge=0, description="Index of the prompt within the run/dataset.")
    token_index: int | None = Field(None, ge=0, description="Token position within the prompt.")
    token: str | None = Field(None, description="Tokenizer surface form for the token at token_index.")

    value_sum: float | None = None
    value_sq_sum: float | None = None

    tensor: List[float] = Field(..., description="Flattened float32 tensor.")
    timestamp: str = Field(default_factory=_now, description="Wall‑clock capture time (UTC ISO).")
    context: Dict[str, Any] = Field(default_factory=dict, description="Instantaneous env info (score, x_pos, etc.)")

    # -- validation ---------------------------------------------------------
    @validator("tensor", pre=True)
    def _flatten_numpy(cls, v):  # noqa: N805
        if isinstance(v, np.ndarray):
            return v.astype(np.float32).ravel().tolist()
        if isinstance(v, (list, tuple)):
            return list(v)
        raise TypeError("tensor must be list/tuple/np.ndarray")

    class Config:
        frozen = True
        json_encoders = {np.ndarray: lambda arr: arr.tolist()}


# ---------------------------------------------------------------------------
# 2. StatBlock ---------------------------------------------------------------
# ---------------------------------------------------------------------------

class StatBlock(BaseModel):
    """Aggregated statistics for a given (layer, channel)."""

    layer: str
    channel: int

    count: int = Field(..., gt=0)
    mean: float
    std: float
    min: float
    max: float

    # List of ("other_layer:idx", pearson_corr) sorted by |corr| desc.
    top_correlations: List[Tuple[str, float]] = Field(default_factory=list)

    last_updated: str = Field(default_factory=_now)

    # Convenience -----------------------------------------------------------
    @classmethod
    def from_array(cls, layer: str, channel: int, arr: Sequence[float]):
        arr_np = np.asarray(arr, dtype=np.float32)
        return cls(
            layer=layer,
            channel=channel,
            count=arr_np.size,
            mean=float(arr_np.mean()),
            std=float(arr_np.std()),
            min=float(arr_np.min()),
            max=float(arr_np.max()),
        )


# ---------------------------------------------------------------------------
# 3. Explanation -------------------------------------------------------------
# ---------------------------------------------------------------------------

class Explanation(BaseModel):
    """Human‑authored description of what a latent detects."""

    layer: str
    channel: int
    version: int = Field(1, ge=1, description="Monotonic revision number per channel.")
    text: str = Field(..., description="Concise prose <= 500 chars.")
    source: str = Field("llm", description="Origin (llm, human, etc.)")
    created_at: str = Field(default_factory=_now)

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# 4. Artifact (e.g. trained transcoder) -------------------------------------
# ---------------------------------------------------------------------------

class Artifact(BaseModel):
    """Binary blob on disk/S3 plus searchable metadata."""

    artifact_id: str = Field(default_factory=_uuid)
    kind: str = Field(..., description="'transcoder', 'checkpoint', …")
    path: str = Field(..., description="Filesystem or S3 path to the file.")

    meta: Mapping[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# Public export --------------------------------------------------------------
# ---------------------------------------------------------------------------

__all__ = [
    "RunInfo",
    "ActivationEvent",
    "StatBlock",
    "Explanation",
    "Artifact",
]
