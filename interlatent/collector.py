"""interlatent.collector

Utility that runs a model in an environment or dataloader, streams
ActivationEvents into a LatentDB, and returns basic run statistics.

Intended for quick experiments—not a full RL rollout manager.  Works with
any env that exposes the classic Gym API (`reset`, `step`).
"""
from __future__ import annotations

import time
import uuid
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
import torch

from .schema import ActivationEvent, RunInfo
from .api.latent_db import LatentDB
from .utils.logging import get_logger
from .metrics import Metric

try:  # TorchHook will be implemented in hooks.py; we import lazily.
    from .hooks import TorchHook  # type: ignore
except ImportError:  # pragma: no cover – until hooks.py exists
    TorchHook = None  # type: ignore

_LOG = get_logger(__name__)

__all__ = ["Collector"]


class Collector:
    """Streams activations during a simulation run into a LatentDB."""

    def __init__(
        self,
        db: LatentDB,
        *,
        metric_fns: list[Metric] | None = None,
        hook_layers: Sequence[str] | None = None,
        batch_size: int = 1,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.db = db
        self.metrics = {m.name: m for m in (metric_fns or [])}
        self.hook_layers = hook_layers or []  # empty → no activations captured
        self.batch_size = batch_size
        self.device = device

    # ------------------------------------------------------------------
    # Public entry ------------------------------------------------------
    # ------------------------------------------------------------------

    def run(self, model: torch.nn.Module, env, *, steps: int = 10_000, tags: Optional[Dict] = None) -> RunInfo:
        """Execute *steps* interactions and record activations.

        Parameters
        ----------
        model:
            PyTorch policy/network; must accept env observations as first arg.
        env:
            Object implementing `reset() -> obs` and `step(action)`.
        steps:
            Number of timesteps to collect.
        tags:
            Arbitrary user metadata (seed, difficulty…).
        """

        env_name = (
            getattr(getattr(env, "spec", None), "id", None)  # e.g. "CartPole-v1"
            or env.__class__.__name__                        # fallback: "CartPoleEnv"
        )
                
        run_id = uuid.uuid4().hex
        run_info = RunInfo(run_id=run_id, env_name=env_name or str(env), tags=tags or {})

        model.eval().to(self.device)

        # Determine action function. We assume model(obs_tensor) → tensor action logits or direct action.
        def policy(obs):
            # Gymnasium sometimes returns (obs, info)
            if isinstance(obs, (tuple, list)):
                obs = obs[0]

            with torch.no_grad():
                logits = model(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                ).squeeze(0)         # shape: (out_dim,)

            # --- map model output → valid env action ---
            if hasattr(env.action_space, "n"):          # Discrete(n)
                logits = logits[: env.action_space.n]   # trim if model has extra dims
                return int(torch.argmax(logits).item())

            # You can extend this branch for Box / MultiDiscrete later
            raise NotImplementedError("Collector demo-policy only supports discrete actions.")
        
        step_ctx: Dict[str, Any] = {}
        # Use TorchHook if available; else run without hooks.
        hook_ctx = (
            TorchHook(model, layers=self.hook_layers, db=self.db, run_id=run_id, context_supplier=lambda: step_ctx,)  # type: ignore
            if TorchHook and self.hook_layers
            else nullcontext()
        )

        with hook_ctx:
            obs, _info = env.reset()
            score, episode_len = 0.0, 0

            for step in range(steps):
                act = policy(obs)
                obs, reward, done, truncated, info = env.step(act)
                score += float(reward)
                episode_len += 1

                metric_vals = {}
                for m in self.metrics.values():
                    val = m.step(obs=obs, reward=reward, info=info)
                    if val is not None:
                        metric_vals[m.name] = float(val)

                step_ctx.clear()
                if metric_vals:
                    step_ctx["metrics"] = metric_vals

                if done:
                    for m in self.metrics.values():
                        m.reset()
                    obs = env.reset()
                    _LOG.debug("Episode done at step %d (len=%d, score=%f)", step, episode_len, score)
                    episode_len, score = 0, 0.0

        _LOG.info("Run %s finished (%d steps)", run_id, steps)
        self.db.flush()
        return run_info
