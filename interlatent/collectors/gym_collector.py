"""Collector for gym-like environments (PyTorch policy)."""
from __future__ import annotations

import time
import uuid
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
import torch

from interlatent.schema import ActivationEvent, RunInfo
from interlatent.api.latent_db import LatentDB
from interlatent.utils.logging import get_logger
from interlatent.metrics import Metric
from interlatent.hooks import PrePostHookCtx

try:  # TorchHook will be implemented in hooks.py; we import lazily.
    from interlatent.hooks import TorchHook  # type: ignore
except ImportError:  # pragma: no cover – until hooks.py exists
    TorchHook = None  # type: ignore

_LOG = get_logger(__name__)

__all__ = ["GymCollector"]


class GymCollector:
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

    def run(self, model: torch.nn.Module, env, *, steps: int = 10_000, tags: Optional[Dict] = None) -> RunInfo:
        """Execute *steps* interactions and record activations."""

        env_name = (
            getattr(getattr(env, "spec", None), "id", None)  # e.g. "CartPole-v1"
            or env.__class__.__name__                        # fallback: "CartPoleEnv"
        )

        action_space = env.action_space

        run_id = uuid.uuid4().hex
        run_info = RunInfo(run_id=run_id, env_name=env_name or str(env), tags=tags or {})

        model.eval().to(self.device)

        def policy(obs):
            # unwrap (obs, info) that Gymnasium reset/step may give back
            if isinstance(obs, (tuple, list)):
                obs = obs[0]

            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                out = model(x)

            if isinstance(out, tuple):
                out = out[0]
            elif isinstance(out, dict):
                out = out.get("logits") or out.get("action") or next(iter(out.values()))

            if not torch.is_tensor(out):
                raise TypeError("Model forward must return a Tensor, tuple or dict of Tensors")

            if out.dim() == 0 or (out.dim() == 1 and out.numel() == 1):
                return int(out.item())

            if hasattr(action_space, "n"):
                logits = out.squeeze(0)
                logits = logits[: action_space.n]
                return int(torch.argmax(logits).item())

            raise NotImplementedError("GymCollector demo-policy only supports discrete actions.")

        step_ctx: Dict[str, Any] = {}
        step_ctx["metrics"] = {}

        def ctx_supplier():  # closure visible to hooks
            return step_ctx

        hook_ctx = (
            PrePostHookCtx(
                model,
                layers=self.hook_layers,
                db=self.db,
                run_id=run_id,
                context_supplier=ctx_supplier,
                device=self.device,
            )
            if self.hook_layers
            else nullcontext()
        )

        with hook_ctx:
            obs, _ = env.reset()
            score = episode_len = 0

            for step in range(steps):
                step_ctx["step"] = step
                act = policy(obs)
                obs, reward, done, truncated, info = env.step(act)

                metric_vals = {}
                for m in self.metrics.values():
                    val = m.step(obs=obs, reward=reward, info=info)
                    if val is not None:
                        metric_vals[m.name] = float(val)
                step_ctx["metrics"] = metric_vals

                score += float(reward)
                episode_len += 1

                if done:
                    for m in self.metrics.values():
                        m.reset()
                    obs, _ = env.reset()
                    _LOG.debug(
                        "Episode done at step %d (len=%d, score=%f)", step, episode_len, score
                    )
                    score = episode_len = 0

        _LOG.info("Run %s finished (%d steps)", run_id, steps)
        self.db.flush()
        return run_info
