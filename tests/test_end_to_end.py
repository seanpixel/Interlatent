"""End-to-end sanity check for Interlatent.

Runs a tiny linear policy on Gym's CartPole for ~100 steps, captures
activations via TorchHook, stores them in an in‑memory SQLite DB, and
verifies that stats can be computed.

Requires optional deps: torch, gymnasium (or gym <0.26).
"""
from __future__ import annotations

import tempfile

import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

from interlatent.api import LatentDB
from interlatent.collector import Collector


def test_cartpole_integration(tmp_path):
    # ------------------------------------------------------------------
    # 1. Setup DB and model --------------------------------------------
    # ------------------------------------------------------------------
    db_uri = f"sqlite:///{tmp_path}/test.db"
    db = LatentDB(db_uri)

    model = torch.nn.Sequential(torch.nn.Linear(4, 8))

    # ------------------------------------------------------------------
    # 2. Run a short episode -------------------------------------------
    # ------------------------------------------------------------------
    env = gym.make("CartPole-v1")

    collector = Collector(db=db, hook_layers=["0"], batch_size=1, device="cpu")
    run_info = collector.run(model, env, steps=128)

    print(run_info)

    # ------------------------------------------------------------------
    # 3. Stats computation ---------------------------------------------
    # ------------------------------------------------------------------
    db.compute_stats()

    statblocks = list(db.iter_statblocks(layer=0, channel=1))
    assert statblocks, "compute_stats produced no StatBlocks"

    # Basic sanity: means should be finite numbers.
    for sb in statblocks:
        assert abs(sb.mean) < 1e6

    # No explanations yet → describe should raise
    import pytest

    with pytest.raises(KeyError):
        db.describe(layer="0", channel=0)

    # ------------------------------------------------------------------
    # 4. correlation ---------------------------------------------------
    # ------------------------------------------------------------------
    statblocks = list(db.iter_statblocks())
    sb = statblocks[0]
    assert sb.top_correlations, "metric correlation missing"
    label, rho = sb.top_correlations[0]
    assert label == "metric"
    assert -1.0 <= rho <= 1.0

    db.close()

