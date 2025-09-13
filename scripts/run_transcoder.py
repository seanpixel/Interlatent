#!/usr/bin/env python3
"""Run a configurable transcoder experiment.

This script generalizes the behaviour of ``tests/test_transcoder.py`` so that
users can try different Gym environments and stable-baselines3 models without
modifying source code.  It collects activations, trains a transcoder for the
specified layer and reports the strongest correlations.
"""

import argparse
from importlib import import_module
from pathlib import Path

import gymnasium as gym
from interlatent.api import LatentDB
from interlatent.collector import Collector
from interlatent.metrics import LambdaMetric
from interlatent.train.pipeline import TranscoderPipeline


def parse_metric(spec: str):
    """Parse a ``name:expr`` specification into a LambdaMetric.

    The expression is evaluated with ``obs`` in scope.  Example::

        pole_angle:obs[2]
    """
    name, expr = spec.split(":", 1)

    def fn(obs, **kwargs):
        return eval(expr, {"obs": obs})

    return LambdaMetric(name, fn)


def load_policy(algo: str, path: str):
    """Dynamically load a Stable-Baselines3 policy from ``path``."""
    sb3 = import_module("stable_baselines3")
    cls = getattr(sb3, algo)
    return cls.load(path).policy


def main():
    p = argparse.ArgumentParser(description="Run transcoder experiment")
    p.add_argument("--env-id", required=True, help="Gym environment id")
    p.add_argument("--algo", default="PPO", help="SB3 algorithm class name")
    p.add_argument("--model-path", required=True, help="Path to trained SB3 model")
    p.add_argument("--steps", type=int, default=200, help="Steps to collect")
    p.add_argument("--epochs", type=int, default=3, help="Transcoder training epochs")
    p.add_argument("--k", type=int, default=4, help="Bottleneck dimension")
    p.add_argument("--layer", default="mlp_extractor.policy_net.0",
                   help="Model layer to hook")
    p.add_argument("--metric", action="append",
                   help="Optional metric as name:expr evaluated on obs")
    p.add_argument("--db-path", default="latents.db",
                   help="Where to store the SQLite database")
    args = p.parse_args()

    env = gym.make(args.env_id)
    policy = load_policy(args.algo, args.model_path)

    db = LatentDB(f"sqlite:///{Path(args.db_path).resolve()}")

    metrics = [parse_metric(m) for m in args.metric] if args.metric else []
    collector = Collector(db, hook_layers=[args.layer], metric_fns=metrics)
    collector.run(policy, env, steps=args.steps)

    pipe = TranscoderPipeline(db, args.layer, k=args.k, epochs=args.epochs)
    pipe.run()

    if metrics:
        db.compute_stats(min_count=1)
        latent_layer = f"latent:{args.layer}"
        for sb in db.iter_statblocks(layer=latent_layer):
            print(sb.top_correlations)

    db.close()


if __name__ == "__main__":
    main()

