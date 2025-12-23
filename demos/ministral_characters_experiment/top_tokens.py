"""
Inspect which tokens most strongly activate each latent/channel.

Usage:
  PYTHONPATH=. python demos/ministral_characters_experiment/top_tokens.py \
    --db latents_character_dilemmas.db \
    --layer latent_sae:llm.layer.30 \
    --topk 10 \
    --min_count 3
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from interlatent.api import LatentDB


def load_events(db_path: Path, layer: str, limit: int | None = None):
    db = LatentDB(f"sqlite:///{db_path}")
    events = db.fetch_activations(layer=layer, limit=limit)
    db.close()
    if not events:
        raise RuntimeError(f"No activations found for layer '{layer}' in {db_path}")
    return events


def aggregate_token_stats(events: Iterable, min_count: int) -> Dict[int, Dict[str, Tuple[float, int, float]]]:
    """
    Return per-channel token stats: mean activation, count, and max activation.
    """
    stats: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, float("-inf")]))

    for ev in events:
        token = ev.token
        if token is None:
            continue
        val = ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0)
        acc = stats[ev.channel][token]
        acc[0] += float(val)
        acc[1] += 1
        acc[2] = max(acc[2], float(val))

    # finalize: filter by min_count and compute mean
    filtered: Dict[int, Dict[str, Tuple[float, int, float]]] = defaultdict(dict)
    for ch, token_map in stats.items():
        for token, (sum_val, count, max_val) in token_map.items():
            if count >= min_count:
                filtered[ch][token] = (sum_val / count, count, max_val)
    return filtered


def report_top_tokens(stats: Dict[int, Dict[str, Tuple[float, int, float]]], topk: int):
    for ch in sorted(stats.keys()):
        tokens = sorted(stats[ch].items(), key=lambda kv: kv[1][0], reverse=True)[:topk]
        if not tokens:
            continue
        print(f"channel {ch}:")
        for token, (mean_val, count, max_val) in tokens:
            print(f"  {token!r:12s} mean={mean_val: .4f} count={count:3d} max={max_val: .4f}")
        print()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True, help="Path to LatentDB sqlite file")
    ap.add_argument("--layer", type=str, required=True, help="Layer name to query, e.g. latent_sae:llm.layer.30")
    ap.add_argument("--topk", type=int, default=10, help="Top tokens per channel")
    ap.add_argument("--min_count", type=int, default=3, help="Minimum occurrences of a token to be reported")
    ap.add_argument("--limit", type=int, default=None, help="Optional hard cap on activations to scan")
    return ap.parse_args()


def main():
    args = parse_args()
    events = load_events(args.db, args.layer, limit=args.limit)
    stats = aggregate_token_stats(events, args.min_count)
    if not stats:
        raise RuntimeError("No tokens met min_count; try lowering --min_count.")
    report_top_tokens(stats, args.topk)


if __name__ == "__main__":
    main()
