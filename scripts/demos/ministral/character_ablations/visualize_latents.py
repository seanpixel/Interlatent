"""
Visualize per-character latent activations for the Ministral character experiment.

Loads activations from a LatentDB, computes mean activation per channel per
character label, plots a heatmap, and prints the top varying channels.

Usage:
  PYTHONPATH=. python scripts/demos/ministral/character_ablations/visualize_latents.py \
    --db latents_character_dilemmas.db \
    --layer latent_sae:llm.layer.20 \
    --output figs/character_latents.png
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from interlatent.api import LatentDB


def load_events(db_path: Path, layer: str):
    db = LatentDB(f"sqlite:///{db_path}")
    events = db.fetch_activations(layer=layer)
    db.close()
    if not events:
        raise RuntimeError(f"No activations found for layer '{layer}' in {db_path}")
    return events


def extract_label(ev) -> int | None:
    ctx = ev.context or {}
    metrics = ctx.get("metrics", {}) if isinstance(ctx, dict) else {}
    val = metrics.get("prompt_label") or ctx.get("prompt_label")
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def aggregate(events) -> Tuple[np.ndarray, List[int]]:
    # Collect per label per channel sums/counts
    sums: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    max_channel = 0
    labels_seen = set()

    for ev in events:
        label = extract_label(ev)
        if label is None:
            continue
        labels_seen.add(label)
        ch = ev.channel
        max_channel = max(max_channel, ch)
        val = ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0)
        sums[label][ch] += float(val)
        counts[label][ch] += 1

    labels = sorted(labels_seen)
    num_channels = max_channel + 1
    mat = np.zeros((num_channels, len(labels)), dtype=np.float32)

    for li, label in enumerate(labels):
        for ch in range(num_channels):
            if counts[label][ch] > 0:
                mat[ch, li] = sums[label][ch] / counts[label][ch]
            else:
                mat[ch, li] = 0.0
    return mat, labels


def plot_heatmap(mat: np.ndarray, labels: List[int], output: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Mean activation")
    plt.xlabel("Character label")
    plt.ylabel("Channel")
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.title("Mean activation per channel per character")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def report_variation(mat: np.ndarray, labels: List[int], top_k: int = 10):
    # Variation across characters per channel: max - min
    var = mat.max(axis=1) - mat.min(axis=1)
    top_idx = np.argsort(-var)[:top_k]
    lines = []
    for ch in top_idx:
        lines.append(
            f"ch {ch:4d} | spread {var[ch]:.4f} | means " +
            " ".join(f"{lab}:{mat[ch, i]:.3f}" for i, lab in enumerate(labels))
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True, help="Path to LatentDB sqlite file")
    ap.add_argument("--layer", type=str, default="latent_sae:llm.layer.20", help="Layer name in DB")
    ap.add_argument("--output", type=Path, default=Path("figs/character_latents.png"))
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    events = load_events(args.db, args.layer)
    mat, labels = aggregate(events)
    if mat.size == 0:
        raise RuntimeError("No activations aggregated; check labels/metrics.")

    plot_heatmap(mat, labels, args.output)
    print(f"[viz] Saved heatmap to {args.output}")
    lines = report_variation(mat, labels, top_k=args.topk)
    print("[viz] Top varying channels (max-min spread across characters):")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    main()
