"""
Token-level analysis for the Ministral character experiment.

Targets specific tokens/groups (e.g., "I", "Husband", "Kinda", slang) and
reports how activations differ across the 4 character labels.

Usage:
  PYTHONPATH=. python demos/ministral_characters_experiment/token_level_analysis.py \
    --db ./latents_character_dilemmas.db \
    --layer latent_sae:llm.layer.30 \
    --out_dir vis/token_level
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from interlatent.api import LatentDB


DEFAULT_GROUPS = {
    "self": ["I", "i", "i ", "me", "my", "mine", "myself"],
    "slang": ["gonna", "wanna", "gotta", "kinda", "sorta", "ya", "y'all", "ain't"],
    "report": ["report", "reported", "reporting"],
}


def normalize_token(tok: str) -> str:
    # SentencePiece uses ▁ as a whitespace marker; normalize to a plain space and lower.
    return tok.replace("▁", " ").strip().lower()


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


def load_events(db_path: Path, layer: str):
    db = LatentDB(f"sqlite:///{db_path}")
    events = db.fetch_activations(layer=layer)
    db.close()
    if not events:
        raise RuntimeError(f"No activations found for layer '{layer}' in {db_path}")
    return events


def build_group_sets(groups: Dict[str, List[str]]) -> Dict[str, set]:
    out = {}
    for name, words in groups.items():
        out[name] = {normalize_token(w) for w in words}
    return out


def aggregate(
    events: Iterable,
    group_sets: Dict[str, set],
) -> Tuple[Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]], List[int]]:
    """
    Returns:
      group_stats[group][label][channel] = (mean, count, max)
      labels: sorted label list
    """
    sums: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts: Dict[str, Dict[int, Dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    maxes: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: float("-inf")))
    )
    labels_seen = set()

    for ev in events:
        if not ev.token:
            continue
        label = extract_label(ev)
        if label is None:
            continue
        tok_norm = normalize_token(ev.token)
        for group, variants in group_sets.items():
            if tok_norm in variants:
                labels_seen.add(label)
                ch = ev.channel
                val = ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0)
                sums[group][label][ch] += float(val)
                counts[group][label][ch] += 1
                maxes[group][label][ch] = max(maxes[group][label][ch], float(val))

    labels = sorted(labels_seen)
    stats: Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]] = defaultdict(dict)
    for group, label_map in sums.items():
        stats[group] = {}
        for label, ch_map in label_map.items():
            stats[group][label] = {}
            for ch, total in ch_map.items():
                cnt = counts[group][label][ch]
                stats[group][label][ch] = (total / cnt if cnt else 0.0, cnt, maxes[group][label][ch])
    return stats, labels


def summarize_variation(
    stats: Dict[int, Dict[int, Tuple[float, int, float]]],
    labels: List[int],
    top_k: int = 10,
) -> List[str]:
    """
    Summarize channels with largest max-min spread across labels.
    """
    if not labels:
        return []
    # Collect channel means by label.
    channels = set()
    for label in labels:
        channels.update(stats.get(label, {}).keys())

    rows = []
    for ch in sorted(channels):
        means = []
        for label in labels:
            mean = stats.get(label, {}).get(ch, (0.0, 0, 0.0))[0]
            means.append(mean)
        spread = max(means) - min(means)
        rows.append((ch, spread, means))

    rows.sort(key=lambda r: r[1], reverse=True)
    lines = []
    for ch, spread, means in rows[:top_k]:
        parts = " ".join(f"{lab}:{means[i]:+.4f}" for i, lab in enumerate(labels))
        lines.append(f"ch {ch:4d} | spread {spread:.4f} | means {parts}")
    return lines


def write_csv(
    out_path: Path,
    group_stats: Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "label", "channel", "mean", "count", "max"])
        for group, label_map in group_stats.items():
            for label, ch_map in label_map.items():
                for ch, (mean, count, max_val) in ch_map.items():
                    writer.writerow([group, label, ch, f"{mean:.6f}", count, f"{max_val:.6f}"])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("latents_character_dilemmas.db"))
    ap.add_argument("--layer", type=str, default="latent_sae:llm.layer.30")
    ap.add_argument("--out_dir", type=Path, default=Path("vis/token_level"))
    ap.add_argument("--topk", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    events = load_events(args.db, args.layer)
    group_sets = build_group_sets(DEFAULT_GROUPS)
    group_stats, labels = aggregate(events, group_sets)

    out_csv = args.out_dir / "token_level_stats.csv"
    write_csv(out_csv, group_stats)
    print(f"[token-level] Wrote stats to {out_csv}")

    for group in sorted(group_stats.keys()):
        print(f"\n[group] {group}")
        lines = summarize_variation(group_stats[group], labels, top_k=args.topk)
        if not lines:
            print("  (no matches)")
            continue
        for line in lines:
            print("  " + line)


if __name__ == "__main__":
    main()
