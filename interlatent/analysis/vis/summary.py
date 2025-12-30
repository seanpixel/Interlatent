"""
Lightweight CLI utilities to inspect an Interlatent database without
pulling data into pandas. Designed for quick terminal summaries.

Usage:
  python -m interlatent.analysis.vis.summary sqlite:///latents_llm_local.db
  python -m interlatent.analysis.vis.summary hdf5:///latents_llm_local.h5 --limit 5
"""
from __future__ import annotations

import argparse
import json
from typing import List, Sequence, Tuple, Optional

from interlatent.api import LatentDB

def _list_layers(db: LatentDB, prefix: str | None = None) -> list[str]:
    if hasattr(db._store, "list_layers"):
        layers = db._store.list_layers()
        if prefix:
            layers = [l for l in layers if l.startswith(prefix)]
        return layers
    raise SystemExit("Backend does not support listing layers.")


def _format_table(headers: Sequence[str], rows: Sequence[Sequence], max_width: int = 24) -> str:
    # Compute column widths with truncation.
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            cell = "" if row[i] is None else str(row[i])
            widths[i] = min(max(widths[i], len(cell)), max_width)

    def _fmt_cell(val, width):
        text = "" if val is None else str(val)
        if len(text) > width:
            text = text[: width - 1] + "…"
        return text.ljust(width)

    sep = " | "
    lines = []
    lines.append(sep.join(_fmt_cell(h, widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(sep.join(_fmt_cell(row[i], widths[i]) for i in range(cols)))
    return "\n".join(lines)


def _ascii_bars(items: Sequence[Tuple[str, int]], width: int = 40) -> str:
    if not items:
        return "(no data)"
    max_count = max(c for _, c in items) or 1
    lines = []
    for name, count in items:
        bar_len = int(count / max_count * width)
        lines.append(f"{name}: {'█' * bar_len} {count}")
    return "\n".join(lines)


def summary(db: LatentDB) -> str:
    layers = _list_layers(db)
    total = 0
    runs = set()
    channels = set()
    for layer in layers:
        events = db.fetch_activations(layer=layer)
        total += len(events)
        for ev in events:
            runs.add(ev.run_id)
            channels.add((layer, ev.channel))
    layer_count = len(layers)
    channel_count = len(channels)

    return (
        f"Total activations: {total}\n"
        f"Runs: {len(runs)} | Layers: {layer_count} | Channels: {channel_count}"
    )


def layer_histogram(db: LatentDB, top: int = 10) -> str:
    counts = []
    for layer in _list_layers(db):
        counts.append((layer, len(db.fetch_activations(layer=layer))))
    counts.sort(key=lambda x: x[1], reverse=True)
    return _ascii_bars(counts[:top])


def head(db: LatentDB, limit: int = 5) -> str:
    rows = []
    for layer in _list_layers(db):
        for batch in db.iter_activations(layer=layer, batch_size=limit):
            for ev in batch:
                rows.append(
                    [
                        ev.run_id,
                        ev.step,
                        ev.layer,
                        ev.channel,
                        ev.prompt_index,
                        ev.token_index,
                        ev.token,
                        ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else None),
                    ]
                )
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break

    headers = ["run_id", "step", "layer", "ch", "p_idx", "t_idx", "token", "value"]
    return _format_table(headers, rows)


def list_layers(db: LatentDB, prefix: str | None = None, top: int = 50) -> str:
    rows = []
    for layer in _list_layers(db, prefix=prefix):
        rows.append((layer, len(db.fetch_activations(layer=layer))))
    rows.sort(key=lambda r: r[1], reverse=True)
    headers = ["layer", "rows"]
    return _format_table(headers, rows[:top], max_width=64)


def layer_stats(db: LatentDB, layer: str) -> str:
    rows = []
    for sb in db.iter_statblocks(layer=layer):
        corrs = sb.top_correlations or []
        top = ", ".join(f"{m}:{rho:+.2f}" for m, rho in corrs[:3]) if corrs else ""
        rows.append(
            [
                sb.layer,
                sb.channel,
                sb.count,
                f"{sb.mean:.3f}",
                f"{sb.std:.3f}",
                f"{sb.min:.3f}",
                f"{sb.max:.3f}",
                top,
                sb.last_updated,
            ]
        )
    headers = ["layer", "ch", "count", "mean", "std", "min", "max", "top_corr", "updated"]
    return _format_table(headers, rows, max_width=32)


def main():
    parser = argparse.ArgumentParser(description="Quick, dependency-free summaries of an Interlatent DB.")
    parser.add_argument("db", help="Path or sqlite:/// / hdf5:/// URI for the DB.")
    parser.add_argument("--limit", type=int, default=5, help="Rows to show in the head table.")
    parser.add_argument("--top", type=int, default=10, help="Number of layers to include in the histogram.")
    parser.add_argument("--list-layers", action="store_true", help="List layers and row counts instead of histogram.")
    parser.add_argument("--layer-prefix", help="Filter layer listing by prefix (e.g., 'latent:' or 'latent_sae:').")
    parser.add_argument("--layer-stats", help="Show stats/correlations for a specific layer (e.g., latent:llm.layer.20).")
    args = parser.parse_args()

    db = LatentDB(args.db)

    print("== Summary ==")
    print(summary(db))
    if args.list_layers:
        print("\n== Layers ==")
        print(list_layers(db, prefix=args.layer_prefix, top=args.top))
    else:
        print("\n== Layer histogram (top {0}) ==".format(args.top))
        print(layer_histogram(db, top=args.top))

    if args.layer_stats:
        print(f"\n== Stats for layer '{args.layer_stats}' ==")
        print(layer_stats(db, args.layer_stats))

    print(f"\n== Head (first {args.limit} rows) ==")
    print(head(db, limit=args.limit))


if __name__ == "__main__":
    main()
