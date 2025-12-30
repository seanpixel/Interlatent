"""
Targeted search over activations for quick “which latents fire on which tokens?”

Examples:
  python -m interlatent.analysis.vis.search latents.db --layer-prefix latent: --token-like sky --top 20
  python -m interlatent.analysis.vis.search latents.db --layer llm.layer.20 --prompt-like hello --channel 0 --top 10
"""
from __future__ import annotations

import argparse
from typing import Sequence, Tuple

from interlatent.api import LatentDB

def _iter_layers(db: LatentDB, *, layer: str | None, layer_prefix: str | None) -> list[str]:
    if layer:
        return [layer]
    if layer_prefix:
        if hasattr(db._store, "list_layers"):
            layers = db._store.list_layers()
            return [l for l in layers if l.startswith(layer_prefix)]
        raise SystemExit("layer_prefix provided but backend does not support list_layers().")
    raise SystemExit("Must provide --layer or --layer-prefix.")


def _format_table(headers: Sequence[str], rows: Sequence[Sequence], max_width: int = 24) -> str:
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


def search(
    db: LatentDB,
    *,
    layer: str | None = None,
    layer_prefix: str | None = None,
    prompt_like: str | None = None,
    token_like: str | None = None,
    channel: int | None = None,
    top: int = 50,
    min_abs: float | None = None,
) -> str:
    rows = []
    for layer_name in _iter_layers(db, layer=layer, layer_prefix=layer_prefix):
        events = db.fetch_activations(layer=layer_name, limit=None)
        for ev in events:
            if channel is not None and ev.channel != channel:
                continue
            if prompt_like and (ev.prompt is None or prompt_like not in ev.prompt):
                continue
            if token_like and (ev.token is None or token_like not in ev.token):
                continue
            val = ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0)
        if min_abs is not None and abs(val) < min_abs:
            continue
        rows.append(
            [
                ev.run_id,  # run_id
                ev.layer,  # layer
                ev.channel,  # channel
                ev.prompt_index,  # prompt_idx
                ev.token_index,  # token_idx
                ev.token,  # token
                f"{val:.4f}",
                f"{abs(val):.4f}",
                ev.prompt,  # prompt text
            ]
        )

    # sort by absolute value descending to surface strong activations
    rows.sort(key=lambda row: float(row[7]), reverse=True)
    rows = rows[:top]

    headers = ["run_id", "layer", "ch", "p_idx", "t_idx", "token", "value", "|value|", "prompt"]
    return _format_table(headers, rows, max_width=32)


def main():
    p = argparse.ArgumentParser(description="Search activations by layer/prompt/token filters.")
    p.add_argument("db", help="SQLite path or sqlite:/// URI")
    p.add_argument("--layer", help="Exact layer name to filter")
    p.add_argument("--layer-prefix", help="Layer prefix (e.g., 'latent:' or 'latent_sae:')")
    p.add_argument("--prompt-like", help="Substring match on prompt text")
    p.add_argument("--token-like", help="Substring match on token text")
    p.add_argument("--channel", type=int, help="Channel index filter")
    p.add_argument("--top", type=int, default=50, help="Rows to show after sorting by |value|")
    p.add_argument("--min-abs", type=float, help="Only show activations with |value| >= threshold")
    args = p.parse_args()

    db = LatentDB(args.db)
    table = search(
        db,
        layer=args.layer,
        layer_prefix=args.layer_prefix,
        prompt_like=args.prompt_like,
        token_like=args.token_like,
        channel=args.channel,
        top=args.top,
        min_abs=args.min_abs,
    )
    print(table)


if __name__ == "__main__":
    main()
