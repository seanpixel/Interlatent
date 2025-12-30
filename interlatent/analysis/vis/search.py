"""
Targeted search over activations for quick “which latents fire on which tokens?”

Examples:
  python -m interlatent.analysis.vis.search latents.db --layer-prefix latent: --token-like sky --top 20
  python -m interlatent.analysis.vis.search latents.db --layer llm.layer.20 --prompt-like hello --channel 0 --top 10
"""
from __future__ import annotations

import argparse
from typing import Sequence, Tuple

import numpy as np

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
        x, meta = db.fetch_vectors(layer=layer_name, limit=None)
        if x.size == 0:
            continue
        n = x.shape[0]
        mask = np.ones(n, dtype=bool)
        if prompt_like:
            if "prompt_id" in meta and "prompts" in meta:
                prompts = meta["prompts"]
                ids = [i for i, p in enumerate(prompts) if prompt_like in p]
                if ids:
                    mask &= np.isin(meta["prompt_id"], np.asarray(ids))
                else:
                    mask &= False
            else:
                prompts = meta.get("prompt") or []
                mask &= np.array([(p is not None and prompt_like in p) for p in prompts], dtype=bool)
        if token_like:
            if "token_id" in meta and "tokens" in meta:
                tokens = meta["tokens"]
                ids = [i for i, t in enumerate(tokens) if token_like in t]
                if ids:
                    mask &= np.isin(meta["token_id"], np.asarray(ids))
                else:
                    mask &= False
            else:
                tokens = meta.get("token") or []
                mask &= np.array([(t is not None and token_like in t) for t in tokens], dtype=bool)
        if not mask.any():
            continue

        row_idx = np.where(mask)[0]
        if channel is not None:
            vals = x[row_idx, channel]
            chs = np.full(len(row_idx), channel)
        else:
            vals = x[row_idx]
            chs = np.repeat(np.arange(x.shape[1]), len(row_idx))
            vals = vals.T.reshape(-1)
            row_idx = np.tile(row_idx, x.shape[1])

        if "prompt_id" in meta and "prompts" in meta:
            prompt_vals = [meta["prompts"][int(meta["prompt_id"][i])] for i in row_idx]
        else:
            prompt_vals = [meta.get("prompt")[i] for i in row_idx]
        if "token_id" in meta and "tokens" in meta:
            token_vals = [meta["tokens"][int(meta["token_id"][i])] for i in row_idx]
        else:
            token_vals = [meta.get("token")[i] for i in row_idx]
        if "run_id_id" in meta and "run_ids" in meta:
            run_vals = [meta["run_ids"][int(meta["run_id_id"][i])] for i in row_idx]
        else:
            run_vals = [meta.get("run_id")[i] for i in row_idx]
        prompt_index_vals = meta.get("prompt_index")[row_idx] if "prompt_index" in meta else [None] * len(row_idx)
        token_index_vals = meta.get("token_index")[row_idx] if "token_index" in meta else [None] * len(row_idx)

        for i in range(len(row_idx)):
            val = float(vals[i])
            if min_abs is not None and abs(val) < min_abs:
                continue
            rows.append(
                [
                    run_vals[i],
                    layer_name,
                    int(chs[i]),
                    int(prompt_index_vals[i]) if prompt_index_vals[i] is not None else None,
                    int(token_index_vals[i]) if token_index_vals[i] is not None else None,
                    token_vals[i],
                    f"{val:.4f}",
                    f"{abs(val):.4f}",
                    prompt_vals[i],
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
