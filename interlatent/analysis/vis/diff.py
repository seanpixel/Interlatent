"""
Compute deltas between latent activations across two slices of data.

Examples:
  python -m interlatent.analysis.vis.diff latents.db --layer-prefix latent: --channels 0 1 --prompt-like-a harmful --prompt-like-b benign
  python -m interlatent.analysis.vis.diff latents_a.db --db-b latents_b.db --layer latent:llm.layer.20 --channels 0 1 2 --token-like bomb
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from interlatent.api import LatentDB


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


def _iter_layers(db: LatentDB, *, layer: str | None, layer_prefix: str | None) -> list[str]:
    if layer:
        return [layer]
    if layer_prefix:
        if hasattr(db._store, "list_layers"):
            layers = db._store.list_layers()
            return [l for l in layers if l.startswith(layer_prefix)]
        raise SystemExit("layer_prefix provided but backend does not support list_layers().")
    raise SystemExit("Must provide --layer or --layer-prefix.")


def _mask_from_meta(meta: dict, prompt_like: str | None, token_like: str | None, n_rows: int):
    mask = np.ones(n_rows, dtype=bool)
    if prompt_like:
        if "prompt_id" in meta and "prompts" in meta:
            prompts = meta["prompts"]
            ids = [i for i, p in enumerate(prompts) if prompt_like in p]
            if ids:
                ids = np.asarray(ids)
                mask &= np.isin(meta["prompt_id"], ids)
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
                ids = np.asarray(ids)
                mask &= np.isin(meta["token_id"], ids)
            else:
                mask &= False
        else:
            tokens = meta.get("token") or []
            mask &= np.array([(t is not None and token_like in t) for t in tokens], dtype=bool)
    return mask


def _aggregated_means(
    db: LatentDB,
    *,
    layer: str | None = None,
    layer_prefix: str | None = None,
    channels: Sequence[int] | None = None,
    prompt_like: str | None = None,
    token_like: str | None = None,
) -> Dict[Tuple[str, int], Tuple[float, int]]:
    """
    Return { (layer, channel): (mean, count) } for rows matching the filters.
    """
    agg: Dict[Tuple[str, int], Tuple[float, int]] = {}
    sums: Dict[Tuple[str, int], float] = {}
    counts: Dict[Tuple[str, int], int] = {}
    for layer_name in _iter_layers(db, layer=layer, layer_prefix=layer_prefix):
        x, meta = db.fetch_vectors(layer=layer_name, limit=None)
        if x.size == 0:
            continue
        mask = _mask_from_meta(meta, prompt_like, token_like, x.shape[0])
        if not mask.any():
            continue
        x_sel = x[mask]
        if channels:
            ch_idx = np.array(list(channels), dtype=int)
            x_sel = x_sel[:, ch_idx]
            means = x_sel.mean(axis=0)
            counts_vec = np.full(len(ch_idx), x_sel.shape[0], dtype=int)
            for i, ch in enumerate(ch_idx):
                key = (layer_name, int(ch))
                sums[key] = float(means[i])
                counts[key] = int(counts_vec[i])
        else:
            means = x_sel.mean(axis=0)
            count = x_sel.shape[0]
            for ch in range(means.shape[0]):
                key = (layer_name, int(ch))
                sums[key] = float(means[ch])
                counts[key] = int(count)

    for key, total in sums.items():
        cnt = counts[key]
        agg[key] = (total, cnt)
    return agg


def latent_diff(
    db_a: LatentDB,
    db_b: LatentDB,
    *,
    layer: str | None = None,
    layer_prefix: str | None = None,
    channels: Sequence[int] | None = None,
    prompt_like_a: str | None = None,
    prompt_like_b: str | None = None,
    token_like_a: str | None = None,
    token_like_b: str | None = None,
    top: int = 20,
) -> str:
    """
    Compute per-latent mean activation deltas between two slices (A vs B).
    """
    agg_a = _aggregated_means(
        db_a,
        layer=layer,
        layer_prefix=layer_prefix,
        channels=channels,
        prompt_like=prompt_like_a,
        token_like=token_like_a,
    )
    agg_b = _aggregated_means(
        db_b,
        layer=layer,
        layer_prefix=layer_prefix,
        channels=channels,
        prompt_like=prompt_like_b,
        token_like=token_like_b,
    )

    all_keys = set(agg_a.keys()) | set(agg_b.keys())
    rows = []
    for layer_name, ch in sorted(all_keys):
        mean_a, count_a = agg_a.get((layer_name, ch), (0.0, 0))
        mean_b, count_b = agg_b.get((layer_name, ch), (0.0, 0))
        delta = mean_b - mean_a
        rows.append(
            [
                layer_name,
                ch,
                f"{mean_a:.4f}",
                f"{mean_b:.4f}",
                f"{delta:+.4f}",
                count_a,
                count_b,
            ]
        )

    rows.sort(key=lambda r: abs(float(r[4])), reverse=True)
    rows = rows[:top]

    headers = ["layer", "ch", "mean_A", "mean_B", "B-A", "n_A", "n_B"]
    if not rows:
        return _format_table(headers, [["(no rows)", "", "", "", "", "", ""]])
    return _format_table(headers, rows, max_width=48)


def main():
    p = argparse.ArgumentParser(description="Diff latent activations across two slices (A vs B).")
    p.add_argument("db_a", help="SQLite path or sqlite:/// URI for slice A.")
    p.add_argument("--db-b", help="Optional SQLite path/URI for slice B (defaults to A).")
    p.add_argument("--layer", help="Exact layer to filter (e.g., latent:llm.layer.20).")
    p.add_argument("--layer-prefix", help="Layer prefix (e.g., latent: or latent_sae:).")
    p.add_argument("--channels", type=int, nargs="+", help="Channel indices to include.")
    p.add_argument("--prompt-like-a", help="Substring match on prompt text for slice A.")
    p.add_argument("--prompt-like-b", help="Substring match on prompt text for slice B.")
    p.add_argument("--token-like-a", help="Substring match on token text for slice A.")
    p.add_argument("--token-like-b", help="Substring match on token text for slice B.")
    p.add_argument("--top", type=int, default=20, help="Rows to display sorted by |B-A|.")
    args = p.parse_args()

    db_a = LatentDB(args.db_a)
    db_b = LatentDB(args.db_b) if args.db_b else db_a

    table = latent_diff(
        db_a,
        db_b,
        layer=args.layer,
        layer_prefix=args.layer_prefix,
        channels=args.channels,
        prompt_like_a=args.prompt_like_a,
        prompt_like_b=args.prompt_like_b,
        token_like_a=args.token_like_a,
        token_like_b=args.token_like_b,
        top=args.top,
    )
    print(table)


if __name__ == "__main__":
    main()
