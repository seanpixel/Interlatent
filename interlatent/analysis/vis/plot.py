"""
Plot activation traces for a given (layer, channel, prompt) from a LatentDB file.

Usage (CLI):
  python -m interlatent.analysis.vis.plot latents_llm.db --layer llm.layer.-1 --channel 0 --prompt-index 0 --output out.png

Programmatic:
  from interlatent.analysis.vis.plot import plot_activation
  plot_activation("latents_llm.db", layer="llm.layer.-1", channel=0, prompt_index=0)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt

from interlatent.api import LatentDB
import numpy as np

@dataclass
class ActivationRow:
    token_index: int
    token: str | None
    value: float
    prompt_index: int | None
    prompt: str | None
    run_id: str | None = None


def _pretty_token(tok: str | None) -> str:
    """Normalize HF tokens for nicer display (strip common space markers)."""
    if tok is None:
        return ""
    out = tok.replace("Ġ", " ").replace("▁", " ")
    if out.strip() == "":
        return "␠"
    return out


def fetch_activations(
    db: str,
    *,
    layer: str,
    channel: int,
    prompt_index: int | None = None,
    prompt_like: str | None = None,
    limit_prompts: int | None = None,
) -> list[ActivationRow]:
    """
    Pull activations for the requested slice. Filters by prompt_index if given,
    else selects prompts matching prompt_like (LIKE '%...%'). Results ordered by
    (prompt_index, token_index).
    """
    dbi = LatentDB(db)
    x, meta = dbi.fetch_vectors(layer=layer)
    if x.size == 0:
        return []

    n = x.shape[0]
    mask = np.ones(n, dtype=bool)
    if prompt_index is not None:
        mask &= meta.get("prompt_index") == prompt_index
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

    if not mask.any():
        return []
    idx = np.where(mask)[0]
    vals = x[idx, channel]

    if "prompt_id" in meta and "prompts" in meta:
        prompt_vals = [meta["prompts"][int(meta["prompt_id"][i])] for i in idx]
    else:
        prompt_vals = [meta.get("prompt")[i] for i in idx]
    if "token_id" in meta and "tokens" in meta:
        token_vals = [meta["tokens"][int(meta["token_id"][i])] for i in idx]
    else:
        token_vals = [meta.get("token")[i] for i in idx]
    prompt_index_vals = meta.get("prompt_index")[idx] if "prompt_index" in meta else [None] * len(idx)
    token_index_vals = meta.get("token_index")[idx] if "token_index" in meta else [None] * len(idx)

    rows = []
    for i in range(len(idx)):
        rows.append(
            ActivationRow(
                token_index=int(token_index_vals[i]) if token_index_vals[i] is not None else 0,
                token=token_vals[i],
                value=float(vals[i]),
                prompt_index=int(prompt_index_vals[i]) if prompt_index_vals[i] is not None else None,
                prompt=prompt_vals[i],
                run_id=None,
            )
        )

    if limit_prompts is not None:
        seen = []
        for row in rows:
            if row.prompt_index not in seen:
                seen.append(row.prompt_index)
            if len(seen) >= limit_prompts:
                break
        rows = [r for r in rows if r.prompt_index in set(seen)]
    rows.sort(key=lambda r: (r.prompt_index or 0, r.token_index))
    return rows


def plot_activation(
    db: str,
    *,
    layer: str,
    channel: int,
    prompt_index: int | None = None,
    prompt_like: str | None = None,
    output: str | None = None,
):
    """
    Render an activation trace (scatter + line) over token positions.

    If both prompt_index and prompt_like are None, the first prompt in the DB
    is used.
    """
    rows = fetch_activations(
        db,
        layer=layer,
        channel=channel,
        prompt_index=prompt_index,
        prompt_like=prompt_like,
    )
    if not rows:
        raise ValueError("No activations found for the requested slice.")

    # Group by prompt_index so we can plot multiple prompts if they match.
    grouped: dict[int | None, list[ActivationRow]] = {}
    for r in rows:
        grouped.setdefault(r.prompt_index, []).append(r)

    # Sort tokens per prompt.
    for lst in grouped.values():
        lst.sort(key=lambda r: r.token_index)

    fig, ax = plt.subplots(figsize=(12, 5))

    # build readable x labels
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (p_idx, lst) in enumerate(sorted(grouped.items(), key=lambda kv: kv[0] or 0)):
        xs = [r.token_index for r in lst]
        ys = [r.value for r in lst]
        tokens = [r.token or "" for r in lst]
        text_labels = [f"{_pretty_token(tok)}\n(p{p_idx},t{ti})" for tok, ti in zip(tokens, xs)]
        color = colors[idx % len(colors)]
        label = f"prompt {p_idx}" if p_idx is not None else "prompt"
        ax.plot(xs, ys, marker="o", color=color, label=label)
        for x, y, text in zip(xs, ys, text_labels):
            tok = text
            if len(tok) > 16:
                tok = tok[:15] + "…"
            ax.text(
                x,
                y,
                tok,
                fontsize=7,
                rotation=30,
                ha="right",
                va="bottom",
                color=color,
                alpha=0.8,
            )

    prompt_text = rows[0].prompt or ""
    if prompt_like:
        title_prompt = f"prompt like '{prompt_like}'"
    elif prompt_index is not None:
        title_prompt = f"prompt index {prompt_index}"
    else:
        title_prompt = "prompt"

    ax.set_title(f"Activations: {layer} / ch {channel} / {title_prompt}\n{prompt_text}", fontsize=10)
    ax.set_xlabel("token index")
    ax.set_ylabel("activation value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = output or f"activation_{layer.replace('.', '_')}_ch{channel}.png"
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_latent_across_prompts(
    db: str,
    *,
    layer: str,
    channel: int,
    max_label_len: int = 32,
    output: str | None = None,
):
    """
    Plot a single latent (layer/channel) across *all* prompts.

    X-axis groups tokens by prompt (one column per prompt); labels show the first
    few words of each prompt. Y-axis is activation value per token.
    """
    rows = fetch_activations(db, layer=layer, channel=channel)
    if not rows:
        raise ValueError("No activations found for the requested slice.")

    grouped: dict[int | None, list[ActivationRow]] = {}
    for r in rows:
        grouped.setdefault(r.prompt_index, []).append(r)

    # Stable order by prompt_index (None last)
    ordered = sorted(grouped.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))

    xs_all: list[float] = []
    ys_all: list[float] = []
    colors: list[str] = []
    tokens: list[str] = []
    xticks: list[float] = []
    xtick_labels: list[str] = []

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (p_idx, lst) in enumerate(ordered):
        lst.sort(key=lambda r: r.token_index)
        prompt_tokens_x = []
        prompt_tokens_y = []
        prompt_tokens_str = []
        for r in lst:
            x = float(idx)  # same column per prompt
            prompt_tokens_x.append(x)
            prompt_tokens_y.append(r.value)
            prompt_tokens_str.append(_pretty_token(r.token))
        if not prompt_tokens_x:
            continue

        label = (lst[0].prompt or f"prompt {p_idx}") if lst else f"prompt {p_idx}"
        label = " ".join(label.split()[:6])
        if len(label) > max_label_len:
            label = label[: max_label_len - 1] + "…"

        xticks.append(float(idx))
        xtick_labels.append(label)

        color = palette[idx % len(palette)]
        xs_all.extend(prompt_tokens_x)
        ys_all.extend(prompt_tokens_y)
        colors.extend([color] * len(prompt_tokens_x))
        tokens.extend(prompt_tokens_str)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(xs_all, ys_all, c=colors, s=30, alpha=0.85)

    for x, y, tok in zip(xs_all, ys_all, tokens):
        if len(tok) > 12:
            tok = tok[:11] + "…"
        ax.text(x + 0.05, y, tok, fontsize=7, rotation=0, ha="left", va="bottom", alpha=0.7)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=25, ha="right")
    ax.set_xlabel("prompts (grouped tokens)")
    ax.set_ylabel("activation value")
    ax.set_title(f"Latent across prompts: {layer} / ch {channel}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = output or f"activation_allprompts_{layer.replace('.', '_')}_ch{channel}.png"
    fig.savefig(out_path, dpi=150)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot activations for a given layer/channel/prompt.")
    parser.add_argument("db", help="LatentDB SQLite path or sqlite:/// URI.")
    parser.add_argument("--layer", required=True, help="Layer name, e.g. llm.layer.-1")
    parser.add_argument("--channel", type=int, required=True, help="Channel index.")
    parser.add_argument("--prompt-index", type=int, help="Prompt index to plot.")
    parser.add_argument("--prompt-like", help="Substring to match prompt text (SQL LIKE).")
    parser.add_argument("--all-prompts", action="store_true", help="Plot this latent across all prompts.")
    parser.add_argument("--output", help="Output PNG path.")
    args = parser.parse_args()

    if args.all_prompts:
        out = plot_latent_across_prompts(
            args.db,
            layer=args.layer,
            channel=args.channel,
            output=args.output,
        )
    else:
        out = plot_activation(
            args.db,
            layer=args.layer,
            channel=args.channel,
            prompt_index=args.prompt_index,
            prompt_like=args.prompt_like,
            output=args.output,
        )
    print(f"Wrote plot to {out}")


if __name__ == "__main__":
    main()
