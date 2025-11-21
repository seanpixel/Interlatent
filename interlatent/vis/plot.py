"""
Plot activation traces for a given (layer, channel, prompt) from a LatentDB SQLite file.

This is a lightweight, pyplot-based visualizer geared toward quick interpretability
passes on small runs. It reads directly from the `activations` table.

Usage (CLI):
  python -m interlatent.vis.plot latents_llm.db --layer llm.layer.-1 --channel 0 --prompt-index 0 --output out.png

Programmatic:
  from interlatent.vis.plot import plot_activation
  plot_activation("latents_llm.db", layer="llm.layer.-1", channel=0, prompt_index=0)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt


@dataclass
class ActivationRow:
    token_index: int
    token: str | None
    value: float
    prompt_index: int | None
    prompt: str | None


def _open_db(uri: str) -> sqlite3.Connection:
    # Accept sqlite:///path or bare path.
    if uri.startswith("sqlite:///"):
        path = uri[len("sqlite:///") :]
    else:
        path = uri
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database not found: {path}")
    conn = sqlite3.connect(path)
    return conn


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
    conn = _open_db(db)
    cur = conn.cursor()

    sql = [
        "SELECT prompt_index, prompt, token_index, token, tensor",
        "FROM activations",
        "WHERE layer=? AND channel=?",
    ]
    params: list = [layer, channel]

    if prompt_index is not None:
        sql.append("AND prompt_index = ?")
        params.append(prompt_index)
    if prompt_like:
        sql.append("AND prompt LIKE ?")
        params.append(f\"%{prompt_like}%\")

    sql.append("ORDER BY prompt_index, token_index")
    if limit_prompts is not None:
        # limit prompts by grouping prompt_index; simple approach via subquery.
        sql = [
            "SELECT prompt_index, prompt, token_index, token, tensor FROM (",
            *sql,
            ") WHERE prompt_index IN (SELECT DISTINCT prompt_index FROM activations WHERE layer=? AND channel=? LIMIT ?)",
        ]
        params.extend([layer, channel, limit_prompts])

    cur.execute(" ".join(sql), params)
    rows = []
    for p_idx, prompt, t_idx, token, tensor_json in cur.fetchall():
        tensor = json.loads(tensor_json) if tensor_json else []
        val = tensor[0] if tensor else 0.0
        rows.append(
            ActivationRow(
                token_index=t_idx if t_idx is not None else 0,
                token=token,
                value=float(val),
                prompt_index=p_idx,
                prompt=prompt,
            )
        )
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

    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (p_idx, lst) in enumerate(sorted(grouped.items(), key=lambda kv: kv[0] or 0)):
        xs = [r.token_index for r in lst]
        ys = [r.value for r in lst]
        tokens = [r.token or "" for r in lst]
        color = colors[idx % len(colors)]
        label = f"prompt {p_idx}" if p_idx is not None else "prompt"
        ax.plot(xs, ys, marker="o", color=color, label=label)
        for x, y, tok in zip(xs, ys, tokens):
            if len(tok) > 12:
                tok = tok[:11] + "…"
            ax.text(x, y, tok, fontsize=7, rotation=45, ha="right", va="bottom", color=color, alpha=0.7)

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


def main():
    parser = argparse.ArgumentParser(description="Plot activations for a given layer/channel/prompt.")
    parser.add_argument("db", help="LatentDB SQLite path or sqlite:/// URI.")
    parser.add_argument("--layer", required=True, help="Layer name, e.g. llm.layer.-1")
    parser.add_argument("--channel", type=int, required=True, help="Channel index.")
    parser.add_argument("--prompt-index", type=int, help="Prompt index to plot.")
    parser.add_argument("--prompt-like", help="Substring to match prompt text (SQL LIKE).")
    parser.add_argument("--output", help="Output PNG path.")
    args = parser.parse_args()

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
