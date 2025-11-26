"""
Targeted search over activations for quick “which latents fire on which tokens?”

Examples:
  python -m interlatent.vis.search latents.db --layer-prefix latent: --token-like sky --top 20
  python -m interlatent.vis.search latents.db --layer llm.layer.20 --prompt-like hello --channel 0 --top 10
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Sequence, Tuple


def _open_db(uri: str) -> sqlite3.Connection:
    if uri.startswith("sqlite:///"):
        path = uri[len("sqlite:///") :]
    else:
        path = uri
    if not os.path.exists(path):
        raise SystemExit(f"Database not found: {path}")
    conn = sqlite3.connect(path)
    return conn


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
    conn: sqlite3.Connection,
    *,
    layer: str | None = None,
    layer_prefix: str | None = None,
    prompt_like: str | None = None,
    token_like: str | None = None,
    channel: int | None = None,
    top: int = 50,
    min_abs: float | None = None,
) -> str:
    sql = [
        "SELECT run_id, layer, channel, prompt_index, token_index, token, prompt, tensor, step",
        "FROM activations",
        "WHERE 1=1",
    ]
    params: list = []

    if layer:
        sql.append("AND layer = ?")
        params.append(layer)
    if layer_prefix:
        sql.append("AND layer LIKE ?")
        params.append(f"{layer_prefix}%")
    if prompt_like:
        sql.append("AND prompt LIKE ?")
        params.append(f"%{prompt_like}%")
    if token_like:
        sql.append("AND token LIKE ?")
        params.append(f"%{token_like}%")
    if channel is not None:
        sql.append("AND channel = ?")
        params.append(channel)

    sql.append("ORDER BY step")

    cur = conn.cursor()
    cur.execute(" ".join(sql), params)

    rows = []
    for r in cur.fetchall():
        tensor = json.loads(r[7] or "[]")
        val = tensor[0] if tensor else 0.0
        if min_abs is not None and abs(val) < min_abs:
            continue
        rows.append(
            [
                r[0],  # run_id
                r[1],  # layer
                r[2],  # channel
                r[3],  # prompt_idx
                r[4],  # token_idx
                r[5],  # token
                f"{val:.4f}",
                f"{abs(val):.4f}",
                r[6],  # prompt text
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

    conn = _open_db(args.db)
    table = search(
        conn,
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
