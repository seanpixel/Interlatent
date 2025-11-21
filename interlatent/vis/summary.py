"""
Lightweight CLI utilities to inspect an Interlatent SQLite database without
pulling data into pandas. Designed for quick terminal summaries.

Usage:
  python -m interlatent.vis.summary sqlite:///latents_llm_local.db
  python -m interlatent.vis.summary latents.db --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import List, Sequence, Tuple


def _open_db(uri: str) -> sqlite3.Connection:
    # Accept sqlite:///path or bare path.
    if uri.startswith("sqlite:///"):
        path = uri[len("sqlite:///") :]
    else:
        path = uri
    if not os.path.exists(path):
        raise SystemExit(f"Database not found: {path}")
    conn = sqlite3.connect(path)
    return conn


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


def summary(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM activations")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT run_id) FROM activations")
    runs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT layer) FROM activations")
    layers = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT channel) FROM activations")
    channels = cur.fetchone()[0]

    return (
        f"Total activations: {total}\n"
        f"Runs: {runs} | Layers: {layers} | Channels: {channels}"
    )


def layer_histogram(conn: sqlite3.Connection, top: int = 10) -> str:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT layer, COUNT(*) as c
        FROM activations
        GROUP BY layer
        ORDER BY c DESC
        LIMIT ?
        """,
        (top,),
    )
    rows = cur.fetchall()
    return _ascii_bars([(r[0], r[1]) for r in rows])


def head(conn: sqlite3.Connection, limit: int = 5) -> str:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_id, step, layer, channel, prompt_index, token_index, token, tensor, timestamp
        FROM activations
        ORDER BY timestamp, step
        LIMIT ?
        """,
        (limit,),
    )
    rows = []
    for r in cur.fetchall():
        tensor = json.loads(r[7]) if r[7] else []
        val = tensor[0] if tensor else None
        rows.append(
            [
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                val,
                r[8],
            ]
        )

    headers = ["run_id", "step", "layer", "ch", "p_idx", "t_idx", "token", "value", "timestamp"]
    return _format_table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description="Quick, dependency-free summaries of an Interlatent SQLite DB.")
    parser.add_argument("db", help="Path or sqlite:/// URI for the DB.")
    parser.add_argument("--limit", type=int, default=5, help="Rows to show in the head table.")
    parser.add_argument("--top", type=int, default=10, help="Number of layers to include in the histogram.")
    args = parser.parse_args()

    conn = _open_db(args.db)

    print("== Summary ==")
    print(summary(conn))
    print("\n== Layer histogram (top {0}) ==".format(args.top))
    print(layer_histogram(conn, top=args.top))
    print(f"\n== Head (first {args.limit} rows) ==")
    print(head(conn, limit=args.limit))


if __name__ == "__main__":
    main()
