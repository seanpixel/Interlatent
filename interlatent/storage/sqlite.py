"""interlatent.storage.sqlite

A lightweight, zero‑dependency SQLite implementation of
:class:`interlatent.storage.base.StorageBackend`.

SQLite is perfect for single‑machine research workflows: it ships with
Python, handles moderate write QPS, and offers FTS5 for full‑text search
(without extra binaries).  This driver keeps schema creation minimal but
future‑proof—migrations can append columns without breaking callers.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import time
import os
from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ..schema import ActivationEvent, Artifact, Explanation, StatBlock
from .base import StorageBackend

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

_ISO_DATE_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _dict_factory(cursor, row):
    # Return rows as dicts for convenience.
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _ensure_column(cur: sqlite3.Cursor, table: str, column_def: str) -> None:
    """Idempotently add a column to *table* if it is missing."""
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_def};")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise


# ---------------------------------------------------------------------------
# SQLiteBackend --------------------------------------------------------------
# ---------------------------------------------------------------------------


class SQLiteBackend(StorageBackend):
    """SQLite driver—stores everything in a single .db file."""

    def __init__(self, uri: str):
        super().__init__(uri)
        if uri.startswith("sqlite:///"):
            path = uri[len("sqlite:///") :]
        elif uri.startswith("file:///"):
            path = uri[len("file:///") :]
        else:
            # fallback: treat uri as direct path
            path = uri
        self._path = pathlib.Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = _dict_factory
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._hidden_dim = int(os.environ.get("LATENTDB_MAX_CHANNELS", "0") or 0) or None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema creation ---------------------------------------------------
    # ------------------------------------------------------------------

    def _ensure_schema(self):
        cur = self._conn.cursor()
        # activations table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS activations (
              run_id     TEXT,
              step       INTEGER,
              layer      TEXT,
              prompt     TEXT,
              prompt_index INTEGER,
              token_index  INTEGER,
              token      TEXT,
              tensor     TEXT,
              context    TEXT,
              PRIMARY KEY (run_id, step, layer)
            ) WITHOUT ROWID;
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_activations_layer_step
            ON activations(layer, step);
            """
        )
        # Backward‑compatible add of new columns for prompt/token metadata.
        _ensure_column(cur, "activations", "prompt TEXT")
        _ensure_column(cur, "activations", "prompt_index INTEGER")
        _ensure_column(cur, "activations", "token_index INTEGER")
        _ensure_column(cur, "activations", "token TEXT")
        # metric sums
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metric_sums (
            metric      TEXT,
            layer       TEXT,
            channel     INTEGER,
            count       INTEGER DEFAULT 0,
            sum_m       REAL    DEFAULT 0,
            sum_m2      REAL    DEFAULT 0,
            sum_xm      REAL    DEFAULT 0,
            PRIMARY KEY (metric, layer, channel)
            ) WITHOUT ROWID;
            """
        )
        # stats table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stats (
            layer        TEXT,
            channel      INTEGER,

            -- running tallies ----------
            count        INTEGER      DEFAULT 0,
            sum_x        REAL         DEFAULT 0,   -- Σ x
            sum_x2       REAL         DEFAULT 0,   -- Σ x²
            sum_m        REAL         DEFAULT 0,   -- Σ m
            sum_m2       REAL         DEFAULT 0,   -- Σ m²
            sum_xm       REAL         DEFAULT 0,   -- Σ x m

            -- derived moments ----------
            mean         REAL,
            std          REAL,
            min          REAL,
            max          REAL,

            correlations TEXT,
            last_updated TEXT,
            PRIMARY KEY (layer, channel)
            ) WITHOUT ROWID;
            """
        )
        # explanations table (multiple versions per channel)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS explanations (
              layer      TEXT,
              channel    INTEGER,
              version    INTEGER,
              text       TEXT,
              source     TEXT,
              created_at TEXT,
              PRIMARY KEY (layer, channel, version)
            );
            """
        )
        # artifacts table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
              artifact_id TEXT PRIMARY KEY,
              kind        TEXT,
              path        TEXT,
              meta        TEXT,
              created_at  TEXT
            );
            """
        )
        # FTS5 virtual table for explanation text search (if available)
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS explanations_fts USING fts5(
                  text, layer UNINDEXED, channel UNINDEXED, version UNINDEXED,
                  content='explanations', content_rowid='rowid');
                """
            )
            # populate if fresh DB
            cur.execute(
                "INSERT INTO explanations_fts(rowid, text) SELECT rowid, text FROM explanations WHERE rowid NOT IN (SELECT rowid FROM explanations_fts);"
            )
        except sqlite3.OperationalError:
            # SQLite built without FTS5—fallback later with LIKE search
            pass

        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        self._conn.commit()
        self._conn.close()

    # ------------------------------------------------------------------
    # Write methods ------------------------------------------------------
    # ------------------------------------------------------------------

    def write_event(self, ev: ActivationEvent) -> None:
        self.write_events([ev])

    def write_events(self, events: Sequence[ActivationEvent]) -> None:
        cur = self._conn.cursor()
        metric_rows = []
        stat_rows = []
        batches: dict[tuple[str, int, str], dict] = {}

        for ev in events:
            metrics: dict[str, float] = ev.context.get("metrics", {})
            if metrics:
                sum_x = ev.value_sum or sum(ev.tensor)
                sum_x2 = ev.value_sq_sum or sum(v * v for v in ev.tensor)
                for name, m in metrics.items():
                    m = float(m)
                    metric_rows.append(
                        (
                            name,
                            ev.layer,
                            ev.channel,
                            m,            # Σ m
                            m * m,        # Σ m²
                            sum_x * m,    # Σ x m
                        )
                    )
                    stat_rows.append((ev.layer, ev.channel, sum_x, sum_x2))

            key = (ev.run_id, ev.step, ev.layer)
            batch = batches.setdefault(
                key,
                {
                    "prompt": ev.prompt,
                    "prompt_index": ev.prompt_index,
                    "token_index": ev.token_index,
                    "token": ev.token,
                    "context": ev.context,
                    "vec": {},
                },
            )
            batch["vec"][ev.channel] = float(ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0))
            if batch["prompt"] is None:
                batch["prompt"] = ev.prompt
            if batch["prompt_index"] is None:
                batch["prompt_index"] = ev.prompt_index
            if batch["token_index"] is None:
                batch["token_index"] = ev.token_index
            if batch["token"] is None:
                batch["token"] = ev.token
            if not batch["context"]:
                batch["context"] = ev.context

        if metric_rows:
            cur.executemany(
                """
                INSERT INTO metric_sums (metric, layer, channel,
                                        count, sum_m, sum_m2, sum_xm)
                VALUES (?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(metric, layer, channel) DO UPDATE SET
                count  = count  + 1,
                sum_m  = sum_m  + EXCLUDED.sum_m,
                sum_m2 = sum_m2 + EXCLUDED.sum_m2,
                sum_xm = sum_xm + EXCLUDED.sum_xm
                """,
                metric_rows,
            )
            cur.executemany(
                """
                INSERT INTO stats (layer, channel, count, sum_x, sum_x2)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(layer, channel) DO UPDATE SET
                count  = count  + 1,
                sum_x  = sum_x  + EXCLUDED.sum_x,
                sum_x2 = sum_x2 + EXCLUDED.sum_x2
                """,
                stat_rows,
            )

        activation_rows = []
        for (run_id, step, layer), batch in batches.items():
            vec = batch["vec"]
            if vec:
                if self._hidden_dim is not None:
                    tensor = [0.0] * self._hidden_dim
                else:
                    max_ch = max(vec.keys())
                    tensor = [0.0] * (max_ch + 1)
                for ch, val in vec.items():
                    if self._hidden_dim is not None and ch >= self._hidden_dim:
                        raise ValueError(f"channel {ch} exceeds hidden_dim {self._hidden_dim}")
                    tensor[int(ch)] = float(val)
            else:
                tensor = [0.0] * (self._hidden_dim or 0)
            activation_rows.append(
                (
                    run_id,
                    step,
                    layer,
                    batch["prompt"],
                    batch["prompt_index"],
                    batch["token_index"],
                    batch["token"],
                    json.dumps(tensor),
                    json.dumps(batch["context"]),
                )
            )

        cur.executemany(
            """
            INSERT OR REPLACE INTO activations
            (run_id, step, layer, prompt, prompt_index, token_index, token, tensor, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            activation_rows,
        )
        self._conn.commit()

    def iter_activations(self, layer: str, batch_size: int = 1000):
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT run_id, step, layer, prompt, prompt_index, token_index, token, tensor, context
            FROM activations
            WHERE layer = ?
            ORDER BY step
            """,
            (layer,),
        )
        loads = json.loads
        ctx_cache: dict[str, dict] = {}
        ActivationEventLocal = ActivationEvent
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            events: list[ActivationEvent] = []
            append = events.append
            for r in rows:
                tensor = loads(r["tensor"] or "[]")
                ctx_text = r["context"] or ""
                if ctx_text:
                    ctx = ctx_cache.get(ctx_text)
                    if ctx is None:
                        ctx = loads(ctx_text)
                        ctx_cache[ctx_text] = ctx
                else:
                    ctx = {}
                for ch, val in enumerate(tensor):
                    append(
                        ActivationEventLocal(
                            run_id=r["run_id"],
                            step=r["step"],
                            layer=r["layer"],
                            channel=ch,
                            prompt=r.get("prompt"),
                            prompt_index=r.get("prompt_index"),
                            token_index=r.get("token_index"),
                            token=r.get("token"),
                            tensor=[float(val)],
                            context=ctx,
                            value_sum=float(val),
                            value_sq_sum=float(val * val),
                        )
                    )
            yield events

    def write_statblock(self, sb: StatBlock) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO stats (layer, channel, count, mean, std, min, max, correlations, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(layer, channel) DO UPDATE SET
              count=excluded.count,
              mean=excluded.mean,
              std =excluded.std,
              min =excluded.min,
              max =excluded.max,
              correlations=excluded.correlations,
              last_updated=excluded.last_updated;
            """,
            (
                sb.layer,
                sb.channel,
                sb.count,
                sb.mean,
                sb.std,
                sb.min,
                sb.max,
                json.dumps(sb.top_correlations),
                sb.last_updated,
            ),
        )
        self._conn.commit()

    def write_explanation(self, ex: Explanation) -> None:
        cur = self._conn.cursor()
        # get next version if exists
        cur.execute(
            "SELECT COALESCE(MAX(version), 0) FROM explanations WHERE layer=? AND channel=?",
            (ex.layer, ex.channel),
        )
        next_ver = cur.fetchone()["COALESCE(MAX(version), 0)"] + 1 if ex.version == 1 else ex.version
        cur.execute(
            """
            INSERT INTO explanations (layer, channel, version, text, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ex.layer,
                ex.channel,
                next_ver,
                ex.text,
                ex.source,
                ex.created_at,
            ),
        )
        # maintain FTS mirror if exists
        try:
            cur.execute("INSERT INTO explanations_fts(rowid, text) VALUES (last_insert_rowid(), ?)", (ex.text,))
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def write_artifact(self, art: Artifact) -> None:
        self._conn.execute(
            """
            INSERT INTO artifacts (artifact_id, kind, path, meta, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                art.artifact_id,
                art.kind,
                art.path,
                json.dumps(dict(art.meta)),
                art.created_at,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read / query -------------------------------------------------------
    # ------------------------------------------------------------------

    def fetch_activations(self, *, layer: str, limit: int | None = None) -> List[ActivationEvent]:
        cur = self._conn.cursor()
        sql = (
            "SELECT run_id, step, layer, prompt, prompt_index, token_index, token, tensor, context "
            "FROM activations WHERE layer = ? "
            "ORDER BY step"
        )
        params = [layer]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = cur.execute(sql, params).fetchall()
        loads = json.loads
        ctx_cache: dict[str, dict] = {}
        events: list[ActivationEvent] = []
        append = events.append
        ActivationEventLocal = ActivationEvent
        for r in rows:
            tensor = loads(r["tensor"] or "[]")
            ctx_text = r["context"] or ""
            if ctx_text:
                ctx = ctx_cache.get(ctx_text)
                if ctx is None:
                    ctx = loads(ctx_text)
                    ctx_cache[ctx_text] = ctx
            else:
                ctx = {}
            for ch, val in enumerate(tensor):
                append(
                    ActivationEventLocal(
                        run_id=r["run_id"],
                        step=r["step"],
                        layer=r["layer"],
                        channel=ch,
                        prompt=r.get("prompt"),
                        prompt_index=r.get("prompt_index"),
                        token_index=r.get("token_index"),
                        token=r.get("token"),
                        tensor=[float(val)],
                        context=ctx,
                        value_sum=float(val),
                        value_sq_sum=float(val * val),
                    )
                )
        return events

    def fetch_vectors(self, *, layer: str, limit: int | None = None):
        cur = self._conn.cursor()
        sql = (
            "SELECT run_id, step, layer, prompt, prompt_index, token_index, token, tensor, context "
            "FROM activations WHERE layer = ? "
            "ORDER BY step"
        )
        params = [layer]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = cur.execute(sql, params).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), {}

        tensors = []
        steps = []
        prompt_index = []
        token_index = []
        prompts = []
        tokens = []
        contexts = []
        run_ids = []
        max_len = 0
        for r in rows:
            vals = json.loads(r["tensor"] or "[]")
            max_len = max(max_len, len(vals))
            tensors.append(vals)
            steps.append(int(r["step"]))
            prompt_index.append(r.get("prompt_index"))
            token_index.append(r.get("token_index"))
            prompts.append(r.get("prompt"))
            tokens.append(r.get("token"))
            contexts.append(r.get("context"))
            run_ids.append(r.get("run_id"))

        x = np.zeros((len(tensors), max_len), dtype=np.float32)
        for i, vals in enumerate(tensors):
            if vals:
                x[i, : len(vals)] = np.asarray(vals, dtype=np.float32)

        meta = {
            "step": np.asarray(steps),
            "prompt_index": np.asarray(prompt_index),
            "token_index": np.asarray(token_index),
            "prompt": prompts,
            "token": tokens,
            "context": contexts,
            "run_id": run_ids,
        }
        return x, meta

    def fetch_events(
        self,
        layer: str,
        channel: int,
        t0: float | None = None,
        t1: float | None = None,
        downsample: int = 1,
    ) -> Sequence[float]:
        cur = self._conn.cursor()
        sql = ["SELECT tensor FROM activations WHERE layer=?"]
        params: list = [layer]
        if t0 is not None:
            sql.append("AND step >= ?")
            params.append(int(t0))
        if t1 is not None:
            sql.append("AND step <= ?")
            params.append(int(t1))
        sql.append("ORDER BY step")
        cur.execute(" ".join(sql), params)
        rows = cur.fetchall()
        if not rows:
            return []
        # downsample by stride
        selected = rows[::downsample] if downsample > 1 else rows
        # flatten JSON arrays
        out: list[float] = []
        for r in selected:
            vals = json.loads(r["tensor"] or "[]")
            if channel < len(vals):
                out.append(float(vals[channel]))
        return out


    def unexplained(self, overwrite: bool) -> Iterable[StatBlock]:
        cur = self._conn.cursor()
        if overwrite:
            cur.execute("SELECT * FROM stats")
        else:
            cur.execute(
                """
                SELECT s.* FROM stats s
                LEFT JOIN explanations e ON s.layer=e.layer AND s.channel=e.channel
                WHERE e.rowid IS NULL
                """
            )
        for row in cur.fetchall():
            yield StatBlock(
                layer=row["layer"],
                channel=row["channel"],
                count=row["count"],
                mean=row["mean"],
                std=row["std"],
                min=row["min"],
                max=row["max"],
                top_correlations=json.loads(row["correlations"] or "[]"),
                last_updated=row["last_updated"],
            )

    def iter_statblocks(self, layer=None, channel=None):
        cur = self._conn.cursor()
        query = "SELECT layer, channel, count, mean, std, min, max, correlations, last_updated FROM stats"
        params = []
        if layer is not None:
            query += " WHERE layer = ?"
            params.append(layer)
            if channel is not None:
                query += " AND channel = ?"
                params.append(channel)
        elif channel is not None:
            query += " WHERE channel = ?"
            params.append(channel)

        cur.execute(query, params)
        for row in cur.fetchall():
            yield StatBlock(
                layer=row["layer"],
                channel=row["channel"],
                count=row["count"],
                mean=row["mean"],
                std=row["std"],
                min=row["min"],
                max=row["max"],
                top_correlations=json.loads(row["correlations"] or "[]"),
                last_updated=row["last_updated"],
            )

    def list_layers(self) -> list[str]:
        cur = self._conn.cursor()
        rows = cur.execute("SELECT DISTINCT layer FROM activations").fetchall()
        return [row["layer"] for row in rows]

    # ------------------------------------------------------------------
    # Stats computation --------------------------------------------------
    # ------------------------------------------------------------------

    def compute_stats(self, *, min_count: int = 1) -> None:
        """
        Aggregate per-(layer, channel) statistics and write them back into `stats`.
        Parameters
        ----------
        min_count:
            Skip channels with fewer than this many samples.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT layer, tensor
            FROM activations
            """
        )
        rows = cur.fetchall()

        aggregates: dict[tuple[str, int], dict] = {}
        for r in rows:
            layer = r["layer"]
            vals = json.loads(r["tensor"] or "[]")
            for ch, val in enumerate(vals):
                key = (layer, ch)
                agg = aggregates.setdefault(
                    key,
                    {"count": 0, "sum_x": 0.0, "sum_x2": 0.0, "min": float("inf"), "max": float("-inf")},
                )
                v = float(val)
                agg["count"] += 1
                agg["sum_x"] += v
                agg["sum_x2"] += v * v
                agg["min"] = min(agg["min"], v)
                agg["max"] = max(agg["max"], v)

        for (layer, channel), agg in aggregates.items():
            if agg["count"] < min_count:
                continue
            mean = agg["sum_x"] / agg["count"]
            var = agg["sum_x2"] / agg["count"] - mean * mean
            std = var ** 0.5 if var > 0 else 0.0
            sb = StatBlock(
                layer=layer,
                channel=channel,
                count=agg["count"],
                mean=mean,
                std=std,
                min=agg["min"],
                max=agg["max"],
                top_correlations=[],
            )
            cur.execute(
                """
                INSERT INTO stats (layer, channel, count, mean, std, min, max,
                                sum_x, sum_x2, correlations, last_updated)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(layer, channel) DO UPDATE SET
                    count       = excluded.count,
                    mean        = excluded.mean,
                    std         = excluded.std,
                    min         = excluded.min,
                    max         = excluded.max,
                    correlations = excluded.correlations,
                    last_updated = excluded.last_updated
                """,
                (
                    sb.layer, sb.channel, sb.count, sb.mean, sb.std,
                    sb.min, sb.max, agg["sum_x"], agg["sum_x2"], json.dumps([]), sb.last_updated
                ),
            )

        cur.execute("SELECT DISTINCT metric FROM metric_sums")
        all_metrics = [row["metric"] for row in cur.fetchall()]

        for row in cur.execute("SELECT * FROM stats WHERE count >= ?", (min_count,)):
            layer, ch, N = row["layer"], row["channel"], row["count"]
            mu_x = row["sum_x"] / N
            var_x = row["sum_x2"] / N - mu_x**2
            sigma_x = var_x ** 0.5 if var_x > 1e-12 else 0.0

            corrs = []
            cur2 = self._conn.cursor()
            for metric in all_metrics:
                ms = cur2.execute(
                    "SELECT count, sum_m, sum_m2, sum_xm FROM metric_sums "
                    "WHERE metric=? AND layer=? AND channel=?",
                    (metric, layer, ch)
                ).fetchone()
                if not ms or ms["count"] < min_count or sigma_x == 0:
                    continue

                mu_m = ms["sum_m"] / ms["count"]
                var_m = ms["sum_m2"] / ms["count"] - mu_m**2
                sigma_m = var_m ** 0.5
                if sigma_m < 1e-12:
                    continue

                rho = (ms["sum_xm"] / ms["count"] - mu_x * mu_m) / (sigma_x * sigma_m)
                corrs.append((metric, float(rho)))

            corrs.sort(key=lambda p: abs(p[1]), reverse=True)
            cur2.execute(
                "UPDATE stats SET correlations=? WHERE layer=? AND channel=?",
                (json.dumps(corrs[:5]), layer, ch),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # House‑keeping ------------------------------------------------------
    # ------------------------------------------------------------------


    def flush(self) -> None:
        self._conn.commit()
