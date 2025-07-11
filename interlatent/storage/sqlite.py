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
              channel    INTEGER,
              tensor     TEXT,          
              timestamp  TEXT,
              context    TEXT,      
              PRIMARY KEY (run_id, step, layer, channel)
            ) WITHOUT ROWID;
            """
        )
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
        cur = self._conn.cursor()
        metrics: dict[str, float] = ev.context.get("metrics", {})
        if metrics:                       # skip fast path if none
            sum_x  = ev.value_sum or sum(ev.tensor)
            sum_x2 = ev.value_sq_sum or sum(v*v for v in ev.tensor)
            for name, m in metrics.items():
                m = float(m)
                cur.execute(
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
                    (
                        name, ev.layer, ev.channel,
                        m,            # Σ m
                        m * m,        # Σ m²
                        sum_x * m,    # Σ x m
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO stats (layer, channel, count, sum_x, sum_x2)
                    VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT(layer, channel) DO UPDATE SET
                    count  = count  + 1,
                    sum_x  = sum_x  + EXCLUDED.sum_x,
                    sum_x2 = sum_x2 + EXCLUDED.sum_x2
                    """,
                    (ev.layer, ev.channel, sum_x, sum_x2),
                )                        
        cur.execute(
            """
            INSERT OR REPLACE INTO activations
            (run_id, step, layer, channel, tensor, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ev.run_id,
                ev.step,
                ev.layer,
                ev.channel,
                json.dumps(ev.tensor),
                ev.timestamp,
                json.dumps(ev.context),
            ),
        )
        self._conn.commit()

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

    def fetch_events(
        self,
        layer: str,
        channel: int,
        t0: float | None = None,
        t1: float | None = None,
        downsample: int = 1,
    ) -> Sequence[float]:
        cur = self._conn.cursor()
        sql = ["SELECT tensor FROM activations WHERE layer=? AND channel=?"]
        params: list = [layer, channel]
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
            out.extend(json.loads(r["tensor"]))
        return out

    def fetch_explanation(self, layer: str, channel: int) -> Explanation | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM explanations WHERE layer=? AND channel=? ORDER BY version DESC LIMIT 1",
            (layer, channel),
        )
        row = cur.fetchone()
        if not row:
            return None
        return Explanation(
            layer=row["layer"],
            channel=row["channel"],
            version=row["version"],
            text=row["text"],
            source=row["source"],
            created_at=row["created_at"],
        )
    

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

        print(query, params)
        cur.execute(query, params)

        for row in cur.fetchall():
            print("row:", row)
            print(row["count"])
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

        # Pull every (layer, channel) with at least min_count rows
        cur.execute(
            """
            SELECT layer, channel, COUNT(*)
            FROM activations
            GROUP BY layer, channel
            HAVING COUNT(*) >= ?
            """,
            (min_count,),
        )
        targets = cur.fetchall()

        print("targets:", targets)

        for row in targets:
            layer   = row["layer"]
            channel = row["channel"]
            count   = row["COUNT(*)"]

            # Fetch tensors for this (layer, channel)
            cur.execute(
                """
                SELECT tensor
                FROM activations
                WHERE layer = ? AND channel = ?
                """,
                (layer, channel),
            )

            tensors = cur.fetchall()

            flat: list[float] = []
            for tensor_dict in tensors:
                data = json.loads(tensor_dict["tensor"])
                flat.extend(data)

            if not flat:
                continue  # nothing numeric, skip

            sb = StatBlock.from_array(layer, channel, flat)
            print(sb)

        cur.execute("SELECT DISTINCT metric FROM metric_sums")
        all_metrics = [row["metric"] for row in cur.fetchall()]

        for row in cur.execute("SELECT * FROM stats WHERE count >= ?", (min_count,)):
            layer, ch, N = row["layer"], row["channel"], row["count"]

            print(row)

            mu_x  = row["sum_x"] / N
            var_x = row["sum_x2"] / N - mu_x**2
            sigma_x = var_x ** 0.5 if var_x > 1e-12 else 0.0

            print("stats", mu_x, var_x, sigma_x)

            corrs = []
            for metric in all_metrics:
                print(metric)
                ms = cur.execute(
                    "SELECT count, sum_m, sum_m2, sum_xm FROM metric_sums "
                    "WHERE metric=? AND layer=? AND channel=?",
                    (metric, layer, ch)
                ).fetchone()
                if not ms or ms["count"] < min_count or sigma_x == 0:
                    continue

                mu_m  = ms["sum_m"] / ms["count"]
                var_m = ms["sum_m2"] / ms["count"] - mu_m**2
                sigma_m = var_m ** 0.5
                if sigma_m < 1e-12:
                    continue

                rho = (ms["sum_xm"] / ms["count"] - mu_x * mu_m) / (sigma_x * sigma_m)
                print("rho", rho)
                corrs.append((metric, float(rho)))
               

            corrs.sort(key=lambda p: abs(p[1]), reverse=True)
            cur.execute(
                "UPDATE stats SET correlations=? WHERE layer=? AND channel=?",
                (json.dumps(corrs[:5]), layer, ch),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Search -------------------------------------------------------------
    # ------------------------------------------------------------------

    def search_explanations(self, query: str, k: int = 10) -> List[Explanation]:
        cur = self._conn.cursor()
        rows: list
        try:
            cur.execute(
                "SELECT layer, channel, version, text, source, created_at FROM explanations_fts WHERE explanations_fts MATCH ? LIMIT ?",
                (query, k),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # fallback using LIKE
            like_q = f"%{query}%"
            cur.execute(
                "SELECT layer, channel, version, text, source, created_at FROM explanations WHERE text LIKE ? LIMIT ?",
                (like_q, k),
            )
            rows = cur.fetchall()

        return [
            Explanation(
                layer=r["layer"],
                channel=r["channel"],
                version=r["version"],
                text=r["text"],
                source=r["source"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # House‑keeping ------------------------------------------------------
    # ------------------------------------------------------------------

    def prune_explanations(self, *, keep_most_recent: int = 3) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT layer, channel, COUNT(*) as cnt FROM explanations GROUP BY layer, channel HAVING cnt > ?", (keep_most_recent,))
        rows = cur.fetchall()
        for r in rows:
            layer, channel = r["layer"], r["channel"]
            cur.execute(
                "DELETE FROM explanations WHERE layer=? AND channel=? AND version NOT IN (SELECT version FROM explanations WHERE layer=? AND channel=? ORDER BY version DESC LIMIT ?)",
                (layer, channel, layer, channel, keep_most_recent),
            )
        self._conn.commit()

    def flush(self) -> None:
        self._conn.commit()