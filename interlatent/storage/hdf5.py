"""interlatent.storage.hdf5

HDF5 backend with per-layer fixed-shape activations and row-aligned metadata.
Designed for fast sequential writes and vectorized reads.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Iterable, List, Sequence, Tuple

import h5py
import numpy as np

from ..schema import ActivationEvent, Artifact, Explanation, StatBlock
from .base import StorageBackend


def _parse_path(uri: str) -> pathlib.Path:
    if uri.startswith("hdf5:///"):
        path = uri[len("hdf5:///") :]
    elif uri.startswith("h5:///"):
        path = uri[len("h5:///") :]
    else:
        path = uri
    return pathlib.Path(path).expanduser().resolve()


def _layer_key(layer: str) -> str:
    return layer.replace("/", "_")


class HDF5Backend(StorageBackend):
    """HDF5 driver with per-layer datasets and dense activation rows."""

    def __init__(self, uri: str):
        super().__init__(uri)
        self._path = _parse_path(uri)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "a")
        self._init_capacity = int(os.environ.get("LATENTDB_HDF5_INIT_CAPACITY", "1024"))
        self._grow_by = int(os.environ.get("LATENTDB_HDF5_GROW_BY", "10000"))
        self._hidden_dim_default = int(os.environ.get("LATENTDB_MAX_CHANNELS", "0") or 0) or None
        self._pending: dict[tuple[str, int, str], dict] = {}
        self._string_tables = {}
        self._string_maps = {}
        self._context_cache: dict[int, dict] = {}
        self._ensure_schema()
        self._load_string_tables()

    # ------------------------------------------------------------------
    # Schema creation ---------------------------------------------------
    # ------------------------------------------------------------------

    def _ensure_schema(self):
        self._file.require_group("activations")
        strings = self._file.require_group("strings")
        for name in ("prompts", "tokens", "contexts", "run_ids"):
            if name not in strings:
                strings.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    chunks=True,
                ).attrs["size"] = 0
        self._file.require_group("stats")
        self._file.require_group("explanations")
        self._file.require_group("artifacts")

    def _load_string_tables(self):
        strings = self._file["strings"]
        for name, ds in strings.items():
            size = int(ds.attrs.get("size", ds.shape[0]))
            vals = [str(v) for v in ds[:size]]
            self._string_tables[name] = vals
            self._string_maps[name] = {v: i for i, v in enumerate(vals)}

    # ------------------------------------------------------------------
    # Lifecycle ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        self.flush()
        self._file.close()

    def flush(self) -> None:
        self._flush_pending(force=True)
        self._file.flush()

    # ------------------------------------------------------------------
    # Internal helpers ---------------------------------------------------
    # ------------------------------------------------------------------

    def _string_id(self, table: str, value: str | None) -> int:
        if value is None:
            return -1
        table_vals = self._string_tables[table]
        table_map = self._string_maps[table]
        if value in table_map:
            return table_map[value]
        idx = len(table_vals)
        table_vals.append(value)
        table_map[value] = idx
        ds = self._file["strings"][table]
        size = int(ds.attrs.get("size", ds.shape[0]))
        if idx >= ds.shape[0]:
            new_cap = max(ds.shape[0] + self._grow_by, idx + 1)
            ds.resize((new_cap,))
        ds[idx] = value
        ds.attrs["size"] = idx + 1
        return idx

    def _get_layer_group(self, layer: str, hidden_dim: int):
        act = self._file["activations"]
        key = _layer_key(layer)
        if key not in act:
            grp = act.create_group(key)
            grp.attrs["layer_name"] = layer
            grp.attrs["hidden_dim"] = int(hidden_dim)
            grp.attrs["size"] = 0
            cap = max(self._init_capacity, 1)
            chunk_rows = min(1024, cap)
            grp.create_dataset("x", shape=(cap, hidden_dim), maxshape=(None, hidden_dim), dtype=np.float32, chunks=(chunk_rows, hidden_dim))
            for name, dtype in (
                ("step", np.int64),
                ("prompt_index", np.int32),
                ("token_index", np.int32),
                ("prompt_id", np.int32),
                ("token_id", np.int32),
                ("context_id", np.int32),
                ("run_id_id", np.int32),
            ):
                grp.create_dataset(name, shape=(cap,), maxshape=(None,), dtype=dtype, chunks=(chunk_rows,))
            return grp
        grp = act[key]
        return grp

    def _ensure_capacity(self, grp, n_new: int):
        size = int(grp.attrs.get("size", 0))
        cap = grp["x"].shape[0]
        if size + n_new <= cap:
            return size
        grow = max(self._grow_by, n_new)
        new_cap = size + grow
        grp["x"].resize((new_cap, grp["x"].shape[1]))
        for name in ("step", "prompt_index", "token_index", "prompt_id", "token_id", "context_id", "run_id_id"):
            grp[name].resize((new_cap,))
        return size

    def _flush_pending(self, force: bool = False):
        if not self._pending:
            return
        ready = []
        for key, rec in list(self._pending.items()):
            if force or rec["count"] >= rec["hidden_dim"]:
                ready.append((key, rec))
                del self._pending[key]
        if not ready:
            return
        per_layer: dict[str, list[dict]] = {}
        for (run_id, step, layer), rec in ready:
            per_layer.setdefault(layer, []).append(
                {
                    "vec": rec["vec"],
                    "step": int(step),
                    "prompt_index": int(rec["prompt_index"]),
                    "token_index": int(rec["token_index"]),
                    "prompt_id": int(rec["prompt_id"]),
                    "token_id": int(rec["token_id"]),
                    "context_id": int(rec["context_id"]),
                    "run_id_id": int(rec["run_id_id"]),
                }
            )
        for layer, rows in per_layer.items():
            self._append_rows(layer, rows)

    def _append_rows(self, layer: str, rows: Sequence[dict]) -> None:
        if not rows:
            return
        hidden_dim = int(rows[0]["vec"].shape[0])
        grp = self._get_layer_group(layer, hidden_dim)
        size = self._ensure_capacity(grp, len(rows))
        end = size + len(rows)

        x = np.stack([r["vec"] for r in rows], axis=0)
        grp["x"][size:end, :] = x
        grp["step"][size:end] = np.asarray([r["step"] for r in rows], dtype=np.int64)
        grp["prompt_index"][size:end] = np.asarray([r["prompt_index"] for r in rows], dtype=np.int32)
        grp["token_index"][size:end] = np.asarray([r["token_index"] for r in rows], dtype=np.int32)
        grp["prompt_id"][size:end] = np.asarray([r["prompt_id"] for r in rows], dtype=np.int32)
        grp["token_id"][size:end] = np.asarray([r["token_id"] for r in rows], dtype=np.int32)
        grp["context_id"][size:end] = np.asarray([r["context_id"] for r in rows], dtype=np.int32)
        grp["run_id_id"][size:end] = np.asarray([r["run_id_id"] for r in rows], dtype=np.int32)
        grp.attrs["size"] = end

    # ------------------------------------------------------------------
    # Write methods ------------------------------------------------------
    # ------------------------------------------------------------------

    def write_event(self, ev: ActivationEvent) -> None:
        self.write_events([ev])

    def write_events(self, events: Sequence[ActivationEvent]) -> None:
        hidden_dim = self._hidden_dim_default
        if hidden_dim is None:
            raise ValueError("LATENTDB_MAX_CHANNELS must be set for HDF5 backend.")

        grouped: dict[tuple[str, int, str], dict] = {}
        for ev in events:
            key = (ev.run_id, ev.step, ev.layer)
            rec = self._pending.get(key)
            if rec is not None:
                if ev.channel >= rec["hidden_dim"]:
                    raise ValueError(
                        f"channel {ev.channel} exceeds hidden_dim {rec['hidden_dim']}; "
                        "set LATENTDB_MAX_CHANNELS to the collector max_channels"
                    )
                if ev.channel not in rec["seen"]:
                    rec["count"] += 1
                    rec["seen"].add(ev.channel)
                rec["vec"][ev.channel] = float(
                    ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0)
                )
                continue

            rec = grouped.get(key)
            if rec is None:
                rec = {
                    "channels": [],
                    "values": [],
                    "prompt": ev.prompt,
                    "token": ev.token,
                    "context": ev.context,
                    "prompt_index": ev.prompt_index if ev.prompt_index is not None else -1,
                    "token_index": ev.token_index if ev.token_index is not None else -1,
                    "run_id": ev.run_id,
                }
                grouped[key] = rec
            rec["channels"].append(int(ev.channel))
            rec["values"].append(
                float(ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0))
            )
            if rec["prompt"] is None:
                rec["prompt"] = ev.prompt
            if rec["token"] is None:
                rec["token"] = ev.token
            if rec["context"] is None:
                rec["context"] = ev.context

        per_layer: dict[str, list[dict]] = {}
        for (run_id, step, layer), rec in grouped.items():
            channels = rec["channels"]
            values = rec["values"]
            if len(channels) == hidden_dim:
                min_ch = min(channels)
                max_ch = max(channels)
                full = min_ch == 0 and max_ch == hidden_dim - 1 and len(set(channels)) == hidden_dim
            else:
                full = False

            if full:
                vec = np.zeros(hidden_dim, dtype=np.float32)
                vec[np.asarray(channels, dtype=np.int32)] = np.asarray(values, dtype=np.float32)
                row = {
                    "vec": vec,
                    "step": int(step),
                    "prompt_index": int(rec["prompt_index"]),
                    "token_index": int(rec["token_index"]),
                    "prompt_id": self._string_id("prompts", rec["prompt"]),
                    "token_id": self._string_id("tokens", rec["token"]),
                    "context_id": self._string_id("contexts", json.dumps(rec["context"] or {})),
                    "run_id_id": self._string_id("run_ids", rec["run_id"]),
                }
                per_layer.setdefault(layer, []).append(row)
                continue

            pending = {
                "hidden_dim": hidden_dim,
                "vec": np.zeros(hidden_dim, dtype=np.float32),
                "seen": set(),
                "count": 0,
                "prompt_index": rec["prompt_index"],
                "token_index": rec["token_index"],
                "prompt_id": self._string_id("prompts", rec["prompt"]),
                "token_id": self._string_id("tokens", rec["token"]),
                "context_id": self._string_id("contexts", json.dumps(rec["context"] or {})),
                "run_id_id": self._string_id("run_ids", rec["run_id"]),
            }
            for ch, val in zip(channels, values):
                if ch >= hidden_dim:
                    raise ValueError(
                        f"channel {ch} exceeds hidden_dim {hidden_dim}; "
                        "set LATENTDB_MAX_CHANNELS to the collector max_channels"
                    )
                if ch not in pending["seen"]:
                    pending["count"] += 1
                    pending["seen"].add(ch)
                pending["vec"][ch] = float(val)
            self._pending[(run_id, step, layer)] = pending

        for layer, rows in per_layer.items():
            self._append_rows(layer, rows)

        self._flush_pending(force=False)

    def write_statblock(self, sb: StatBlock) -> None:
        stats = self._file["stats"]
        if "rows" not in stats:
            stats.create_dataset(
                "rows",
                shape=(0,),
                maxshape=(None,),
                dtype=np.dtype(
                    [
                        ("layer", h5py.string_dtype(encoding="utf-8")),
                        ("channel", np.int32),
                        ("count", np.int64),
                        ("mean", np.float64),
                        ("std", np.float64),
                        ("min", np.float64),
                        ("max", np.float64),
                        ("correlations", h5py.string_dtype(encoding="utf-8")),
                        ("last_updated", h5py.string_dtype(encoding="utf-8")),
                    ]
                ),
                chunks=True,
            ).attrs["size"] = 0
        ds = stats["rows"]
        size = int(ds.attrs.get("size", ds.shape[0]))
        if size >= ds.shape[0]:
            ds.resize((size + self._grow_by,))
        ds[size] = (
            sb.layer,
            int(sb.channel),
            int(sb.count),
            float(sb.mean),
            float(sb.std),
            float(sb.min),
            float(sb.max),
            json.dumps(sb.top_correlations),
            sb.last_updated,
        )
        ds.attrs["size"] = size + 1

    def write_explanation(self, ex: Explanation) -> None:
        expl = self._file["explanations"]
        if "rows" not in expl:
            expl.create_dataset(
                "rows",
                shape=(0,),
                maxshape=(None,),
                dtype=np.dtype(
                    [
                        ("layer", h5py.string_dtype(encoding="utf-8")),
                        ("channel", np.int32),
                        ("version", np.int32),
                        ("text", h5py.string_dtype(encoding="utf-8")),
                        ("source", h5py.string_dtype(encoding="utf-8")),
                        ("created_at", h5py.string_dtype(encoding="utf-8")),
                    ]
                ),
                chunks=True,
            ).attrs["size"] = 0
        ds = expl["rows"]
        size = int(ds.attrs.get("size", ds.shape[0]))
        if size >= ds.shape[0]:
            ds.resize((size + self._grow_by,))
        ds[size] = (
            ex.layer,
            int(ex.channel),
            int(ex.version),
            ex.text,
            ex.source,
            ex.created_at,
        )
        ds.attrs["size"] = size + 1

    def write_artifact(self, art: Artifact) -> None:
        arts = self._file["artifacts"]
        if "rows" not in arts:
            arts.create_dataset(
                "rows",
                shape=(0,),
                maxshape=(None,),
                dtype=np.dtype(
                    [
                        ("artifact_id", h5py.string_dtype(encoding="utf-8")),
                        ("kind", h5py.string_dtype(encoding="utf-8")),
                        ("path", h5py.string_dtype(encoding="utf-8")),
                        ("meta", h5py.string_dtype(encoding="utf-8")),
                        ("created_at", h5py.string_dtype(encoding="utf-8")),
                    ]
                ),
                chunks=True,
            ).attrs["size"] = 0
        ds = arts["rows"]
        size = int(ds.attrs.get("size", ds.shape[0]))
        if size >= ds.shape[0]:
            ds.resize((size + self._grow_by,))
        ds[size] = (
            art.artifact_id,
            art.kind,
            art.path,
            json.dumps(dict(art.meta)),
            art.created_at,
        )
        ds.attrs["size"] = size + 1

    # ------------------------------------------------------------------
    # Read / query -------------------------------------------------------
    # ------------------------------------------------------------------

    def list_layers(self) -> list[str]:
        layers = []
        for key, grp in self._file["activations"].items():
            layers.append(str(grp.attrs.get("layer_name", key)))
        return layers

    def _layer_group(self, layer: str):
        return self._file["activations"].get(_layer_key(layer))

    def fetch_activations(self, *, layer: str, limit: int | None = None) -> List[ActivationEvent]:
        grp = self._layer_group(layer)
        if grp is None:
            return []
        size = int(grp.attrs.get("size", 0))
        n = min(size, limit) if limit else size
        if n <= 0:
            return []
        x = grp["x"][:n]
        step = grp["step"][:n]
        prompt_index = grp["prompt_index"][:n]
        token_index = grp["token_index"][:n]
        prompt_id = grp["prompt_id"][:n]
        token_id = grp["token_id"][:n]
        context_id = grp["context_id"][:n]
        run_id_id = grp["run_id_id"][:n]

        prompts = self._string_tables["prompts"]
        tokens = self._string_tables["tokens"]
        contexts = self._string_tables["contexts"]
        run_ids = self._string_tables["run_ids"]

        events: list[ActivationEvent] = []
        ActivationEventLocal = ActivationEvent
        append = events.append
        for i in range(n):
            cid = int(context_id[i])
            if cid >= 0:
                ctx = self._context_cache.get(cid)
                if ctx is None:
                    ctx = json.loads(contexts[cid])
                    self._context_cache[cid] = ctx
            else:
                ctx = {}
            prompt = prompts[prompt_id[i]] if prompt_id[i] >= 0 else None
            token = tokens[token_id[i]] if token_id[i] >= 0 else None
            run_id = run_ids[run_id_id[i]] if run_id_id[i] >= 0 else ""
            row_vals = x[i].tolist()
            for ch, val in enumerate(row_vals):
                append(
                    ActivationEventLocal(
                        run_id=run_id,
                        step=int(step[i]),
                        layer=layer,
                        channel=ch,
                        prompt=prompt,
                        prompt_index=int(prompt_index[i]) if prompt_index[i] >= 0 else None,
                        token_index=int(token_index[i]) if token_index[i] >= 0 else None,
                        token=token,
                        tensor=[float(val)],
                        context=ctx,
                        value_sum=float(val),
                        value_sq_sum=float(val * val),
                    )
                )
        return events

    def fetch_vectors(self, *, layer: str, limit: int | None = None):
        grp = self._layer_group(layer)
        if grp is None:
            return np.zeros((0, 0), dtype=np.float32), {}
        size = int(grp.attrs.get("size", 0))
        n = min(size, limit) if limit else size
        if n <= 0:
            return np.zeros((0, grp["x"].shape[1]), dtype=np.float32), {}

        x = grp["x"][:n]
        meta = {
            "step": grp["step"][:n],
            "prompt_index": grp["prompt_index"][:n],
            "token_index": grp["token_index"][:n],
            "prompt_id": grp["prompt_id"][:n],
            "token_id": grp["token_id"][:n],
            "context_id": grp["context_id"][:n],
            "run_id_id": grp["run_id_id"][:n],
            "prompts": self._string_tables["prompts"],
            "tokens": self._string_tables["tokens"],
            "contexts": self._string_tables["contexts"],
            "run_ids": self._string_tables["run_ids"],
        }
        return x, meta

    def iter_activations(self, layer: str, batch_size: int = 1000):
        grp = self._layer_group(layer)
        if grp is None:
            return
        size = int(grp.attrs.get("size", 0))
        prompts = self._string_tables["prompts"]
        tokens = self._string_tables["tokens"]
        contexts = self._string_tables["contexts"]
        run_ids = self._string_tables["run_ids"]
        for start in range(0, size, batch_size):
            end = min(size, start + batch_size)
            x = grp["x"][start:end]
            step = grp["step"][start:end]
            prompt_index = grp["prompt_index"][start:end]
            token_index = grp["token_index"][start:end]
            prompt_id = grp["prompt_id"][start:end]
            token_id = grp["token_id"][start:end]
            context_id = grp["context_id"][start:end]
            run_id_id = grp["run_id_id"][start:end]
            events: list[ActivationEvent] = []
            ActivationEventLocal = ActivationEvent
            append = events.append
            for i in range(end - start):
                cid = int(context_id[i])
                if cid >= 0:
                    ctx = self._context_cache.get(cid)
                    if ctx is None:
                        ctx = json.loads(contexts[cid])
                        self._context_cache[cid] = ctx
                else:
                    ctx = {}
                prompt = prompts[prompt_id[i]] if prompt_id[i] >= 0 else None
                token = tokens[token_id[i]] if token_id[i] >= 0 else None
                run_id = run_ids[run_id_id[i]] if run_id_id[i] >= 0 else ""
                row_vals = x[i].tolist()
                for ch, val in enumerate(row_vals):
                    append(
                        ActivationEventLocal(
                            run_id=run_id,
                            step=int(step[i]),
                            layer=layer,
                            channel=ch,
                            prompt=prompt,
                            prompt_index=int(prompt_index[i]) if prompt_index[i] >= 0 else None,
                            token_index=int(token_index[i]) if token_index[i] >= 0 else None,
                            token=token,
                            tensor=[float(val)],
                            context=ctx,
                            value_sum=float(val),
                            value_sq_sum=float(val * val),
                        )
                    )
            yield events

    def fetch_events(
        self,
        layer: str,
        channel: int,
        t0: float | None = None,
        t1: float | None = None,
        downsample: int = 1,
    ) -> Sequence[float]:
        grp = self._layer_group(layer)
        if grp is None:
            return []
        size = int(grp.attrs.get("size", 0))
        if size <= 0:
            return []
        step = grp["step"][:size]
        x = grp["x"][:size, channel]
        mask = np.ones(size, dtype=bool)
        if t0 is not None:
            mask &= step >= int(t0)
        if t1 is not None:
            mask &= step <= int(t1)
        out = x[mask]
        if downsample > 1:
            out = out[::downsample]
        return out.tolist()

    def unexplained(self, overwrite: bool) -> Iterable[StatBlock]:
        stats = list(self.iter_statblocks())
        if overwrite:
            yield from stats
            return
        explained = set()
        ds = self._file["explanations"].get("rows")
        if ds is not None:
            size = int(ds.attrs.get("size", ds.shape[0]))
            for r in ds[:size]:
                explained.add((str(r["layer"]), int(r["channel"])))
        for sb in stats:
            if (sb.layer, sb.channel) not in explained:
                yield sb

    def iter_statblocks(self, layer: str | None = None, channel: int | None = None) -> Iterable[StatBlock]:
        ds = self._file["stats"].get("rows")
        if ds is None:
            return []
        size = int(ds.attrs.get("size", ds.shape[0]))
        for r in ds[:size]:
            if layer is not None and str(r["layer"]) != layer:
                continue
            if channel is not None and int(r["channel"]) != channel:
                continue
            yield StatBlock(
                layer=str(r["layer"]),
                channel=int(r["channel"]),
                count=int(r["count"]),
                mean=float(r["mean"]),
                std=float(r["std"]),
                min=float(r["min"]),
                max=float(r["max"]),
                top_correlations=json.loads(r["correlations"] or "[]"),
                last_updated=str(r["last_updated"]),
            )

    # ------------------------------------------------------------------
    # Stats computation --------------------------------------------------
    # ------------------------------------------------------------------

    def compute_stats(self, *, min_count: int = 1) -> None:
        rows = []
        for layer in self.list_layers():
            grp = self._layer_group(layer)
            if grp is None:
                continue
            size = int(grp.attrs.get("size", 0))
            if size <= 0:
                continue
            x = grp["x"][:size]
            count = size
            if count < min_count:
                continue
            mean = x.mean(axis=0)
            std = x.std(axis=0)
            minv = x.min(axis=0)
            maxv = x.max(axis=0)
            for ch in range(x.shape[1]):
                rows.append(
                    (
                        layer,
                        int(ch),
                        int(count),
                        float(mean[ch]),
                        float(std[ch]),
                        float(minv[ch]),
                        float(maxv[ch]),
                        json.dumps([]),
                        "",
                    )
                )
        stats = self._file["stats"]
        if "rows" in stats:
            del stats["rows"]
        stats.create_dataset(
            "rows",
            shape=(0,),
            maxshape=(None,),
            dtype=np.dtype(
                [
                    ("layer", h5py.string_dtype(encoding="utf-8")),
                    ("channel", np.int32),
                    ("count", np.int64),
                    ("mean", np.float64),
                    ("std", np.float64),
                    ("min", np.float64),
                    ("max", np.float64),
                    ("correlations", h5py.string_dtype(encoding="utf-8")),
                    ("last_updated", h5py.string_dtype(encoding="utf-8")),
                ]
            ),
            chunks=True,
        ).attrs["size"] = 0
        ds = stats["rows"]
        if rows:
            ds.resize((len(rows),))
            ds[:] = np.array(rows, dtype=ds.dtype)
            ds.attrs["size"] = len(rows)
