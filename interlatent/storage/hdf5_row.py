"""interlatent.storage.hdf5_row

HDF5 backend optimized for row-wise storage:
one row = one step/token (per run, per layer), with normalized metadata.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Iterable, List, Sequence

import h5py
import numpy as np

from ..schema import ActivationEvent, Artifact, Explanation, StatBlock
from .base import StorageBackend


def _parse_path(uri: str) -> pathlib.Path:
    if uri.startswith("hdf5row:///"):
        path = uri[len("hdf5row:///") :]
    elif uri.startswith("hdf5v2:///"):
        path = uri[len("hdf5v2:///") :]
    else:
        path = uri
    return pathlib.Path(path).expanduser().resolve()


def _layer_key(layer_id: int) -> str:
    return str(layer_id)


class HDF5RowBackend(StorageBackend):
    """Row-oriented HDF5 backend with normalized metadata dictionaries."""

    def __init__(self, uri: str):
        super().__init__(uri)
        self._path = _parse_path(uri)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "a")
        self._hidden_dim = int(os.environ.get("LATENTDB_MAX_CHANNELS", "0") or 0) or None
        if self._hidden_dim is None:
            raise ValueError("LATENTDB_MAX_CHANNELS must be set for HDF5 row backend.")
        self._chunk_rows = int(os.environ.get("LATENTDB_CHUNK_ROWS", "8192"))
        dtype_name = os.environ.get("LATENTDB_ACTIVATION_DTYPE", "float16").lower()
        if dtype_name not in ("float16", "float32"):
            raise ValueError("LATENTDB_ACTIVATION_DTYPE must be 'float16' or 'float32'.")
        self._act_dtype = np.float16 if dtype_name == "float16" else np.float32
        self._pending: dict[tuple[str, str, int], dict] = {}
        self._string_tables: dict[str, list[str]] = {}
        self._string_maps: dict[str, dict[str, int]] = {}
        self._context_cache: dict[int, dict] = {}
        self._profile_io = os.environ.get("LATENTDB_PROFILE_IO", "0") == "1"
        self._profile_every = int(os.environ.get("LATENTDB_PROFILE_EVERY", "50"))
        self._profile_counts = {"write_events": 0, "flush_pending": 0}
        self._profile_accum = {
            "write_events_s": 0.0,
            "grouping_s": 0.0,
            "step_index_s": 0.0,
            "act_write_s": 0.0,
            "flush_pending_s": 0.0,
        }
        self._ensure_schema()
        self._load_string_tables()

    # ------------------------------------------------------------------
    # Schema creation ---------------------------------------------------
    # ------------------------------------------------------------------

    def _ensure_schema(self):
        self._file.require_group("dict")
        self._file.require_group("runs")
        for name in ("prompts", "tokens", "contexts", "layers"):
            if name not in self._file["dict"]:
                self._file["dict"].create_dataset(
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
        dicts = self._file["dict"]
        for name, ds in dicts.items():
            size_attr = int(ds.attrs.get("size", ds.shape[0]))
            size = min(size_attr, ds.shape[0])
            if size < size_attr:
                ds.attrs["size"] = size
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
        ds = self._file["dict"][table]
        size = int(ds.attrs.get("size", ds.shape[0]))
        if idx >= ds.shape[0]:
            new_cap = max(ds.shape[0] + self._chunk_rows, idx + 1)
            ds.resize((new_cap,))
        ds[idx] = value
        ds.attrs["size"] = idx + 1
        return idx

    def _get_run_group(self, run_id: str):
        runs = self._file["runs"]
        if run_id not in runs:
            grp = runs.create_group(run_id)
            grp.require_group("layers")
            step_dtype = np.dtype(
                [
                    ("prompt_index", np.int32),
                    ("token_index", np.int32),
                    ("token_id", np.int32),
                    ("prompt_id", np.int32),
                    ("context_id", np.int32),
                ]
            )
            grp.create_dataset(
                "step_index",
                shape=(0,),
                maxshape=(None,),
                dtype=step_dtype,
                chunks=(self._chunk_rows,),
            )
            return grp
        return runs[run_id]

    def _ensure_step_index_size(self, run_grp, size: int) -> None:
        ds = run_grp["step_index"]
        if size <= ds.shape[0]:
            return
        ds.resize((size,))

    def _get_layer_dataset(self, run_grp, layer_id: int):
        layers = run_grp["layers"]
        key = _layer_key(layer_id)
        if key not in layers:
            grp = layers.create_group(key)
            grp.attrs["layer_name"] = self._string_tables["layers"][layer_id]
            grp.attrs["hidden_dim"] = int(self._hidden_dim)
            grp.create_dataset(
                "act",
                shape=(0, self._hidden_dim),
                maxshape=(None, self._hidden_dim),
                dtype=self._act_dtype,
                chunks=(self._chunk_rows, self._hidden_dim),
            )
            return grp["act"]
        return layers[key]["act"]

    def _ensure_act_size(self, act, size: int) -> None:
        if size <= act.shape[0]:
            return
        act.resize((size, act.shape[1]))

    def _write_step_index_rows(self, run_grp, rows: Sequence[dict]) -> None:
        if not rows:
            return
        rows_sorted = sorted(rows, key=lambda r: r["step"])
        step_ds = run_grp["step_index"]
        max_step = rows_sorted[-1]["step"]
        self._ensure_step_index_size(run_grp, max_step + 1)

        i = 0
        while i < len(rows_sorted):
            start_step = rows_sorted[i]["step"]
            j = i + 1
            while j < len(rows_sorted) and rows_sorted[j]["step"] == start_step + (j - i):
                j += 1
            count = j - i
            block = np.zeros(count, dtype=step_ds.dtype)
            for k, row in enumerate(rows_sorted[i:j]):
                block[k] = (
                    row["prompt_index"],
                    row["token_index"],
                    row["token_id"],
                    row["prompt_id"],
                    row["context_id"],
                )
            step_ds[start_step : start_step + count] = block
            i = j

    def _write_activation_rows(self, run_id: str, layer_id: int, rows: Sequence[dict]) -> None:
        if not rows:
            return
        run_grp = self._get_run_group(run_id)
        act = self._get_layer_dataset(run_grp, layer_id)
        rows_sorted = sorted(rows, key=lambda r: r["step"])
        max_step = rows_sorted[-1]["step"]
        self._ensure_act_size(act, max_step + 1)

        i = 0
        while i < len(rows_sorted):
            start_step = rows_sorted[i]["step"]
            j = i + 1
            while j < len(rows_sorted) and rows_sorted[j]["step"] == start_step + (j - i):
                j += 1
            block = np.stack([r["vec"] for r in rows_sorted[i:j]], axis=0)
            act[start_step : start_step + block.shape[0], :] = block
            i = j

    def _flush_pending(self, force: bool = False):
        t0 = time.perf_counter() if self._profile_io else None
        if not self._pending:
            return
        ready = []
        for key, rec in list(self._pending.items()):
            if force or rec["count"] >= rec["hidden_dim"]:
                ready.append((key, rec))
                del self._pending[key]
        if not ready:
            return
        per_run_layer: dict[tuple[str, int], list[dict]] = {}
        step_rows: dict[str, list[dict]] = {}
        for (run_id, layer, step), rec in ready:
            layer_id = self._string_id("layers", layer)
            per_run_layer.setdefault((run_id, layer_id), []).append(
                {"step": step, "vec": rec["vec"]}
            )
            step_rows.setdefault(run_id, []).append(
                {
                    "step": step,
                    "prompt_index": rec["prompt_index"],
                    "token_index": rec["token_index"],
                    "token_id": rec["token_id"],
                    "prompt_id": rec["prompt_id"],
                    "context_id": rec["context_id"],
                }
            )
        for run_id, rows in step_rows.items():
            run_grp = self._get_run_group(run_id)
            t_step = time.perf_counter() if self._profile_io else None
            self._write_step_index_rows(run_grp, rows)
            if self._profile_io:
                self._profile_accum["step_index_s"] += time.perf_counter() - t_step
        for (run_id, layer_id), rows in per_run_layer.items():
            t_act = time.perf_counter() if self._profile_io else None
            self._write_activation_rows(run_id, layer_id, rows)
            if self._profile_io:
                self._profile_accum["act_write_s"] += time.perf_counter() - t_act
        if self._profile_io:
            self._profile_accum["flush_pending_s"] += time.perf_counter() - t0
            self._profile_counts["flush_pending"] += 1
            if force or self._profile_counts["flush_pending"] % self._profile_every == 0:
                self._profile_report("flush_pending")

    # ------------------------------------------------------------------
    # Write methods ------------------------------------------------------
    # ------------------------------------------------------------------

    def write_event(self, ev: ActivationEvent) -> None:
        self.write_events([ev])

    def write_events(self, events: Sequence[ActivationEvent]) -> None:
        t0 = time.perf_counter() if self._profile_io else None
        t_group = time.perf_counter() if self._profile_io else None
        hidden_dim = self._hidden_dim
        grouped: dict[tuple[str, str, int], dict] = {}

        for ev in events:
            key = (ev.run_id, ev.layer, ev.step)
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

        if self._profile_io:
            self._profile_accum["grouping_s"] += time.perf_counter() - t_group
        step_rows: dict[str, list[dict]] = {}
        per_run_layer: dict[tuple[str, int], list[dict]] = {}

        for (run_id, layer, step), rec in grouped.items():
            channels = rec["channels"]
            values = rec["values"]
            if len(channels) == hidden_dim:
                min_ch = min(channels)
                max_ch = max(channels)
                full = min_ch == 0 and max_ch == hidden_dim - 1 and len(set(channels)) == hidden_dim
            else:
                full = False

            prompt_id = self._string_id("prompts", rec["prompt"])
            token_id = self._string_id("tokens", rec["token"])
            context_id = self._string_id("contexts", json.dumps(rec["context"] or {}))

            step_rows.setdefault(run_id, []).append(
                {
                    "step": step,
                    "prompt_index": int(rec["prompt_index"]),
                    "token_index": int(rec["token_index"]),
                    "token_id": int(token_id),
                    "prompt_id": int(prompt_id),
                    "context_id": int(context_id),
                }
            )

            if full:
                vec = np.zeros(hidden_dim, dtype=self._act_dtype)
                vec[np.asarray(channels, dtype=np.int32)] = np.asarray(values, dtype=self._act_dtype)
                layer_id = self._string_id("layers", layer)
                per_run_layer.setdefault((run_id, layer_id), []).append(
                    {"step": step, "vec": vec}
                )
                continue

            pending = {
                "hidden_dim": hidden_dim,
                "vec": np.zeros(hidden_dim, dtype=self._act_dtype),
                "seen": set(),
                "count": 0,
                "prompt_index": int(rec["prompt_index"]),
                "token_index": int(rec["token_index"]),
                "token_id": int(token_id),
                "prompt_id": int(prompt_id),
                "context_id": int(context_id),
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
            self._pending[(run_id, layer, step)] = pending

        for run_id, rows in step_rows.items():
            run_grp = self._get_run_group(run_id)
            self._write_step_index_rows(run_grp, rows)
        for (run_id, layer_id), rows in per_run_layer.items():
            self._write_activation_rows(run_id, layer_id, rows)

        self._flush_pending(force=False)
        if self._profile_io:
            self._profile_accum["write_events_s"] += time.perf_counter() - t0
            self._profile_counts["write_events"] += 1
            if self._profile_counts["write_events"] % self._profile_every == 0:
                self._profile_report("write_events")

    def _profile_report(self, label: str) -> None:
        msg = (
            f"[hdf5row.profile] {label} "
            f"write_events={self._profile_accum['write_events_s']:.3f}s "
            f"grouping={self._profile_accum['grouping_s']:.3f}s "
            f"flush_pending={self._profile_accum['flush_pending_s']:.3f}s "
            f"step_index={self._profile_accum['step_index_s']:.3f}s "
            f"act_write={self._profile_accum['act_write_s']:.3f}s"
        )
        print(msg)

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
            ds.resize((size + self._chunk_rows,))
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
            ds.resize((size + self._chunk_rows,))
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
            ds.resize((size + self._chunk_rows,))
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
        return list(self._string_tables["layers"])

    def list_runs(self) -> list[str]:
        return list(self._file["runs"].keys())

    def _layer_id(self, layer: str) -> int | None:
        return self._string_maps["layers"].get(layer)

    def get_block(self, *, run_id: str, layer: str, start: int, end: int):
        run_grp = self._file["runs"].get(run_id)
        if run_grp is None:
            return np.zeros((0, self._hidden_dim), dtype=self._act_dtype), {}
        layer_id = self._layer_id(layer)
        if layer_id is None:
            return np.zeros((0, self._hidden_dim), dtype=self._act_dtype), {}
        layers = run_grp["layers"]
        key = _layer_key(layer_id)
        if key not in layers:
            return np.zeros((0, self._hidden_dim), dtype=self._act_dtype), {}
        act = layers[key]["act"]
        step_index = run_grp["step_index"]
        end = min(end, act.shape[0], step_index.shape[0])
        if end <= start:
            return np.zeros((0, self._hidden_dim), dtype=self._act_dtype), {}
        x = act[start:end]
        idx = step_index[start:end]
        meta = {
            "step": np.arange(start, end, dtype=np.int64),
            "prompt_index": idx["prompt_index"],
            "token_index": idx["token_index"],
            "token_id": idx["token_id"],
            "prompt_id": idx["prompt_id"],
            "context_id": idx["context_id"],
            "run_id": run_id,
        }
        return x, meta

    def iter_events(
        self,
        *,
        run_id: str,
        layer: str,
        start: int = 0,
        end: int | None = None,
        channels: Sequence[int] | None = None,
    ):
        run_grp = self._file["runs"].get(run_id)
        if run_grp is None:
            return
        layer_id = self._layer_id(layer)
        if layer_id is None:
            return
        layers = run_grp["layers"]
        key = _layer_key(layer_id)
        if key not in layers:
            return
        act = layers[key]["act"]
        step_index = run_grp["step_index"]
        size = min(act.shape[0], step_index.shape[0])
        if end is None or end > size:
            end = size
        if end <= start:
            return
        x = act[start:end]
        idx = step_index[start:end]
        prompts = self._string_tables["prompts"]
        tokens = self._string_tables["tokens"]
        contexts = self._string_tables["contexts"]
        ActivationEventLocal = ActivationEvent
        if channels is None:
            ch_list = None
        else:
            ch_list = list(channels)

        for row_i in range(end - start):
            cid = int(idx["context_id"][row_i])
            if cid >= 0:
                ctx = self._context_cache.get(cid)
                if ctx is None:
                    ctx = json.loads(contexts[cid])
                    self._context_cache[cid] = ctx
            else:
                ctx = {}
            prompt_id = int(idx["prompt_id"][row_i])
            token_id = int(idx["token_id"][row_i])
            prompt = prompts[prompt_id] if prompt_id >= 0 else None
            token = tokens[token_id] if token_id >= 0 else None
            step = start + row_i
            if ch_list is None:
                row_vals = x[row_i].tolist()
                for ch, val in enumerate(row_vals):
                    yield ActivationEventLocal(
                        run_id=run_id,
                        step=step,
                        layer=layer,
                        channel=ch,
                        prompt=prompt,
                        prompt_index=int(idx["prompt_index"][row_i]) if idx["prompt_index"][row_i] >= 0 else None,
                        token_index=int(idx["token_index"][row_i]) if idx["token_index"][row_i] >= 0 else None,
                        token=token,
                        tensor=[float(val)],
                        context=ctx,
                        value_sum=float(val),
                        value_sq_sum=float(val * val),
                    )
            else:
                for ch in ch_list:
                    if ch < 0 or ch >= self._hidden_dim:
                        continue
                    val = float(x[row_i, ch])
                    yield ActivationEventLocal(
                        run_id=run_id,
                        step=step,
                        layer=layer,
                        channel=ch,
                        prompt=prompt,
                        prompt_index=int(idx["prompt_index"][row_i]) if idx["prompt_index"][row_i] >= 0 else None,
                        token_index=int(idx["token_index"][row_i]) if idx["token_index"][row_i] >= 0 else None,
                        token=token,
                        tensor=[float(val)],
                        context=ctx,
                        value_sum=float(val),
                        value_sq_sum=float(val * val),
                    )

    def fetch_activations(self, *, layer: str, limit: int | None = None) -> List[ActivationEvent]:
        events: list[ActivationEvent] = []
        for run_id in self.list_runs():
            run_grp = self._file["runs"].get(run_id)
            if run_grp is None:
                continue
            step_index = run_grp["step_index"]
            size = step_index.shape[0]
            end = min(size, limit) if limit else size
            for ev in self.iter_events(run_id=run_id, layer=layer, start=0, end=end):
                events.append(ev)
                if limit and len(events) >= limit:
                    return events
        return events

    def fetch_vectors(self, *, layer: str, limit: int | None = None):
        xs = []
        metas = {
            "step": [],
            "prompt_index": [],
            "token_index": [],
            "token_id": [],
            "prompt_id": [],
            "context_id": [],
            "run_id": [],
        }
        for run_id in self.list_runs():
            run_grp = self._file["runs"].get(run_id)
            if run_grp is None:
                continue
            step_index = run_grp["step_index"]
            size = step_index.shape[0]
            end = min(size, limit) if limit else size
            x, meta = self.get_block(run_id=run_id, layer=layer, start=0, end=end)
            if x.size == 0:
                continue
            xs.append(x)
            metas["step"].append(meta["step"])
            for key in ("prompt_index", "token_index", "token_id", "prompt_id", "context_id"):
                metas[key].append(meta[key])
            metas["run_id"].extend([run_id] * x.shape[0])
            if limit and sum(arr.shape[0] for arr in xs) >= limit:
                break
        if not xs:
            return np.zeros((0, self._hidden_dim), dtype=self._act_dtype), {}
        x_all = np.concatenate(xs, axis=0)
        meta_out = {
            "step": np.concatenate(metas["step"]) if metas["step"] else np.array([], dtype=np.int64),
            "prompt_index": np.concatenate(metas["prompt_index"]) if metas["prompt_index"] else np.array([], dtype=np.int32),
            "token_index": np.concatenate(metas["token_index"]) if metas["token_index"] else np.array([], dtype=np.int32),
            "token_id": np.concatenate(metas["token_id"]) if metas["token_id"] else np.array([], dtype=np.int32),
            "prompt_id": np.concatenate(metas["prompt_id"]) if metas["prompt_id"] else np.array([], dtype=np.int32),
            "context_id": np.concatenate(metas["context_id"]) if metas["context_id"] else np.array([], dtype=np.int32),
            "run_id": metas["run_id"],
            "prompts": self._string_tables["prompts"],
            "tokens": self._string_tables["tokens"],
            "contexts": self._string_tables["contexts"],
        }
        if limit and x_all.shape[0] > limit:
            x_all = x_all[:limit]
            meta_out["step"] = meta_out["step"][:limit]
            for key in ("prompt_index", "token_index", "token_id", "prompt_id", "context_id"):
                meta_out[key] = meta_out[key][:limit]
            meta_out["run_id"] = meta_out["run_id"][:limit]
        return x_all, meta_out

    def iter_activations(self, layer: str, batch_size: int = 1000):
        for run_id in self.list_runs():
            run_grp = self._file["runs"].get(run_id)
            if run_grp is None:
                continue
            step_index = run_grp["step_index"]
            size = step_index.shape[0]
            for start in range(0, size, batch_size):
                end = min(size, start + batch_size)
                events = list(self.iter_events(run_id=run_id, layer=layer, start=start, end=end))
                if events:
                    yield events

    def fetch_events(
        self,
        layer: str,
        channel: int,
        t0: float | None = None,
        t1: float | None = None,
        downsample: int = 1,
    ) -> Sequence[float]:
        out: list[float] = []
        for run_id in self.list_runs():
            run_grp = self._file["runs"].get(run_id)
            if run_grp is None:
                continue
            layer_id = self._layer_id(layer)
            if layer_id is None:
                continue
            layers = run_grp["layers"]
            key = _layer_key(layer_id)
            if key not in layers:
                continue
            act = layers[key]["act"]
            size = act.shape[0]
            start = int(t0) if t0 is not None else 0
            end = int(t1) + 1 if t1 is not None else size
            end = min(end, size)
            if end <= start:
                continue
            vals = act[start:end, channel]
            if downsample > 1:
                vals = vals[::downsample]
            out.extend(vals.astype(np.float32).tolist())
        return out

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
            for run_id in self.list_runs():
                run_grp = self._file["runs"].get(run_id)
                if run_grp is None:
                    continue
                layer_id = self._layer_id(layer)
                if layer_id is None:
                    continue
                layers = run_grp["layers"]
                key = _layer_key(layer_id)
                if key not in layers:
                    continue
                act = layers[key]["act"]
                if act.shape[0] <= 0:
                    continue
                x = act[:]
                count = x.shape[0]
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
