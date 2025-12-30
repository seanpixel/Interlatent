"""interlatent.storage.hdf5

HDF5 backend with per-layer activation batches.
Each layer is stored as a dataset of activation rows, where each row
contains the full channel vector for a single step/token.
"""
from __future__ import annotations

import json
import pathlib
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import h5py

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
    """HDF5 driver with per-layer activation batches."""

    def __init__(self, uri: str):
        super().__init__(uri)
        self._path = _parse_path(uri)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "a")
        self._ensure_schema()

    def _ensure_schema(self):
        self._file.require_group("activations")
        self._file.require_group("stats")
        self._file.require_group("explanations")
        self._file.require_group("artifacts")

    def close(self):
        self._file.flush()
        self._file.close()

    # ------------------------------------------------------------------
    # Internal helpers ---------------------------------------------------
    # ------------------------------------------------------------------

    @property
    def _act_dtype(self):
        return np.dtype(
            [
                ("run_id", h5py.string_dtype(encoding="utf-8")),
                ("step", np.int64),
                ("layer", h5py.string_dtype(encoding="utf-8")),
                ("prompt", h5py.string_dtype(encoding="utf-8")),
                ("prompt_index", np.int32),
                ("token_index", np.int32),
                ("token", h5py.string_dtype(encoding="utf-8")),
                ("tensor", h5py.vlen_dtype(np.float32)),
                ("context", h5py.string_dtype(encoding="utf-8")),
            ]
        )

    def _get_layer_ds(self, layer: str):
        group = self._file["activations"]
        key = _layer_key(layer)
        if key not in group:
            ds = group.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=self._act_dtype,
                chunks=True,
            )
            ds.attrs["layer_name"] = layer
        return group[key]

    def _append_rows(self, ds, rows: list):
        if not rows:
            return
        n = ds.shape[0]
        ds.resize((n + len(rows),))
        ds[n:] = rows

    # ------------------------------------------------------------------
    # Write methods ------------------------------------------------------
    # ------------------------------------------------------------------

    def write_event(self, ev: ActivationEvent) -> None:
        self.write_events([ev])

    def write_events(self, events: Sequence[ActivationEvent]) -> None:
        batches: dict[tuple[str, int, str], dict] = {}
        for ev in events:
            key = (ev.run_id, ev.step, ev.layer)
            batch = batches.setdefault(
                key,
                {
                    "prompt": ev.prompt or "",
                    "prompt_index": ev.prompt_index if ev.prompt_index is not None else -1,
                    "token_index": ev.token_index if ev.token_index is not None else -1,
                    "token": ev.token or "",
                    "context": ev.context or {},
                    "vec": {},
                },
            )
            batch["vec"][ev.channel] = float(ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0))

        for (run_id, step, layer), batch in batches.items():
            vec = batch["vec"]
            if vec:
                max_ch = max(vec.keys())
                tensor = np.zeros(max_ch + 1, dtype=np.float32)
                for ch, val in vec.items():
                    tensor[int(ch)] = float(val)
            else:
                tensor = np.zeros(0, dtype=np.float32)
            row = np.array(
                [
                    (
                        run_id,
                        int(step),
                        layer,
                        batch["prompt"],
                        int(batch["prompt_index"]),
                        int(batch["token_index"]),
                        batch["token"],
                        tensor,
                        json.dumps(batch["context"]),
                    )
                ],
                dtype=self._act_dtype,
            )
            ds = self._get_layer_ds(layer)
            self._append_rows(ds, row)
        self._file.flush()

    def write_statblock(self, sb: StatBlock) -> None:
        ds = self._file["stats"].get("rows")
        dtype = np.dtype(
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
        )
        if ds is None:
            ds = self._file["stats"].create_dataset("rows", shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
        row = np.array(
            [
                (
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
            ],
            dtype=dtype,
        )
        self._append_rows(ds, row)
        self._file.flush()

    def write_explanation(self, ex: Explanation) -> None:
        ds = self._file["explanations"].get("rows")
        dtype = np.dtype(
            [
                ("layer", h5py.string_dtype(encoding="utf-8")),
                ("channel", np.int32),
                ("version", np.int32),
                ("text", h5py.string_dtype(encoding="utf-8")),
                ("source", h5py.string_dtype(encoding="utf-8")),
                ("created_at", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        if ds is None:
            ds = self._file["explanations"].create_dataset("rows", shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
        row = np.array(
            [
                (
                    ex.layer,
                    int(ex.channel),
                    int(ex.version),
                    ex.text,
                    ex.source,
                    ex.created_at,
                )
            ],
            dtype=dtype,
        )
        self._append_rows(ds, row)
        self._file.flush()

    def write_artifact(self, art: Artifact) -> None:
        ds = self._file["artifacts"].get("rows")
        dtype = np.dtype(
            [
                ("artifact_id", h5py.string_dtype(encoding="utf-8")),
                ("kind", h5py.string_dtype(encoding="utf-8")),
                ("path", h5py.string_dtype(encoding="utf-8")),
                ("meta", h5py.string_dtype(encoding="utf-8")),
                ("created_at", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        if ds is None:
            ds = self._file["artifacts"].create_dataset("rows", shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
        row = np.array(
            [
                (
                    art.artifact_id,
                    art.kind,
                    art.path,
                    json.dumps(dict(art.meta)),
                    art.created_at,
                )
            ],
            dtype=dtype,
        )
        self._append_rows(ds, row)
        self._file.flush()

    # ------------------------------------------------------------------
    # Read / query -------------------------------------------------------
    # ------------------------------------------------------------------

    def list_layers(self) -> list[str]:
        layers = []
        group = self._file["activations"]
        for key in group.keys():
            ds = group[key]
            layer_name = ds.attrs.get("layer_name", key)
            layers.append(str(layer_name))
        return layers

    def fetch_activations(self, *, layer: str, limit: int | None = None) -> List[ActivationEvent]:
        ds = self._file["activations"].get(_layer_key(layer))
        if ds is None:
            return []
        rows = ds[: limit if limit else ds.shape[0]]
        events: list[ActivationEvent] = []
        for r in rows:
            tensor = r["tensor"]
            ctx = json.loads(r["context"]) if r["context"] else {}
            prompt_index = int(r["prompt_index"])
            token_index = int(r["token_index"])
            for ch, val in enumerate(tensor):
                events.append(
                    ActivationEvent(
                        run_id=str(r["run_id"]),
                        step=int(r["step"]),
                        layer=str(r["layer"]),
                        channel=ch,
                        prompt=str(r["prompt"]) or None,
                        prompt_index=prompt_index if prompt_index >= 0 else None,
                        token_index=token_index if token_index >= 0 else None,
                        token=str(r["token"]) or None,
                        tensor=[float(val)],
                        context=ctx,
                        value_sum=float(val),
                        value_sq_sum=float(val * val),
                    )
                )
        return events

    def iter_activations(self, layer: str, batch_size: int = 1000):
        ds = self._file["activations"].get(_layer_key(layer))
        if ds is None:
            return
        total = ds.shape[0]
        for start in range(0, total, batch_size):
            rows = ds[start : min(total, start + batch_size)]
            events: list[ActivationEvent] = []
            for r in rows:
                tensor = r["tensor"]
                ctx = json.loads(r["context"]) if r["context"] else {}
                prompt_index = int(r["prompt_index"])
                token_index = int(r["token_index"])
                for ch, val in enumerate(tensor):
                    events.append(
                        ActivationEvent(
                            run_id=str(r["run_id"]),
                            step=int(r["step"]),
                            layer=str(r["layer"]),
                            channel=ch,
                            prompt=str(r["prompt"]) or None,
                            prompt_index=prompt_index if prompt_index >= 0 else None,
                            token_index=token_index if token_index >= 0 else None,
                            token=str(r["token"]) or None,
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
        ds = self._file["activations"].get(_layer_key(layer))
        if ds is None:
            return []
        rows = ds[:]
        out: list[float] = []
        for r in rows:
            step = int(r["step"])
            if t0 is not None and step < int(t0):
                continue
            if t1 is not None and step > int(t1):
                continue
            tensor = r["tensor"]
            if channel < len(tensor):
                out.append(float(tensor[channel]))
        if downsample > 1:
            out = out[::downsample]
        return out

    def unexplained(self, overwrite: bool) -> Iterable[StatBlock]:
        stats = list(self.iter_statblocks())
        if overwrite:
            yield from stats
            return
        explained = set()
        ds = self._file["explanations"].get("rows")
        if ds is not None:
            for r in ds[:]:
                explained.add((str(r["layer"]), int(r["channel"])))
        for sb in stats:
            if (sb.layer, sb.channel) not in explained:
                yield sb

    def iter_statblocks(self, layer: str | None = None, channel: int | None = None) -> Iterable[StatBlock]:
        ds = self._file["stats"].get("rows")
        if ds is None:
            return []
        for r in ds[:]:
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
        aggregates: dict[tuple[str, int], dict] = {}
        for layer in self.list_layers():
            ds = self._file["activations"].get(_layer_key(layer))
            if ds is None:
                continue
            for r in ds[:]:
                vals = r["tensor"]
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

        dtype = np.dtype(
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
        )
        rows = []
        for (layer, ch), agg in aggregates.items():
            if agg["count"] < min_count:
                continue
            mean = agg["sum_x"] / agg["count"]
            var = agg["sum_x2"] / agg["count"] - mean * mean
            std = var ** 0.5 if var > 0 else 0.0
            rows.append(
                (
                    layer,
                    int(ch),
                    int(agg["count"]),
                    float(mean),
                    float(std),
                    float(agg["min"]),
                    float(agg["max"]),
                    json.dumps([]),
                    "",
                )
            )
        stats_group = self._file["stats"]
        if "rows" in stats_group:
            del stats_group["rows"]
        ds = stats_group.create_dataset("rows", shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
        if rows:
            arr = np.array(rows, dtype=dtype)
            self._append_rows(ds, arr)
        self._file.flush()

    # ------------------------------------------------------------------
    # House‑keeping ------------------------------------------------------
    # ------------------------------------------------------------------

    def flush(self) -> None:
        self._file.flush()
