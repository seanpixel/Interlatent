# How Interlatent Works

This document is an internal, technical overview of the system architecture and data flow. It starts with a birds‑eye view and then zooms into storage, collection, datasets, training, and analysis/visualization. Code snippets are taken directly from the repository and labeled with their source paths.

---

## 1) Birds‑Eye View
Interlatent is built around a simple loop:

1. **Collect activations** from models (LLM or RL policy).
2. **Persist activations + metadata** in a scalable storage backend.
3. **Build datasets** for probing or sparse autoencoders from stored activations.
4. **Train models** (linear probes, transcoders, SAEs).
5. **Analyze / visualize** latent features and token traces.

The storage backend is the spine: all downstream tasks operate on it, either by reading per‑step vectors (`fetch_vectors` / `get_block`) or by materializing per‑channel events when needed.

Entry point (backend resolution) in `interlatent/api/latent_db.py`:
```python
_SCHEME_TO_BACKEND = {
    "sqlite": "..storage.sqlite:SQLiteBackend",
    "file": "..storage.sqlite:SQLiteBackend",
    "hdf5": "..storage.hdf5:HDF5Backend",
    "h5": "..storage.hdf5:HDF5Backend",
    "hdf5row": "..storage.hdf5_row:HDF5RowBackend",
    "hdf5v2": "..storage.hdf5_row:HDF5RowBackend",
}
```

The `LatentDB` write path batches `ActivationEvent` objects and flushes them to the backend:
```python
if self._write_batch_size > 0:
    self._write_buffer.append(event)
    if len(self._write_buffer) >= self._write_batch_size:
        self._flush_buffer()
else:
    self._store.write_event(event)
```
`interlatent/api/latent_db.py`

---

## 2) Data Model and Storage

### 2.0 ActivationEvent contract
The system’s wire format is `ActivationEvent`, defined in `interlatent/schema.py`:
```python
class ActivationEvent(BaseModel):
    run_id: str
    step: int
    layer: str
    channel: int
    prompt: str | None
    prompt_index: int | None
    token_index: int | None
    token: str | None
    value_sum: float | None
    value_sq_sum: float | None
    tensor: List[float]
    timestamp: str
    context: Dict[str, Any]
```
Storage backends map their physical layout into this logical shape. The row backend keeps `step` as the row index so it can reconstruct events on demand.

### 2.1 SQLite Backend (small/medium runs)

### 2.1 SQLite Backend (small/medium runs)
**Goal:** simple single‑file DB for quick experiments.

Layout:
- `activations` table: one row per `(run_id, step, layer)` with a dense JSON vector.
- `stats`, `metric_sums`: aggregated stats and correlations.
- `explanations`, `artifacts`: auxiliary tables.

Schema (from `interlatent/storage/sqlite.py`):
```sql
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
```

Writing is batched per `(run_id, step, layer)` with a single JSON vector:
```python
activation_rows.append(
    (run_id, step, layer, prompt, prompt_index, token_index, token,
     json.dumps(tensor), json.dumps(context))
)
cur.executemany("INSERT OR REPLACE INTO activations ...", activation_rows)
```
`interlatent/storage/sqlite.py`

Reads expand JSON rows into per‑channel events:
```python
tensor = json.loads(r["tensor"] or "[]")
for ch, val in enumerate(tensor):
    events.append(ActivationEvent(run_id=r["run_id"], step=r["step"], layer=r["layer"], channel=ch, ...))
```
`interlatent/storage/sqlite.py`

**Access pattern:** good for small runs but per‑row JSON parsing and per‑channel expansion is expensive at scale.

### 2.2 HDF5 Row Backend (large runs)
**Goal:** one row = one step/token per layer; minimize metadata duplication.

Logical layout:
- `/runs/<run_id>/step_index`  
  Structured array of integers:
  `prompt_index`, `token_index`, `token_id`, `prompt_id`, `context_id`.
- `/runs/<run_id>/layers/<layer_id>/act`  
  Dense numeric matrix `(steps × channels)` stored as float16/float32.
- `/dict/prompts`, `/dict/tokens`, `/dict/contexts`, `/dict/layers`  
  Normalized dictionaries.

**Join key:** `(run_id, layer_id, step)` where `step` is the row number.  
This is enough to reconstruct a full `ActivationEvent` when required.

Core schema creation (from `interlatent/storage/hdf5_row.py`):
```python
grp.create_dataset(
    "step_index",
    shape=(0,),
    maxshape=(None,),
    dtype=step_dtype,
    chunks=(self._chunk_rows,),
)
grp.create_dataset(
    "act",
    shape=(0, self._hidden_dim),
    maxshape=(None, self._hidden_dim),
    dtype=self._act_dtype,
    chunks=(self._chunk_rows, self._hidden_dim),
)
```

Row writes are buffered and flushed in contiguous slices:
```python
block = np.stack([r["vec"] for r in rows_sorted[i:j]], axis=0)
act[start_step : start_step + block.shape[0], :] = block
```
`interlatent/storage/hdf5_row.py`

Fast APIs:
- `get_block(run_id, layer, start, end)` returns `(x, idx)` without event materialization.
- `fetch_vectors(layer)` returns all vectors + metadata for a layer.

Facade call in `interlatent/api/latent_db.py`:
```python
def get_block(self, *, run_id: str, layer: str, start: int, end: int):
    if hasattr(self._store, "get_block"):
        return self._store.get_block(run_id=run_id, layer=layer, start=start, end=end)
    x, meta = self._store.fetch_vectors(layer=layer, limit=None)
    return x[start:end], {k: v[start:end] for k, v in meta.items()}
```

Fast block read (from `interlatent/storage/hdf5_row.py`):
```python
x = act[start:end]
idx = step_index[start:end]
meta = {"prompt_index": idx["prompt_index"], "token_index": idx["token_index"], ...}
return x, meta
```

Slow APIs:
- `iter_events(...)` reconstructs `ActivationEvent` objects only for slices or selected channels.

Event materialization (from `interlatent/storage/hdf5_row.py`):
```python
for ch in ch_list:
    val = float(x[row_i, ch])
    yield ActivationEvent(run_id=run_id, step=step, layer=layer, channel=ch, ...)
```

---

## 3) Collection: LLM and RL

### 3.1 LLMCollector
**Inputs:** a HF model, tokenizer, prompts, and layer indices.

Process:
- Runs a forward/generate call **once** per batch.
- Reads multiple layers from the returned `hidden_states` in memory.
- Writes per‑token activations to the DB:
  - `step` increments once per token (shared across layers).
  - Metadata includes `prompt_index`, `token_index`, `token_id`, and context.

Core loop (from `interlatent/collectors/llm_collector.py`):
```python
layer_indices = self._resolve_layers(len(hidden_states))
for b_idx, prompt_text in enumerate(batch):
    ...
    for token_idx in range(max_len):
        for layer_idx in layer_indices:
            layer_tensor = hidden_states[layer_idx]
            for ch in range(H):
                self.db.write_event(
                    ActivationEvent(run_id=run_id, step=event_step, layer=layer_name, ...)
                )
        event_step += 1
```

Key parameters:
- `layer_indices`: list of hidden state layers to capture.
- `max_channels`: cap hidden size (or use `LATENTDB_MAX_CHANNELS` for full width).
- `log_every_prompts`: progress visibility.

### 3.2 GymCollector (RL)
Captures activations at each environment step. Can log `{layer}:pre` and `{layer}:post` for transcoders. Metadata includes environment metrics such as reward or state features.

---

## 4) Datasets: From Stored Activations to Training Batches

**Fast path (preferred for large runs):**
- `fetch_vectors` returns dense `(N × D)` activations + metadata arrays.
- Datasets construct tensors directly from these blocks.

Example fast path (from `interlatent/analysis/dataset/activation_vector_dataset.py`):
```python
x, meta = db.fetch_vectors(layer=layer, limit=limit)
if x.size:
    vec = torch.tensor(x[i], dtype=torch.float32)
```

**Slow path (legacy/small runs):**
- `fetch_activations` returns per‑channel `ActivationEvent`s.
- Datasets group events by step/prompt/token and reconstruct vectors.

Legacy grouping (from `interlatent/analysis/dataset/activation_vector_dataset.py`):
```python
grouped.setdefault(key, {})[ev.channel] = ev.value_sum or sum(ev.tensor)
vec = torch.tensor([vec_dict.get(ch, 0.0) for ch in channel_order], dtype=torch.float32)
```

Core datasets:
- `ActivationVectorDataset`: per‑step activation vectors.
- `ActivationPairDataset`: pre/post pairs (RL use cases).
- `LinearProbeDataset`: vectors + targets from `context["metrics"]`.

Target extraction (from `interlatent/analysis/dataset/linear_probe_dataset.py`):
```python
metrics = (ctx or {}).get("metrics", {})
tgt = metrics.get(target_key)
if tgt is None:
    tgt = (ctx or {}).get(target_key)
samples.append((torch.tensor(x[i], dtype=torch.float32), torch.tensor(float(tgt))))
```

---

## 5) Training Pipelines

### 5.1 Linear Probe
Fits a simple linear model from activations to labels (classification or regression). Labels come from `context["metrics"]` in stored events.

### 5.2 TranscoderPipeline
Sparse bottleneck autoencoder that projects activations into `latent:{layer}`. This backfills new activations into the DB, enabling downstream analysis of learned features.

Backfill loop (from `interlatent/analysis/train/transcoder_pipeline.py`):
```python
latent_layer = f"latent:{self.layer}"
for idx, val in enumerate(z):
    self.db.write_event(ActivationEvent(run_id=run_id, step=step, layer=latent_layer, ...))
```

### 5.3 SAEPipeline
Standard sparse autoencoder with backfilled `latent_sae:{layer}` activations.

---

## 6) Analysis and Visualization

Primary tooling:
- `analysis/vis/summary.py`: layer listings, activation counts.
- `analysis/vis/search.py`: filter activations by token/prompt/layer.
- `analysis/vis/plot.py`: trace plots over tokens.
- `analysis/vis/diff.py`: compare two runs or subsets.

**Scaling tip:** Prefer block‑wise operations on `fetch_vectors` or `get_block` for large runs. Avoid per‑channel `ActivationEvent` expansion unless needed.

---

## 7) Performance Knobs

**Storage (row backend):**
- `LATENTDB_ACTIVATION_DTYPE`: `float16` or `float32`.
- `LATENTDB_CHUNK_ROWS`: row chunk size (8k–64k typical).
- `LATENTDB_MAX_CHANNELS`: hidden width, required for row backend.

**Collector:**
- `layer_indices`: number of layers captured.
- `max_channels`: number of channels captured.
- `batch_size`: prompts per forward pass (VRAM bound).
- `max_new_tokens`: affects sequence length.

---

## 8) End‑to‑End Execution (LLM)

1. Build labeled prompts (`PromptDataset`).
2. Run `LLMCollector` with desired layers/channels.
3. Train probes / transcoders / SAEs.
4. Visualize or search across latents.

The core design principle is: **store dense rows once, normalize metadata, and reconstruct events only on demand**.
