# How Interlatent Works

This document is an internal, technical overview of the system architecture and data flow. It starts with a birds‑eye view and then zooms into storage, collection, datasets, training, and analysis/visualization.

---

## 1) Birds‑Eye View
Interlatent is built around a simple loop:

1. **Collect activations** from models (LLM or RL policy).
2. **Persist activations + metadata** in a scalable storage backend.
3. **Build datasets** for probing or sparse autoencoders from stored activations.
4. **Train models** (linear probes, transcoders, SAEs).
5. **Analyze / visualize** latent features and token traces.

The storage backend is the spine: all downstream tasks operate on it, either by reading per‑step vectors (`fetch_vectors` / `get_block`) or by materializing per‑channel events when needed.

---

## 2) Storage: Backends and Data Layout

### 2.1 SQLite Backend (small/medium runs)
**Goal:** simple single‑file DB for quick experiments.

Layout:
- `activations` table: one row per `(run_id, step, layer)` with a dense JSON vector.
- `stats`, `metric_sums`: aggregated stats and correlations.
- `explanations`, `artifacts`: auxiliary tables.

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

Fast APIs:
- `get_block(run_id, layer, start, end)` returns `(x, idx)` without event materialization.
- `fetch_vectors(layer)` returns all vectors + metadata for a layer.

Slow APIs:
- `iter_events(...)` reconstructs `ActivationEvent` objects only for slices or selected channels.

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

**Slow path (legacy/small runs):**
- `fetch_activations` returns per‑channel `ActivationEvent`s.
- Datasets group events by step/prompt/token and reconstruct vectors.

Core datasets:
- `ActivationVectorDataset`: per‑step activation vectors.
- `ActivationPairDataset`: pre/post pairs (RL use cases).
- `LinearProbeDataset`: vectors + targets from `context["metrics"]`.

---

## 5) Training Pipelines

### 5.1 Linear Probe
Fits a simple linear model from activations to labels (classification or regression). Labels come from `context["metrics"]` in stored events.

### 5.2 TranscoderPipeline
Sparse bottleneck autoencoder that projects activations into `latent:{layer}`. This backfills new activations into the DB, enabling downstream analysis of learned features.

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
