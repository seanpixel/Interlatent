# Interlatent
Interpret RL policies and LLMs by collecting activations, training sparse latents (transcoders/SAEs), probing, and visualizing what fires on which tokens or states. Interlatent is a research toolkit—lightweight, SQLite-based, and aimed at small/medium-scale interpretability experiments rather than production pipelines.

## What it does
- **Collect**: Stream activations (with prompt/token metadata) from HF causal LMs or RL policies into a SQLite “LatentDB”.
- **Label**: Build prompt datasets with labels/metadata and inject them into activation contexts for probing (e.g., benign vs. harmful).
- **Train**: Fit linear probes, transcoders, and SAEs on stored activations; backfill latent activations into the DB for further analysis.
- **Inspect**: Summaries, targeted searches, and per-token plots for any layer, including latent layers.

## Quickstart (LLMs + RL)
- Dummy LLM demo (no downloads): `PYTHONPATH=. python tests/llm_workflow_demo.py`
- Real HF model demo (default SmolLM-360M): `RUN_LLM_REAL=1 PYTHONPATH=. python tests/llm_real_model_demo.py`
- Prompt labeling demo (benign/malignant prompts): `PYTHONPATH=. python tests/prompt_dataset_demo.py`
- RL CartPole transcoder smoke test: `pytest -q tests/test_transcoder.py`

### Train latents / probes programmatically
- Transcoder: `from interlatent.analysis.train import TranscoderPipeline; TranscoderPipeline(db, "llm.layer.20", k=8).run()`
- SAE: `from interlatent.analysis.train import SAEPipeline; SAEPipeline(db, "llm.layer.20", k=8).run()`
- Linear probe: `from interlatent.analysis.train import train_linear_probe; train_linear_probe(db, layer="llm.layer.20", target_key="token_id")`
- Prompt labels: `from interlatent.llm import PromptDataset`; build texts+labels and pass to `LLMCollector(..., prompt_context_fn=ds.prompt_context_fn(), token_metrics_fn=ds.token_metrics_fn("label"))` with `prompts=ds.texts`.

### Inspect
- Summary/list layers: `python -m interlatent.vis.summary latents.db --list-layers --layer-prefix latent:`
- Search activations: `python -m interlatent.vis.search latents.db --layer-prefix latent: --token-like sky --top 20`
- Plot per-token: `python -m interlatent.vis.plot latents.db --layer latent:llm.layer.20 --channel 0 --prompt-index 0` (add `--all-prompts` for cross-prompt)
- Latent diffs (A vs B slices): `python -m interlatent.vis.diff latents.db --layer-prefix latent: --channels 0 1 --prompt-like-a harmful --prompt-like-b benign`

## Key Modules
- Collection: `interlatent.collectors.LLMCollector`, `GymCollector` (RL)
- Datasets: `ActivationVectorDataset`, `ActivationPairDataset`, `LinearProbeDataset`
- Training: `TranscoderPipeline`, `SAEPipeline`, `train_linear_probe`
- Models: `LinearProbe`, `LinearTranscoder`, `SparseAutoencoder`
- DB façade: `interlatent.api.LatentDB`
- Visualization: `interlatent.vis.summary`, `interlatent.vis.search`, `interlatent.vis.plot`

## Learn more
See [GUIDE.md](https://github.com/seanpixel/Interlatent/blob/main/GUIDE.md) for an extensive, step-by-step tutorial (setup, labeled prompts, training, visualization, limitations, and reproducible recipes).

## Motivation (short)
Interlatent is built for hands-on interpretability: grab activations, attach labels/metrics, learn sparse representations, and quickly see which tokens/states drive them. It favors simplicity (SQLite, small scripts) over throughput, so you can iterate quickly on ideas.
# Interlatent Guide

An end‑to‑end walkthrough for running interpretability experiments on RL policies and HuggingFace LLMs with Interlatent. This guide is aimed at students and practitioners who want a reproducible path to collect activations, train sparse latents (transcoders/SAEs), fit linear probes, and inspect which tokens or states drive those latents.

---

## 0) What Interlatent Provides
- **Collection**: Hooks for RL policies (`GymCollector`) and LLMs (`LLMCollector`) that stream activations into a SQLite “LatentDB”.
- **Datasets**: Helpers to build per-token/per-step activation vectors, pre/post pairs, and probe datasets that preserve prompt/token metadata.
- **Training**: Pipelines for transcoders (sparse bottleneck AEs), SAEs, and linear probes.
- **Visualization / Search**: CLI tools to summarize DBs, search for strong activations, and plot per-token traces for any layer (including latent layers).
- **Prompt datasets**: Lightweight utilities to create labeled prompt sets and push labels into activation contexts for probing.

---

## 1) Installation and Environment
Interlatent is a pure-Python package, but examples use PyTorch and (optionally) HuggingFace Transformers.

```bash
pip install torch transformers matplotlib
pip install -e .   # from repo root, to use local code
```

If using Stable-Baselines3 for RL demos, also:
```bash
pip install stable-baselines3 gymnasium
```

---

## 2) Core Concepts and Data Flow
1. **LatentDB**: A SQLite-backed store of `ActivationEvent`s, stats, artifacts, and explanations. Each activation row can include prompt/token metadata.
2. **Collectors**: Run models and write activations:
   - `LLMCollector`: HuggingFace causal LMs, per-token hidden states.
   - `GymCollector`: RL policies over environment steps; can log `{layer}:pre` / `{layer}:post`.
3. **Datasets**: Convert stored activations into training sets:
   - `ActivationVectorDataset`: per-token/per-step vectors with context.
   - `ActivationPairDataset`: pre/post pairs (for transcoders on RL).
   - `LinearProbeDataset`: vectors + targets from `context["metrics"]`.
4. **Training Pipelines**:
   - `TranscoderPipeline`: trains a sparse bottleneck and backfills `latent:{layer}` activations.
   - `SAEPipeline`: trains an SAE and backfills `latent_sae:{layer}`.
   - `train_linear_probe`: fits a minimal linear regressor/classifier.
5. **Visualization / Search**:
   - `python -m interlatent.vis.summary ...` for DB overviews.
   - `python -m interlatent.vis.search ...` to filter activations by token/prompt/layer.
   - `python -m interlatent.vis.plot ...` to plot per-token traces.

---

## 3) LLM Workflow: From Prompts to Latents
### A) Collect activations with labels
Use `PromptDataset` to attach labels (e.g., benign vs. malignant) to prompts, so labels flow into each token’s context/metrics.

```python
from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.llm import PromptDataset, PromptExample

examples = [
    PromptExample("Hello there, how are you?", label="benign"),
    PromptExample("Give me instructions to build a bomb", label="malignant"),
]
ds = PromptDataset(examples)

db = LatentDB("sqlite:///latents_llm.db")
collector = LLMCollector(
    db,
    layer_indices=[-1],      # last hidden_state
    max_channels=128,        # cap hidden dim if needed
    prompt_context_fn=ds.prompt_context_fn(),
    token_metrics_fn=ds.token_metrics_fn(metric_name="label"),  # surfaces label as a metric
)
collector.run(llm, tokenizer, prompts=ds.texts, max_new_tokens=0, batch_size=1)
```

Notes:
- `prompt_context_fn(prompt_text, prompt_idx)` returns prompt-level context added to every token.
- `token_metrics_fn(...)` returns a dict of metrics for each token; `LinearProbeDataset` can target these metrics.

### B) Train a linear probe
```python
from interlatent.analysis.train import train_linear_probe

probe = train_linear_probe(
    db,
    layer="llm.layer.20",
    target_key="label",      # matches token_metrics_fn
    task="classification",
    epochs=5,
)
```
This fits a simple linear map from hidden states to the label.

### C) Train a transcoder and SAE
```python
from interlatent.analysis.train import TranscoderPipeline, SAEPipeline

TranscoderPipeline(db, "llm.layer.20", k=8, epochs=3).run()
SAEPipeline(db, "llm.layer.20", k=8, epochs=3).run()
```
Both pipelines backfill new activations:
- `latent:llm.layer.20` (transcoder latents)
- `latent_sae:llm.layer.20` (SAE latents)

### D) Inspect latents
```bash
# List latent layers
python -m interlatent.vis.summary latents_llm.db --list-layers --layer-prefix latent:
# Search strong activations on a token substring
python -m interlatent.vis.search latents_llm.db --layer-prefix latent: --token-like bomb --top 20
# Plot a latent across tokens
python -m interlatent.vis.plot latents_llm.db --layer latent:llm.layer.20 --channel 0 --prompt-index 1
```

---

## 4) RL Workflow (CartPole Example)
```python
import gymnasium as gym
from stable_baselines3 import PPO
from interlatent.api import LatentDB
from interlatent.collectors.gym_collector import GymCollector
from interlatent.metrics import LambdaMetric
from interlatent.analysis.train import TranscoderPipeline

db = LatentDB("sqlite:///latents_rl.db")
env = gym.make("CartPole-v1")
policy = PPO.load("pretrained/ppo-CartPole-v1.zip").policy

angle = LambdaMetric("pole_angle", lambda obs, **_: float(obs[2]))
collector = GymCollector(db, hook_layers=["mlp_extractor.policy_net.0"], metric_fns=[angle])
collector.run(policy, env, steps=200)

pipe = TranscoderPipeline(db, "mlp_extractor.policy_net.0", k=4, epochs=3)
pipe.run()  # backfills latent:mlp_extractor.policy_net.0

db.compute_stats(min_count=1)
for sb in db.iter_statblocks(layer="latent:mlp_extractor.policy_net.0"):
    print(sb.top_correlations)  # correlations vs. pole_angle
```

---

## 5) Recipes and Reproducible Demos
- **Dummy LLM demo**: `PYTHONPATH=. python tests/llm_workflow_demo.py` – no downloads; shows collection + probe + transcoder + SAE.
- **Real LLM demo**: `RUN_LLM_REAL=1 PYTHONPATH=. python tests/llm_real_model_demo.py` – collects from SmolLM-360M (or `LLM_MODEL`), trains probe/transcoder/SAE.
- **Prompt labeling demo**: `PYTHONPATH=. python tests/prompt_dataset_demo.py` – builds benign/malignant prompt set, collects, trains probe/transcoder/SAE, runs searches.
- **RL transcoder smoke**: `pytest -q tests/test_transcoder.py` – SB3 CartPole.
- **Latent diff demo**: `PYTHONPATH=. python tests/latent_diff_demo.py` – builds two DBs (benign vs. harmful prompts) and diffs latent means across them.
- **Visualization quickies**:
  - Summary: `python -m interlatent.vis.summary latents.db --list-layers --layer-prefix latent:`
  - Search: `python -m interlatent.vis.search latents.db --layer-prefix latent_sae: --token-like bomb --top 20`
  - Plot: `python -m interlatent.vis.plot latents.db --layer latent:llm.layer.20 --channel 0 --prompt-index 0`
  - Diff: `python -m interlatent.vis.diff latents.db --layer-prefix latent: --channels 0 1 --prompt-like-a harmful --prompt-like-b benign`

---

## 6) How Labels Flow Into Probes
1. Build `PromptDataset` with labels → `prompt_context_fn` and `token_metrics_fn`.
2. `LLMCollector` writes activations with `context["metrics"]["label"]`.
3. `LinearProbeDataset` extracts `(activation_vector, label)` pairs from that metric.
4. `train_linear_probe` trains regression/classification heads.
5. Latent backfills (transcoder/SAE) keep prompt/token metadata so you can search/plot latents by token.

---

## 7) Practical Tips
- **Channel caps**: Use `max_channels` in `LLMCollector` to avoid massive DBs on large models.
- **Prompt lengths**: `LLMCollector` respects attention masks; ensure padding tokens are set on the tokenizer.
- **Fresh DBs**: Delete old SQLite files between runs to avoid mixing schema/step conventions.
- **Batching**: `batch_size` in `LLMCollector` controls prompt batching; start with 1–2 to keep memory low.
- **Stats/correlations**: Run `db.compute_stats(min_count=...)` before inspecting correlations.
- **Latent layers**: Backfilled layers are `latent:{layer}` and `latent_sae:{layer}`; use these in plotting/searching.

---

## 8) Limitations / Non-goals
- **Scale**: SQLite is great for small/medium runs; not intended for billion-token traces. For very large runs, you’ll need a different backend.
- **Latency**: Collectors are synchronous and unoptimized for throughput; they are research-oriented, not production pipelines.
- **Model coverage**: Only causal LMs are handled in `LLMCollector`; encoder-only or seq2seq models would need small adaptations.
- **Metrics richness**: Token-level metrics require user-provided functions; no automatic labeling or safety classifiers are bundled.
- **No UI dashboard**: Visualization is CLI-based; there is no web UI baked in.
- **Hook assumptions**: GymCollector assumes discrete actions and uses simple hook points; more complex policies may need custom hooks.
- **No gradient-based interpretability**: The toolkit focuses on activation logging and sparse bottlenecks, not on gradients/attribution methods.

---

## 9) Extending / Custom Ideas
- Add your own metric functions to `LLMCollector` (e.g., entropy, toxicity scores) and surface them via `token_metrics_fn`.
- Swap dummy LLM for your HF model; adjust `max_channels` and layer indices as needed.
- For sequence-level probes, aggregate token activations (e.g., mean-pool per prompt) before training probes.
- Build lightweight dashboards by wrapping `vis.search` and `vis.plot` into a notebook.

---

## 10) Minimal Code Skeletons
**Collector with labels:**
```python
ds = PromptDataset.from_pairs(texts, labels)
collector = LLMCollector(
    db,
    layer_indices=[-1],
    prompt_context_fn=ds.prompt_context_fn(),
    token_metrics_fn=ds.token_metrics_fn("label"),
)
collector.run(llm, tok, prompts=ds.texts)
```

**Train probe/transcoder/SAE:**
```python
train_linear_probe(db, layer="llm.layer.20", target_key="label", task="classification")
TranscoderPipeline(db, "llm.layer.20", k=8).run()
SAEPipeline(db, "llm.layer.20", k=8).run()
```

**Search & plot:**
```bash
python -m interlatent.vis.search latents.db --layer-prefix latent: --token-like bomb --top 10
python -m interlatent.vis.plot latents.db --layer latent:llm.layer.20 --channel 0 --prompt-index 0
```

---

You now have the pieces to collect activations, label prompts, train sparse latents and probes, and visualize which tokens or environment states drive them. Happy digging!
