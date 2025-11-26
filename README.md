# Interlatent
Interpret RL policies and HuggingFace LLMs by collecting activations, training sparse latents (transcoders/SAEs), probing, and visualizing what fires on which tokens or states. Interlatent is a research toolkit—lightweight, SQLite-based, and aimed at small/medium-scale interpretability experiments rather than production pipelines.

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

## Key Modules
- Collection: `interlatent.collectors.LLMCollector`, `GymCollector` (RL)
- Datasets: `ActivationVectorDataset`, `ActivationPairDataset`, `LinearProbeDataset`
- Training: `TranscoderPipeline`, `SAEPipeline`, `train_linear_probe`
- Models: `LinearProbe`, `LinearTranscoder`, `SparseAutoencoder`
- DB façade: `interlatent.api.LatentDB`
- Visualization: `interlatent.vis.summary`, `interlatent.vis.search`, `interlatent.vis.plot`

## Learn more
See `GUIDE.md` for an extensive, step-by-step tutorial (setup, labeled prompts, training, visualization, limitations, and reproducible recipes).

## Motivation (short)
Interlatent is built for hands-on interpretability: grab activations, attach labels/metrics, learn sparse representations, and quickly see which tokens/states drive them. It favors simplicity (SQLite, small scripts) over throughput, so you can iterate quickly on ideas.
