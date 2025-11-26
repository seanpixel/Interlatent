# Interlatent
Interpret RL policies and HuggingFace LLMs by collecting activations, training sparse latents (transcoders/SAEs), probing, and visualizing what fires on which tokens.

## What’s new
- LLM workflows: token-level collection with metrics, transcoder + SAE pipelines that backfill latent activations into the DB, and linear probes over LLM hidden states.
- Datasets: activation vectors (per token/step), pre/post pairs, and probe datasets that preserve prompt/token metadata.
- Visualization: quick summaries, targeted searches, and per-token plots that work for latent layers (`latent:`) and SAE latents (`latent_sae:`).

## Quickstart
1) Collect activations
```bash
# LLM (defaults to SmolLM-360M unless LLM_MODEL is set)
PYTHONPATH=. python tests/llm_workflow_demo.py            # dummy model demo
RUN_LLM_REAL=1 PYTHONPATH=. python tests/llm_real_model_demo.py  # real HF model demo

# RL
python tests/test_transcoder.py   # SB3 CartPole smoke; collects and trains a transcoder
```

2) Train latents / probes
- Transcoder: `from interlatent.analysis.train import TranscoderPipeline; TranscoderPipeline(db, "llm.layer.-1", k=8).run()`
- SAE: `from interlatent.analysis.train import SAEPipeline; SAEPipeline(db, "llm.layer.-1", k=8).run()`
- Linear probe: `from interlatent.analysis.train import train_linear_probe; train_linear_probe(db, layer="llm.layer.-1", target_key="token_id")`
- Prompt datasets + labels: use `from interlatent.llm import PromptDataset`; build with texts + labels (e.g., benign/malignant) and pass to `LLMCollector(..., prompt_context_fn=ds.prompt_context_fn(), token_metrics_fn=ds.token_metrics_fn("label"))` while using `prompts=ds.texts`.

3) Inspect & plot
- Summary/list layers: `python -m interlatent.vis.summary latents.db --list-layers --layer-prefix latent:`
- Search activations: `python -m interlatent.vis.search latents.db --layer-prefix latent: --token-like sky --top 20`
- Plot per-token: `python -m interlatent.vis.plot latents.db --layer latent:llm.layer.-1 --channel 0 --prompt-index 0`
- Plot across prompts: add `--all-prompts`

## Key Modules
- Collection: `interlatent.collectors.LLMCollector`, `GymCollector` (RL) write `ActivationEvent`s into SQLite.
- Datasets: `ActivationVectorDataset`, `ActivationPairDataset`, `LinearProbeDataset`.
- Training: `TranscoderPipeline`, `SAEPipeline`, `train_linear_probe`.
- Models: `LinearProbe`, `LinearTranscoder`, `SparseAutoencoder`.
- DB façade: `interlatent.api.LatentDB` (SQLite by default).
- Visualization: `interlatent.vis.summary`, `interlatent.vis.search`, `interlatent.vis.plot`.

## Notes
- LLM collection preserves `prompt_index`/`token_index` so latents backfilled by the pipelines align to tokens.
- Transcoder/SAE backfills write new layers `latent:{layer}` and `latent_sae:{layer}` so stats/correlations and plots work the same as base layers.
- Scripts are lightweight demos; adjust epochs/k for real training.

## Future
- Better docs, richer metrics for LLMs, and UI for browsing latent-token interactions. PRs welcome.
