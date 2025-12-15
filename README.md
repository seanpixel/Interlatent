# Interlatent

Interlatent is a lightweight interpretability toolkit where you can: collect activations with context, attach labels, learn sparse latents (transcoders/SAEs) and probes, and quickly see which tokens or states drive them. The goal is to allow new independent researchers / engineers to dabble with understanding their models. It uses SQLite and small scripts by design, aimed at small/medium-scale experiments.

## Use It Quickly
- LLM workflow demo (no downloads): `PYTHONPATH=. python scripts/demos/ministral/llm_workflow_demo.py`
- Real HF model demo: `RUN_LLM_REAL=1 PYTHONPATH=. python scripts/demos/ministral/llm_real_model_demo.py` (set `LLM_MODEL` to override)
- Prompt labeling demo: `PYTHONPATH=. python scripts/demos/ministral/prompt_dataset_demo.py`
- Ministral character experiment: see `scripts/demos/ministral/character_ablations/` (build dataset, run experiment, visualize)
- Train programmatically:
  - Transcoder: `from interlatent.analysis.train import TranscoderPipeline; TranscoderPipeline(db, "llm.layer.8", k=8).run()`
  - SAE: `from interlatent.analysis.train import SAEPipeline; SAEPipeline(db, "llm.layer.8", k=8).run()`
  - Linear probe: `from interlatent.analysis.train import train_linear_probe; train_linear_probe(db, layer="llm.layer.8", target_key="prompt_label")`
  - Prompt labels: `from interlatent.llm import PromptDataset`; pass `prompt_context_fn` and `token_metrics_fn` into `LLMCollector`.

## Inspect Latents Fast
- Summary/list layers: `python -m interlatent.vis.summary latents.db --list-layers --layer-prefix latent:`
- Search activations: `python -m interlatent.vis.search latents.db --layer-prefix latent: --token-like sky --top 20`
- Plot per-token: `python -m interlatent.vis.plot latents.db --layer latent:llm.layer.8 --channel 0 --prompt-index 0`
- Latent diffs: `python -m interlatent.vis.diff latents.db --layer-prefix latent: --channels 0 1 --prompt-like-a A --prompt-like-b B`

## Learn More
See [GUIDE.md](GUIDE.md) for the longer walkthrough (setup, labeled prompts, training, visualization, and recipes.
