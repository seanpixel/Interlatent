Interlatent is a lightweight interpretability toolkit: collect activations with context, attach labels, learn sparse latents (transcoders/SAEs) and probes, and quickly see which tokens or states drive them. The philosophy is to stay close to the model’s internal signals—log, label, compress, and visualize—so you can iterate on ideas rather than build production pipelines. It uses SQLite and small scripts by design, aimed at small/medium-scale experiments.

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
See [GUIDE.md](GUIDE.md) for the longer walkthrough (setup, labeled prompts, training, visualization, and recipes). keep debugging and feedback welcome.
2. `LLMCollector` writes activations with `context["metrics"]["label"]`.
3. `LinearProbeDataset` extracts `(activation_vector, label)` pairs from that metric.
4. `train_linear_probe` trains regression/classification heads.
5. Latent backfills (transcoder/SAE) keep prompt/token metadata so you can search/plot latents by token.

---

## Practical Tips
- **Channel caps**: Use `max_channels` in `LLMCollector` to avoid massive DBs on large models.
- **Prompt lengths**: `LLMCollector` respects attention masks; ensure padding tokens are set on the tokenizer.
- **Fresh DBs**: Delete old SQLite files between runs to avoid mixing schema/step conventions.
- **Batching**: `batch_size` in `LLMCollector` controls prompt batching; start with 1–2 to keep memory low.
- **Stats/correlations**: Run `db.compute_stats(min_count=...)` before inspecting correlations.
- **Latent layers**: Backfilled layers are `latent:{layer}` and `latent_sae:{layer}`; use these in plotting/searching.

---

## Limitations / Non-goals
- **Scale**: SQLite is great for small/medium runs; not intended for billion-token traces. For very large runs, you’ll need a different backend.
- **Latency**: Collectors are synchronous and unoptimized for throughput; they are research-oriented, not production pipelines.
- **Model coverage**: Only causal LMs are handled in `LLMCollector`; encoder-only or seq2seq models would need small adaptations.
- **Metrics richness**: Token-level metrics require user-provided functions; no automatic labeling or safety classifiers are bundled.
- **No UI dashboard**: Visualization is CLI-based; there is no web UI baked in.
- **Hook assumptions**: GymCollector assumes discrete actions and uses simple hook points; more complex policies may need custom hooks.
- **No gradient-based interpretability**: The toolkit focuses on activation logging and sparse bottlenecks, not on gradients/attribution methods.

---

## Extending / Custom Ideas
- Add your own metric functions to `LLMCollector` (e.g., entropy, toxicity scores) and surface them via `token_metrics_fn`.
- Swap dummy LLM for your HF model; adjust `max_channels` and layer indices as needed.
- For sequence-level probes, aggregate token activations (e.g., mean-pool per prompt) before training probes.
- Build lightweight dashboards by wrapping `vis.search` and `vis.plot` into a notebook.

---

## Minimal Code Skeletons
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
