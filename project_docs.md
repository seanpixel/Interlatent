• What It Does

  - Tooling to sniff activations from RL models, store them, train sparse linear “transcoder” bottlenecks, and correlate learned latents with environment
    metrics for interpretability (see README.md)

  Core Modules

  - interlatent/api/latent_db.py: User-facing façade; selects a storage backend from URI (SQLite by default), validates events, computes stats async, fetches
    activations/timelines, iterates stat blocks, flushes/close.
  - interlatent/storage/sqlite.py: SQLite backend; creates tables for activations, running metric sums, stats, explanations, artifacts. write_event streams
    activations + metric tallies, compute_stats aggregates moments and metric correlations, fetch_events returns flattened activation series. Stores
    artifacts and explanations. (Debug prints remain.)
  - interlatent/schema.py: Pydantic models defining contracts (RunInfo, ActivationEvent, StatBlock with correlations, Explanation, Artifact).
  - interlatent/hooks.py: Torch forward/pre/post hook contexts. TorchHook streams activations (flattened per channel) into LatentDB during a with block.
    PrePostHookCtx records both pre/post values for layers, tagging them as {layer}:pre/{layer}:post.
  - interlatent/collectors/gym_collector.py: Runs a model in a Gym-like env; optional metric functions; registers hooks; converts model outputs to discrete actions; streams
    ActivationEvents and metric context into DB. Returns RunInfo.
  - interlatent/collectors/llm_collector.py: Runs a HuggingFace causal LM over prompts, capturing per-token activations into LatentDB.
  - interlatent/metrics.py: Metric protocol + helpers (LambdaMetric, EpisodeAccumulator) that emit one scalar per step.
  - interlatent/analysis/dataset/activation_pair_dataset.py: Builds paired datasets (pre, post) for a layer using logged activations; each sample is a per-step vector of channel sums.
  - interlatent/analysis/dataset/linear_probe_dataset.py: Builds (activation vector, target) pairs by reading metrics from activation contexts for linear probe training.
  - interlatent/analysis/train/transcoder_trainer.py: Minimal linear encoder/decoder trainer with L1 sparsity.
  - interlatent/analysis/train/transcoder_pipeline.py: Orchestrates training a transcoder for one layer: train, save artifact, then backfill latent activations (layer name
    latent:{layer}) using the trained encoder so they can be correlated like normal channels.
  - interlatent/analysis/train/online_sae_trainer.py: Streaming SAE training loop that avoids writing activations to disk.
  - interlatent/analysis/models/linear_transcoder.py: Simple relu bottleneck transcoder module (encoder/decoder).

  Utilities & Scripts

  - interlatent/utils/logging.py, hook_utils.py (hook cleanup).
- Scripts: demos under `demos/basics/` (e.g., `run_transcoder.py` to run collector + transcoder training on a chosen SB3 policy)
    env/metrics.

  Tests/Examples

  - tests/test_end_to_end.py: CartPole smoke test using GymCollector + TorchHook + stats (references a non-existent LatentDB.describe—will fail).
  - tests/test_transcoder.py: Uses SB3 PPO policy, collects activations, trains transcoder, checks artifact row and that correlations exist.
  - tests/test_metrics.py, tests/test_correlations.py: Example scripts (no assertions) showing metric logging and correlation inspection.
  - latents.db: Example SQLite DB artifact in repo root.

  Notable Gaps/Rough Edges

  - LatentDB lacks describe method referenced in tests/test_end_to_end.py; test currently fails.
  - StorageBackend interface duplicated iter_statblocks signatures; SQLite backend exposes debug prints and direct _store._conn reaches from facade.
  - Logging format in utils/logging.py has a stray space (“% (asctime)s…”) that likely breaks timestamps.
  - Minimal validation/safety around action spaces and only covers discrete envs; no batching/perf tuning yet.

  Typical Flow

  1. Instantiate LatentDB("sqlite:///path/to/latents.db").
  2. Define metrics (LambdaMetric/EpisodeAccumulator) and layers to hook.
  3. Run GymCollector.run(model, env, steps=...) to populate activations + metrics.
  4. Optionally run TranscoderPipeline(db, layer, k=..., epochs=...).run() to train and backfill latent activations.
  5. Call db.compute_stats(min_count=...) then iterate db.iter_statblocks(...) for moments/correlations; artifacts/explanations can be added via backend.
