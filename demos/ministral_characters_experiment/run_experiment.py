"""
Character-style moral dilemma experiment (Ministral-3) with Interlatent.

Pipeline:
  1) Load/prepare prompt dataset (character rewrites of daily_dilemmas).
  2) Run LLMCollector on a chosen layer to capture activations + completions.
  3) Train a linear probe to predict character labels from activations.
  4) Train Transcoder + SAE to surface latent features; report shapes/counts.
  (No ablation step here.)

Usage (example):
  RUN_MINISTRAL3=1 HF_TRUST_REMOTE_CODE=1 PYTHONPATH=. \\
    python demos/ministral_characters_experiment/run_experiment.py \\
    --model mistralai/Ministral-3-14B-Instruct-2512 \\
    --layer llm.layer.20 \\
    --db latents_character_dilemmas.db \\
    --csv data/character_dilemmas.csv

Notes:
  - Expects a CSV built via build_dataset.py (or similar) with text/label cols.
  - Stores completions in the DB as artifacts for later inspection.
  - Adjust k/epochs/batch sizes to trade off speed vs fidelity.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.analysis.dataset import LinearProbeDataset, PromptDataset
from interlatent.analysis.train.linear_probe_trainer import train_linear_probe
from interlatent.analysis.train.transcoder_pipeline import TranscoderPipeline
from interlatent.analysis.train.sae_pipeline import SAEPipeline


def load_model_and_tokenizer(model_id: str, trust_remote_code: bool):
    from transformers import AutoConfig, Mistral3ForConditionalGeneration, MistralCommonBackend

    tok = MistralCommonBackend.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    llm = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        config=config,
        torch_dtype=dtype,
        device_map={"": device},
    )
    return tok, llm, device, config


def collect(
    db: LatentDB,
    tok,
    llm,
    dataset: PromptDataset,
    layers: list[int],
    device: str,
    max_channels: int | None,
    log_every_prompts: int,
):
    collector = LLMCollector(
        db,
        layer_indices=layers,
        max_channels=max_channels,
        device=device,
        prompt_context_fn=dataset.prompt_context_fn(),
        token_metrics_fn=dataset.token_metrics_fn(metric_name="prompt_label"),
        log_every_prompts=log_every_prompts,
    )
    print("[collect] Starting collection...")
    collector.run(
        llm,
        tok,
        prompts=dataset.texts,
        max_new_tokens=64,
        batch_size=1,
    )
    db.flush()
    print("[collect] Done.")


def resolve_db_uri(db_arg: str) -> tuple[str, Path | None]:
    if "://" in db_arg:
        if db_arg.startswith(("sqlite:///", "file:///", "hdf5:///", "h5:///", "hdf5v2:///", "hdf5row:///")):
            path = Path(db_arg.split(":///", 1)[1])
            return db_arg, path
        return db_arg, None
    path = Path(db_arg)
    return f"sqlite:///{path}", path


def run(args):
    if os.environ.get("RUN_MINISTRAL3") != "1":
        print("Set RUN_MINISTRAL3=1 to run (downloads weights); skipping.")
        return

    dataset = PromptDataset.from_csv(args.csv, text_col="text", label_col="label")
    print(f"Loaded {len(dataset.examples)} prompts from {args.csv}")

    trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1") == "1"
    print("[load] Loading model and tokenizer...")
    tok, llm, device, config = load_model_and_tokenizer(args.model, trust_remote_code)
    print(f"[load] Model on {device}")

    if args.all_channels:
        hidden_size = int(
            getattr(config, "hidden_size", 0)
            or getattr(config, "d_model", 0)
            or getattr(config, "n_embd", 0)
            or getattr(config, "dim", 0)
            or getattr(config, "model_dim", 0)
            or 0
        )
        if hidden_size <= 0 and hasattr(llm, "get_input_embeddings"):
            emb = llm.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                hidden_size = int(getattr(emb.weight, "shape", [0, 0])[1] or 0)
        if hidden_size <= 0:
            raise ValueError("Model config missing hidden_size; set --max_channels manually.")
        args.max_channels = hidden_size
    os.environ["LATENTDB_MAX_CHANNELS"] = str(args.max_channels)

    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]
    else:
        num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        if num_layers <= 0:
            raise ValueError("Model config missing num_hidden_layers; set --layers manually.")
        picks = [0, num_layers // 3, (2 * num_layers) // 3, num_layers - 1]
        layer_indices = []
        for idx in picks:
            if idx not in layer_indices:
                layer_indices.append(idx)
        if len(layer_indices) < 4:
            for idx in range(num_layers):
                if idx not in layer_indices:
                    layer_indices.append(idx)
                if len(layer_indices) >= 4:
                    break
    print(f"[setup] capturing layers={layer_indices} max_channels={args.max_channels}")
    db_uri, db_path = resolve_db_uri(args.db)
    if db_path is not None and db_path.exists():
        db_path.unlink()
    db = LatentDB(db_uri)

    collect(db, tok, llm, dataset, layer_indices, device, args.max_channels, args.log_every_prompts)
    primary_layer = f"llm.layer.{layer_indices[0]}"
    base_x, _ = db.fetch_vectors(layer=primary_layer)
    base_rows = int(base_x.shape[0]) if base_x.size else 0
    print(f"[collector] captured {base_rows} activations for layer {primary_layer}")

    lp_ds = LinearProbeDataset(db, layer=primary_layer, target_key="prompt_label")
    print("[probe] Training linear probe...")
    probe = train_linear_probe(
        db, layer=args.layer, target_key="prompt_label", epochs=args.probe_epochs, lr=1e-3, batch_size=16
    )
    print(f"[linear probe] samples={len(lp_ds)}, weight_shape={tuple(probe.proj.weight.shape)}")

    print("[transcoder] Training...")
    pipe = TranscoderPipeline(db, primary_layer, k=args.transcoder_k, epochs=args.transcoder_epochs)
    trainer = pipe.run()
    latent_x, _ = db.fetch_vectors(layer=f"latent:{primary_layer}")
    latent_rows = int(latent_x.shape[0]) if latent_x.size else 0
    print(f"[transcoder] latent rows={latent_rows}, encoder_shape={tuple(trainer.T.weight.shape)}")

    print("[sae] Training...")
    sae_pipe = SAEPipeline(db, primary_layer, k=args.sae_k, epochs=args.sae_epochs)
    sae_model = sae_pipe.run()
    sae_latents_x, _ = db.fetch_vectors(layer=f"latent_sae:{primary_layer}")
    sae_latent_rows = int(sae_latents_x.shape[0]) if sae_latents_x.size else 0
    print(f"[sae] latent rows={sae_latent_rows}, encoder_shape={tuple(sae_model.encoder.weight.shape)}")

    db.close()
    print(f"Done. DB at {db_uri}, completions in completions_character_dilemmas.jsonl.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="CSV with text/label columns (character rewrites).")
    ap.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--layers", type=str, default="", help="Comma-separated layer indices (e.g., '0,8,16,24').")
    ap.add_argument("--db", type=str, default="latents_character_dilemmas.db")
    ap.add_argument(
        "--max_channels",
        type=int,
        default=int(os.environ.get("MAX_CHANNELS", "128")),
        help="Limit the number of channels to record per layer (default from MAX_CHANNELS or 128).",
    )
    ap.add_argument("--all-channels", action="store_true", help="Capture all channels from the model.")
    ap.add_argument("--log-every-prompts", type=int, default=1, help="Log progress every N prompts.")
    ap.add_argument("--probe_epochs", type=int, default=1)
    ap.add_argument("--transcoder_k", type=int, default=8)
    ap.add_argument("--transcoder_epochs", type=int, default=1)
    ap.add_argument("--sae_k", type=int, default=8)
    ap.add_argument("--sae_epochs", type=int, default=1)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
