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
    python scripts/demos/ministral/character_ablations/run_experiment.py \\
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
import json
import os
from pathlib import Path
from typing import List

import torch

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.llm.prompt_dataset import PromptDataset
from interlatent.analysis.datasets import LinearProbeDataset
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
    return tok, llm, device


def collect(db: LatentDB, tok, llm, dataset: PromptDataset, layer: str, device: str):
    # Store completions as an artifact to inspect style differences
    completions_path = Path("completions_character_dilemmas.jsonl")
    with completions_path.open("w") as outf:
        def on_output(prompt_idx: int, prompt: str, output: str):
            ex = dataset.examples[prompt_idx]
            json.dump(
                {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "label": ex.label,
                    "meta": ex.meta,
                    "output": output,
                },
                outf,
            )
            outf.write("\n")

        collector = LLMCollector(
            db,
            layer_indices=[int(layer.split(".")[-1]) if layer.startswith("llm.layer.") else layer],
            max_channels=128,
            device=device,
            prompt_context_fn=dataset.prompt_context_fn(),
            token_metrics_fn=dataset.token_metrics_fn(metric_name="prompt_label"),
            on_output=on_output,
        )
        collector.run(
            llm,
            tok,
            prompts=dataset.texts,
            max_new_tokens=64,
            batch_size=1,
        )
    db.add_artifact("completions", str(completions_path))


def run(args):
    if os.environ.get("RUN_MINISTRAL3") != "1":
        print("Set RUN_MINISTRAL3=1 to run (downloads weights); skipping.")
        return

    dataset = PromptDataset.from_csv(args.csv, text_col="text", label_col="label")
    print(f"Loaded {len(dataset.examples)} prompts from {args.csv}")

    trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1") == "1"
    tok, llm, device = load_model_and_tokenizer(args.model, trust_remote_code)

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    db = LatentDB(f"sqlite:///{db_path}")

    collect(db, tok, llm, dataset, args.layer, device)
    base_rows = len(db.fetch_activations(layer=args.layer))
    print(f"[collector] captured {base_rows} activations for layer {args.layer}")

    lp_ds = LinearProbeDataset(db, layer=args.layer, target_key="prompt_label")
    probe = train_linear_probe(
        db, layer=args.layer, target_key="prompt_label", epochs=args.probe_epochs, lr=1e-3, batch_size=16
    )
    print(f"[linear probe] samples={len(lp_ds)}, weight_shape={tuple(probe.proj.weight.shape)}")

    pipe = TranscoderPipeline(db, args.layer, k=args.transcoder_k, epochs=args.transcoder_epochs)
    trainer = pipe.run()
    latent_events = db.fetch_activations(layer=f"latent:{args.layer}")
    print(f"[transcoder] latent rows={len(latent_events)}, encoder_shape={tuple(trainer.T.weight.shape)}")

    sae_pipe = SAEPipeline(db, args.layer, k=args.sae_k, epochs=args.sae_epochs)
    sae_model = sae_pipe.run()
    sae_latents = db.fetch_activations(layer=f"latent_sae:{args.layer}")
    print(f"[sae] latent rows={len(sae_latents)}, encoder_shape={tuple(sae_model.encoder.weight.shape)}")

    db.close()
    print(f"Done. DB at {db_path}, completions in completions_character_dilemmas.jsonl.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="CSV with text/label columns (character rewrites).")
    ap.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--layer", type=str, default="llm.layer.20")
    ap.add_argument("--db", type=str, default="latents_character_dilemmas.db")
    ap.add_argument("--probe_epochs", type=int, default=1)
    ap.add_argument("--transcoder_k", type=int, default=8)
    ap.add_argument("--transcoder_epochs", type=int, default=1)
    ap.add_argument("--sae_k", type=int, default=8)
    ap.add_argument("--sae_epochs", type=int, default=1)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
