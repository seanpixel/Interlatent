"""
Online/streaming SAE demo on a small HF LLM (SmolLM-360M) without writing activations.

It trains an SAE on an initial prompt batch, saves the weights, then continues
training on new prompts to show how you can keep refining the same SAE.

Usage:
  PYTHONPATH=. python demos/basics/online_sae_demo.py \
    --model HuggingFaceTB/SmolLM-360M \
    --layer-idx -2 \
    --k 128 \
    --outdir artifacts/online_sae
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlatent.analysis.train.online_sae_trainer import (
    StreamingSAEConfig,
    StreamingSAETrainer,
)


def load_model_and_tokenizer(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    llm.eval()
    return tok, llm


def save_sae(sae, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder": sae.encoder.state_dict(), "decoder": sae.decoder.state_dict()},
        path,
    )
    print(f"[save] Saved SAE to {path}")


def run_training_round(
    trainer: StreamingSAETrainer,
    tok,
    llm,
    prompts: Sequence[str],
    tag: str,
    outdir: Path,
):
    print(f"[train:{tag}] prompts={len(prompts)}")
    sae = trainer.train(llm, tok, prompts)
    if sae is None:
        raise RuntimeError("SAE did not initialize; check prompts/model outputs.")
    save_sae(sae, outdir / f"sae_{tag}.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-360M")
    ap.add_argument("--layer-idx", type=int, default=-2, help="hidden_states index to train on")
    ap.add_argument("--k", type=int, default=128, help="SAE bottleneck size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--sample-tokens", type=int, default=2048)
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/online_sae"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, llm = load_model_and_tokenizer(args.model, device)

    cfg = StreamingSAEConfig(
        layer_index=args.layer_idx,
        k=args.k,
        lr=args.lr,
        l1=args.l1,
        batch_size=args.batch_size,
        sample_tokens=args.sample_tokens,
        device=device,
    )
    trainer = StreamingSAETrainer(cfg)

    prompts_phase1 = [
        "Explain how transformers work in one paragraph.",
        "List three creative uses for a paperclip.",
        "Write a short limerick about the ocean.",
        "What is overfitting in machine learning?",
    ]
    run_training_round(trainer, tok, llm, prompts_phase1, tag="phase1", outdir=args.outdir)

    prompts_phase2 = [
        "Summarize the benefits of dropout in neural networks.",
        "How does attention help sequence models?",
        "Give an analogy for gradient descent.",
        "Name two challenges in training large language models.",
    ]
    run_training_round(trainer, tok, llm, prompts_phase2, tag="phase2_continued", outdir=args.outdir)

    print(
        "\n[done] Trained an SAE online across two batches. "
        "Use the saved checkpoints to wrap layers or resume training with more data."
    )


if __name__ == "__main__":
    main()
