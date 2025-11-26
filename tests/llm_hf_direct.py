"""Smoke test: run LLMCollector directly on a HF model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector


def main():
    parser = argparse.ArgumentParser(
        description="Run LLMCollector directly on a HF model (local path or hub id)."
    )
    parser.add_argument("model_id_or_path", help="HF hub id (e.g., HuggingFaceTB/SmolLM-360M) or local directory.")
    args = parser.parse_args()

    raw = args.model_id_or_path
    model_path = Path(raw).expanduser()
    model_ref: str | Path = model_path if model_path.exists() else raw

    tok = AutoTokenizer.from_pretrained(model_ref)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_ref).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    db = LatentDB("sqlite:///latents_hf_local.db")
    collector = LLMCollector(db, layer_indices=[-1], max_channels=512)
    collector.run(
        model,
        tok,
        prompts=["Hello world", "Why is the sky blue?"],
        max_new_tokens=16,
    )


if __name__ == "__main__":
    main()
