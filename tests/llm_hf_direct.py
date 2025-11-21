"""
Smoke test: run VLLMCollector directly on a local HuggingFace model
without going through vLLM (no download).

Usage:
  export HF_MODEL_DIR=/path/to/local/model/dir
  python tests/llm_hf_direct.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlatent.api import LatentDB
from interlatent.llm import VLLMCollector


def main():
    parser = argparse.ArgumentParser(description="Run VLLMCollector directly on a local HF model (no download).")
    parser.add_argument("model_dir", type=Path, help="Path to local HF model directory.")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    db = LatentDB("sqlite:///latents_hf_local.db")
    collector = VLLMCollector(db, layer_indices=[-1], max_channels=512)
    collector.run(
        model,
        tok,
        prompts=["Hello world", "Why is the sky blue?"],
        max_new_tokens=16,
    )


if __name__ == "__main__":
    main()
