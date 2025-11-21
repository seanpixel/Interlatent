"""
Smoke test: run VLLMCollector against a vLLM instance that loads weights
from a local HuggingFace checkout (no network pull).

Usage:
  export VLLM_MODEL_DIR=/path/to/local/model/dir
  python tests/llm_vllm_local.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vllm import LLM
from transformers import AutoTokenizer

from interlatent.api import LatentDB
from interlatent.llm import VLLMCollector


def main():
    parser = argparse.ArgumentParser(description="Run VLLMCollector with a local model directory (no download).")
    parser.add_argument("model_dir", type=Path, help="Path to local HF model directory.")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir)

    db = LatentDB("sqlite:///latents_llm_local.db")
    collector = VLLMCollector(db, layer_indices=[-1], max_channels=512)
    collector.run(
        llm,
        tok,
        prompts=["Hello world", "Why is the sky blue?"],
        max_new_tokens=16,
    )


if __name__ == "__main__":
    main()
