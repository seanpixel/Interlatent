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
    parser = argparse.ArgumentParser(
        description="Run VLLMCollector with a HF model (hub id or local path)."
    )
    parser.add_argument("model", help="HF hub id (e.g., HuggingFaceTB/SmolLM-360M) or local model directory.")
    args = parser.parse_args()

    raw = args.model
    model_path = Path(raw).expanduser()
    model_ref = str(model_path.resolve()) if model_path.exists() else raw

    tok = AutoTokenizer.from_pretrained(model_ref)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token

    llm = LLM(model=model_ref)

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
