"""
End-to-end demo: collect activations from a HF model and generate plots.

Usage:
  python tests/vis_demo.py --model HuggingFaceTB/SmolLM-360M --outdir graphs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlatent.api import LatentDB
from interlatent.llm import VLLMCollector
from interlatent.vis.plot import plot_activation


def main():
    parser = argparse.ArgumentParser(description="Collect activations and render plots.")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-360M", help="HF model id or local path.")
    parser.add_argument("--outdir", default="graphs", help="Directory to save plots.")
    parser.add_argument("--db", default="sqlite:///latents_vis_demo.db", help="LatentDB SQLite URI/path.")
    parser.add_argument("--channels", type=int, nargs="+", default=[0], help="Channels to plot.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    prompts = [
        "Hello world!",
        "Why is the sky blue?",
        "Explain how transformers work in simple terms.",
    ]

    db = LatentDB(args.db)
    collector = VLLMCollector(db, layer_indices=[-1], max_channels=max(args.channels) + 1)
    collector.run(
        model,
        tok,
        prompts=prompts,
        max_new_tokens=16,
        batch_size=1,
    )

    layer = "llm.layer.-1"
    outputs = []
    for ch in args.channels:
        for p_idx in range(len(prompts)):
            outfile = outdir / f"activation_layer-1_ch{ch}_prompt{p_idx}.png"
            out_path = plot_activation(
                args.db,
                layer=layer,
                channel=ch,
                prompt_index=p_idx,
                output=str(outfile),
            )
            outputs.append(out_path)

    print("Plots saved:")
    for p in outputs:
        print(" -", p)


if __name__ == "__main__":
    main()
