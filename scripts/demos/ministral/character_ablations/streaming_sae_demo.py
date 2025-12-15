"""
Streaming SAE training demo (no DB) for Ministral character prompts.

Streams activations from a specified layer through an SAE without persisting
full activations. Intended as a lightweight alternative to dense logging.

Usage:
  RUN_MINISTRAL3=1 HF_TRUST_REMOTE_CODE=1 PYTHONPATH=. \\
    python scripts/demos/ministral/character_ablations/streaming_sae_demo.py \\
      --csv data/character_dilemmas.csv \\
      --model mistralai/Ministral-3-14B-Instruct-2512 \\
      --layer_idx -2 \\
      --k 128
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from interlatent.llm.prompt_dataset import PromptDataset
from interlatent.sae.streaming import StreamingSAEConfig, StreamingSAETrainer


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


def main(args):
    if os.environ.get("RUN_MINISTRAL3") != "1":
        print("Set RUN_MINISTRAL3=1 to run (downloads weights); skipping.")
        return

    dataset = PromptDataset.from_csv(args.csv, text_col="text", label_col="label")
    trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1") == "1"
    tok, llm, device = load_model_and_tokenizer(args.model, trust_remote_code)

    cfg = StreamingSAEConfig(
        layer_index=args.layer_idx,
        k=args.k,
        lr=args.lr,
        l1=args.l1,
        max_channels=args.max_channels,
        batch_size=args.batch_size,
        sample_tokens=args.sample_tokens,
        device=device,
    )
    print(f"[streaming] config: {cfg}")
    trainer = StreamingSAETrainer(cfg)
    sae = trainer.train(llm, tok, dataset.texts)

    if sae is None:
        print("SAE training did not initialize (no activations seen).")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": sae.encoder.state_dict(), "decoder": sae.decoder.state_dict()}, output_path)
    print(f"[streaming] Saved SAE to {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="CSV with text/label")
    ap.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--layer_idx", type=int, default=-2, help="Hidden_states index (negative for from-end)")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_channels", type=int, default=None)
    ap.add_argument("--sample_tokens", type=int, default=2048)
    ap.add_argument("--output", type=Path, default=Path("artifacts/streaming_sae.pth"))
    args = ap.parse_args()
    main(args)
