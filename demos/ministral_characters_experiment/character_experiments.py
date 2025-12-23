"""
End-to-end character ablation workflow (CLI script, no notebook needed).

Steps:
 1) Load the character prompts CSV.
 2) Generate single-line decisions for each prompt (requires RUN_MINISTRAL3=1).
 3) Visualize SAE latents (image + CLI) using the latent DB.
 4) Print top tokens per latent.

Defaults assume:
  - CSV: data/character_dilemmas.csv
  - SAE checkpoint: artifacts/sae_llm_layer_30_20251217_070930.pth
  - Latent DB: latents_character_dilemmas.db
  - Layer: llm.layer.30
  - Model: mistralai/Ministral-3-14B-Instruct-2512
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch


def resolve_path(default: Path, pattern: str) -> Path:
    if default.exists():
        return default
    matches = list(Path(".").rglob(pattern))
    if matches:
        print(f"[info] Using discovered file: {matches[0]}")
        return matches[0]
    raise FileNotFoundError(f"Could not find {pattern}; tried default {default}")


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
    llm.eval()
    return tok, llm, device


def decision_prompt(text: str) -> str:
    return f"{text}\n\nRespond with a single-line final decision only. No explanations."


def generate_decisions(prompts, tok, llm, *, max_new_tokens: int, device: str):
    outputs = []
    for prompt in prompts:
        enc = tok([prompt], return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        with torch.no_grad():
            out = llm.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        text = tok.decode(out[0].tolist(), skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


def visualize_latents(db_path: Path, latent_layer: str, out_png: Path, topk: int):
    from demos.ministral_characters_experiment.visualize_latents import (
        load_events,
        aggregate,
        plot_heatmap,
        report_variation,
    )

    events = load_events(db_path, latent_layer)
    mat, stds, labels = aggregate(events)
    plot_heatmap(mat, labels, out_png)
    print(f"[latents] Saved heatmap to {out_png}")
    lines = report_variation(mat, labels, stds, top_k=topk)
    print("[latents] Top varying channels:")
    for line in lines:
        print("  " + line)


def top_tokens(db_path: Path, latent_layer: str, min_count: int, topk: int):
    from demos.ministral_characters_experiment.top_tokens import (
        load_events as load_events_tt,
        aggregate_token_stats,
    )

    events_tt = load_events_tt(db_path, latent_layer, limit=None)
    stats = aggregate_token_stats(events_tt, min_count=min_count)
    for ch in sorted(stats.keys()):
        tokens = sorted(stats[ch].items(), key=lambda kv: kv[1][0], reverse=True)[:topk]
        if not tokens:
            continue
        print(f"channel {ch}:")
        for token, (mean_val, count, max_val) in tokens:
            print(f"  {token!r:12s} mean={mean_val: .4f} count={count:3d} max={max_val: .4f}")
        print()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=os.getenv("CHARACTER_CSV", "data/character_dilemmas.csv"))
    ap.add_argument("--sae", type=str, default=os.getenv("SAE_PATH", "artifacts/sae_llm_layer_30_20251217_070930.pth"))
    ap.add_argument("--db", type=str, default=os.getenv("LATENT_DB", "latents_character_dilemmas.db"))
    ap.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--layer", type=str, default="llm.layer.30")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--decisions_out", type=str, default="vis/character_decisions.csv")
    ap.add_argument("--latents_png", type=str, default="vis/character_latents.png")
    ap.add_argument("--topk_latents", type=int, default=10)
    ap.add_argument("--topk_tokens", type=int, default=10)
    ap.add_argument("--min_count_tokens", type=int, default=3)
    return ap.parse_args()


def main():
    args = parse_args()
    data_csv = resolve_path(Path(args.csv), "character_dilemmas.csv")
    sae_path = resolve_path(Path(args.sae), "sae_llm_layer_30_*.pth")
    db_path = resolve_path(Path(args.db), "latents_character_dilemmas.db")

    df = pd.read_csv(data_csv)
    print(f"[data] Loaded {len(df)} prompts from {data_csv}")

    if os.environ.get("RUN_MINISTRAL3") == "1":
        trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1") == "1"
        tok, llm, device = load_model_and_tokenizer(args.model, trust_remote_code)
        print(f"[model] Loaded on {device}")

        prompts = [decision_prompt(t) for t in df["text"].tolist()]
        decisions = generate_decisions(prompts, tok, llm, max_new_tokens=args.max_new_tokens, device=device)
        df_out = df.copy()
        df_out["decision"] = decisions

        out_path = Path(args.decisions_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)

        unique_decisions = sorted(set(decisions))
        print(f"[decisions] Saved to {out_path}")
        print(f"[decisions] Unique decisions: {len(unique_decisions)} / {len(decisions)}")
        for line in decisions[:5]:
            print("  ", line)
    else:
        print("[skip] RUN_MINISTRAL3 is not set; skipping decision generation.")

    latent_layer = f"latent_sae:{args.layer}"
    visualize_latents(db_path, latent_layer, Path(args.latents_png), topk=args.topk_latents)
    top_tokens(db_path, latent_layer, min_count=args.min_count_tokens, topk=args.topk_tokens)


if __name__ == "__main__":
    main()
