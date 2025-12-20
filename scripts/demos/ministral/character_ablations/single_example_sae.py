"""
Run one daily_dilemmas example through 4 character rewrites, then apply a
pretrained SAE encoder to the chosen layer's hidden states and visualize the
latent activations.

Outputs:
  - PNG heatmap(s) of latent activations per character.
  - CSV summary per (character, latent): mean/std/min/max and argmax token.

Example:
  RUN_MINISTRAL3=1 HF_TRUST_REMOTE_CODE=1 PYTHONPATH=. \\
    python scripts/demos/ministral/character_ablations/single_example_sae.py \\
      --model mistralai/Ministral-3-14B-Instruct-2512 \\
      --layer llm.layer.30 \\
      --sae artifacts/sae_llm_layer_30_20250101_000000.pth \\
      --out_png vis/single_example_sae.png \\
      --out_csv vis/single_example_sae.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

from characters import ch_1, ch_2, ch_3, ch_4


CHARACTERS: Dict[int, str] = {0: ch_1, 1: ch_2, 2: ch_3, 3: ch_4}
DEFAULT_FILTER_VALUE = "honesty"


def pick_text(example: dict) -> str:
    if example.get("dilemma_situation"):
        return str(example["dilemma_situation"])
    raise KeyError("dilemma_situation missing; expected field not found.")


def has_value(example: dict, target: str) -> bool:
    values = example.get("values_aggregated")
    if not values:
        return False
    if isinstance(values, str):
        return target in values.lower()
    try:
        return any(target in str(v).lower() for v in values)
    except TypeError:
        return False


def rewrite_prompt(base_prompt: str, character_text: str, use_api: bool) -> str:
    template = (
        "Rewrite the following prompt in the style of the character below without changing the meaning."
        "Do not add or remove content beyond stylistic changes. Return the rewritten prompt only without any other text. \n"
        f"character:\n{character_text}\n"
        f"prompt:\n{base_prompt}"
    )
    if use_api and os.environ.get("MISTRAL_API_KEY"):
        try:
            from utils import generate

            return generate(prompt=template)
        except Exception as exc:
            print(f"[warn] rewrite failed ({exc}); falling back to templated output.")
    character_title = character_text.splitlines()[0].strip("* ").strip().strip("\n")
    return f"[{character_title}] {base_prompt}"


def resolve_layer_index(layer: str) -> int:
    """
    Accepts:
      - "llm.layer.30" -> 30
      - "latent_sae:llm.layer.30" -> 30
      - "30" -> 30
    """
    if layer.isdigit():
        return int(layer)
    if ":" in layer:
        layer = layer.split(":", 1)[1]
    if layer.startswith("llm.layer."):
        return int(layer.split(".")[-1])
    raise ValueError(f"Unrecognized layer format: {layer!r}")


def load_sae_encoder(path: Path) -> nn.Linear:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        state = ckpt["encoder"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unexpected SAE checkpoint format in {path}")

    weight = state.get("weight")
    if weight is None:
        raise ValueError(f"Missing encoder.weight in {path}")
    out_dim, in_dim = weight.shape
    bias = "bias" in state
    enc = nn.Linear(in_dim, out_dim, bias=bias)
    enc.load_state_dict(state)
    enc.eval()
    return enc


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


def forward_hidden_states(tok, llm, prompts: List[str], layer_idx: int, device: str):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    with torch.no_grad():
        out = llm(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    if hs is None:
        raise RuntimeError("Model did not return hidden_states.")

    layer_tensor = hs[layer_idx]  # (B, S, H)
    decoded_tokens = [tok.convert_ids_to_tokens(input_ids[b].tolist()) for b in range(input_ids.size(0))]
    lengths = [int(attn_mask[b].ne(0).sum().item()) if attn_mask is not None else input_ids.size(1) for b in range(input_ids.size(0))]
    return layer_tensor, decoded_tokens, lengths


def generate_completions(
    tok,
    llm,
    prompts_by_label: Dict[int, str],
    *,
    device: str,
    max_new_tokens: int,
) -> Dict[int, str]:
    completions: Dict[int, str] = {}
    for label, prompt in sorted(prompts_by_label.items()):
        enc = tok([prompt], return_tensors="pt", padding=False, truncation=True)
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

        # Decode only the generated continuation.
        gen_ids = out[0, input_ids.shape[1] :]
        try:
            text = tok.decode(gen_ids.tolist(), skip_special_tokens=True)
        except Exception:
            # Conservative fallback: decode full sequence.
            text = tok.decode(out[0].tolist(), skip_special_tokens=True)
        completions[label] = text.strip()
    return completions


def plot_latent_means(
    latents_by_label: Dict[int, np.ndarray],
    lengths: Dict[int, int],
    out_png: Path,
    *,
    title: str,
):
    """
    Plot a single heatmap: (latent x character) where each cell is the mean
    activation of that latent across the prompt tokens for that character.
    """
    labels = sorted(latents_by_label.keys())
    k = next(iter(latents_by_label.values())).shape[0]

    mat = np.zeros((k, len(labels)), dtype=float)
    for j, label in enumerate(labels):
        L = lengths[label]
        z = latents_by_label[label][:, :L]  # (K, S)
        mat[:, j] = z.mean(axis=1)

    vmax = float(np.abs(mat).max()) if mat.size else 1.0
    vmin = -vmax

    plt.figure(figsize=(8, 10))
    plt.imshow(mat, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Mean latent activation (over tokens)")
    plt.xlabel("Character label")
    plt.ylabel("Latent")
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def summarize_to_csv(
    out_csv: Path,
    *,
    base_prompt: str,
    rewritten_prompts: Dict[int, str],
    latents_by_label: Dict[int, np.ndarray],
    tokens_by_label: Dict[int, List[str]],
    lengths: Dict[int, int],
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "character",
                "base_prompt",
                "rewritten_prompt",
                "latent",
                "mean",
                "std",
                "min",
                "max",
                "argmax_token_index",
                "argmax_token",
            ],
        )
        w.writeheader()
        for label, z in sorted(latents_by_label.items()):
            L = lengths[label]
            z = z[:, :L]
            for latent_idx in range(z.shape[0]):
                row = z[latent_idx]
                argmax = int(row.argmax())
                w.writerow(
                    {
                        "character": label,
                        "base_prompt": base_prompt,
                        "rewritten_prompt": rewritten_prompts.get(label, ""),
                        "latent": latent_idx,
                        "mean": float(row.mean()),
                        "std": float(row.std()),
                        "min": float(row.min()),
                        "max": float(row.max()),
                        "argmax_token_index": argmax,
                        "argmax_token": tokens_by_label[label][argmax] if argmax < len(tokens_by_label[label]) else "",
                    }
                )


def print_cli_summary(latents_by_label: Dict[int, np.ndarray], lengths: Dict[int, int], topk: int = 10):
    labels = sorted(latents_by_label.keys())
    # mean per latent per label
    means = []
    for label in labels:
        z = latents_by_label[label][:, : lengths[label]]
        means.append(z.mean(axis=1))
    means = np.stack(means, axis=1)  # (K, num_labels)

    spread = means.max(axis=1) - means.min(axis=1)
    pooled_std = means.std(axis=1) + 1e-8
    norm = spread / pooled_std
    top_idx = np.argsort(-norm)[:topk]

    print("[summary] Top varying latents by spread/std of per-prompt mean:")
    for ch in top_idx:
        mean_str = " ".join(f"{lab}:{means[ch, i]:.3f}" for i, lab in enumerate(labels))
        print(f"  latent {ch:3d} | spread {spread[ch]:.4f} | spread/std {norm[ch]:.2f} | means {mean_str}")


def write_token_latents_csv(
    out_csv: Path,
    *,
    base_prompt: str,
    rewritten_prompts: Dict[int, str],
    latents_by_label: Dict[int, np.ndarray],
    tokens_by_label: Dict[int, List[str]],
    lengths: Dict[int, int],
):
    """
    Write a token-level CSV: one row per (character, token), with all latent values.
    This is typically the most useful view for inspecting "what tokens light up a latent".
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    k = next(iter(latents_by_label.values())).shape[0]
    fieldnames = [
        "character",
        "base_prompt",
        "rewritten_prompt",
        "token_index",
        "token",
    ] + [f"latent_{i}" for i in range(k)]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, z in sorted(latents_by_label.items()):
            L = lengths[label]
            toks = tokens_by_label[label][:L]
            z = z[:, :L]  # (K, L)
            for t in range(L):
                row = {
                    "character": label,
                    "base_prompt": base_prompt,
                    "rewritten_prompt": rewritten_prompts.get(label, ""),
                    "token_index": t,
                    "token": toks[t] if t < len(toks) else "",
                }
                for i in range(k):
                    row[f"latent_{i}"] = float(z[i, t])
                w.writerow(row)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--layer", type=str, default="llm.layer.30", help="Base layer used to train the SAE.")
    ap.add_argument("--sae", type=Path, required=True, help="Path to SAE checkpoint (.pth) saved by SAEPipeline.")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filter_value", type=str, default=DEFAULT_FILTER_VALUE, help="Filter examples by values_aggregated tag.")
    ap.add_argument("--dataset_index", type=int, default=None, help="Optional explicit dataset row index (overrides sampling).")
    ap.add_argument("--no_api", action="store_true", help="Skip Mistral API rewrite and use a templated prefix instead.")
    ap.add_argument("--out_png", type=Path, default=Path("vis/single_example_sae.png"))
    ap.add_argument("--out_csv", type=Path, default=Path("vis/single_example_sae.csv"))
    ap.add_argument(
        "--out_tokens_csv",
        type=Path,
        default=Path("vis/single_example_sae_tokens.csv"),
        help="Token-level CSV containing all latent activations per token.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=-1,
        help="Generation cap; -1 picks a large automatic cap (still stops on EOS).",
    )
    ap.add_argument("--topk", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    if os.environ.get("RUN_MINISTRAL3") != "1":
        print("Set RUN_MINISTRAL3=1 to run (downloads weights); skipping.")
        return

    ds = load_dataset("kellycyy/daily_dilemmas", split=args.split)
    rng = random.Random(args.seed)

    if args.dataset_index is not None:
        ex = ds[int(args.dataset_index)]
        base = pick_text(ex)
        print(f"[data] Using dataset index {args.dataset_index} from split {args.split}")
    else:
        candidates = [idx for idx, ex in enumerate(ds) if has_value(ex, args.filter_value)]
        if not candidates:
            raise RuntimeError(f"No examples found containing {args.filter_value!r} in values_aggregated.")
        chosen = rng.choice(candidates)
        ex = ds[int(chosen)]
        base = pick_text(ex)
        print(f"[data] Chose dataset index {chosen} from {len(candidates)} candidates in split {args.split}")

    prompts: List[str] = []
    prompts_by_label: Dict[int, str] = {}
    for label, character in CHARACTERS.items():
        rewritten = rewrite_prompt(base, character, use_api=not args.no_api) + "\ngive me a hard decision in one line."
        rewritten = rewritten.strip()
        prompts.append(rewritten)
        prompts_by_label[label] = rewritten

    print("\n[prompt] Base prompt:")
    print(base)
    print("\n[prompt] Rewritten prompts:")
    for label in sorted(prompts_by_label.keys()):
        print(f"\n  character {label}:")
        print(prompts_by_label[label])
    print()

    trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1") == "1"
    tok, llm, device = load_model_and_tokenizer(args.model, trust_remote_code)
    layer_idx = resolve_layer_index(args.layer)

    if args.max_new_tokens == -1:
        # "Unlimited" generation isn't supported; pick a large cap and rely on EOS.
        max_new_tokens = 128
    else:
        max_new_tokens = int(args.max_new_tokens)
    if max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0 (or -1 for auto)")

    completions = generate_completions(tok, llm, prompts_by_label, device=device, max_new_tokens=max_new_tokens)
    print(f"[gen] Completions (max_new_tokens={max_new_tokens}):")
    for label in sorted(completions.keys()):
        print(f"\n  character {label} completion:\n{completions[label]}")
    print()

    encoder = load_sae_encoder(args.sae).to(device)
    layer_tensor, decoded_tokens, lengths_list = forward_hidden_states(tok, llm, prompts, layer_idx, device)
    sae_in_dim = int(getattr(encoder, "in_features", 0) or 0)
    if sae_in_dim <= 0:
        raise RuntimeError("Could not determine SAE encoder input dimension (expected nn.Linear).")
    if layer_tensor.shape[-1] < sae_in_dim:
        raise RuntimeError(
            f"Hidden state dim {layer_tensor.shape[-1]} is smaller than SAE encoder input dim {sae_in_dim}."
        )
    if layer_tensor.shape[-1] != sae_in_dim:
        print(
            f"[warn] SAE expects in_dim={sae_in_dim}, but model hidden dim={layer_tensor.shape[-1]}; "
            f"using the first {sae_in_dim} channels (consistent with collector max_channels)."
        )

    latents_by_label: Dict[int, np.ndarray] = {}
    tokens_by_label: Dict[int, List[str]] = {}
    lengths: Dict[int, int] = {}

    with torch.no_grad():
        enc_dtype = next(encoder.parameters()).dtype
        for label in sorted(CHARACTERS.keys()):
            L = int(lengths_list[label])
            x = layer_tensor[label, :L, :sae_in_dim].to(device=device, dtype=enc_dtype)  # (S, H_sae)
            z = encoder(x).T  # (K, S)
            latents_by_label[label] = z.float().cpu().numpy()
            tokens_by_label[label] = decoded_tokens[label][:L]
            lengths[label] = L

    title = f"SAE latent means for one example @ layer {args.layer} (k={next(iter(latents_by_label.values())).shape[0]})"
    plot_latent_means(latents_by_label, lengths, args.out_png, title=title)
    summarize_to_csv(
        args.out_csv,
        base_prompt=base,
        rewritten_prompts=prompts_by_label,
        latents_by_label=latents_by_label,
        tokens_by_label=tokens_by_label,
        lengths=lengths,
    )
    write_token_latents_csv(
        args.out_tokens_csv,
        base_prompt=base,
        rewritten_prompts=prompts_by_label,
        latents_by_label=latents_by_label,
        tokens_by_label=tokens_by_label,
        lengths=lengths,
    )

    print(f"[out] Wrote {args.out_png}")
    print(f"[out] Wrote {args.out_csv}")
    print(f"[out] Wrote {args.out_tokens_csv}")
    print("[tokens] Tokenized lengths:", " ".join(f"{lab}={lengths[lab]}" for lab in sorted(lengths)))
    print_cli_summary(latents_by_label, lengths, topk=args.topk)


if __name__ == "__main__":
    main()
