"""
Activation patching demo: snitch vs report in a religious confession prompt.

This script computes a hidden-state delta between the target words in two
prompts and patches that delta into the other prompt during generation.

Usage:
  RUN_MINISTRAL3=1 HF_TRUST_REMOTE_CODE=1 PYTHONPATH=. \
    python demos/ministral_characters_experiment/activation_patching.py \
    --layer llm.layer.30 \
    --prompt-a "I overheard ... Do i snitch? Give me a hard decision in one line." \
    --prompt-b "I overheard ... Do i report? Give me a hard decision in one line."
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.distributed as dist

from interlatent.analysis.intervention.api import _resolve_layer_module
from interlatent.analysis.vis.diff import _open_db, latent_diff
from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.schema import ActivationEvent


DEFAULT_PROMPT_A = (
    "I overheard someone say they will commit a crime during a holy and secret "
    "religious confession. Do i snitch? Give me a hard decision in one line."
)
DEFAULT_PROMPT_B = (
    "I overheard someone say they will commit a crime during a holy and secret "
    "religious confession. Do i report? Give me a hard decision in one line."
)


def _ensure_single_process_group():
    if not dist.is_available() or dist.is_initialized():
        return False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    try:
        dist.init_process_group("gloo", rank=0, world_size=1)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not init process group; continuing without distributed init: {exc}")
        return False


def load_model_and_tokenizer(model_id: str, trust_remote_code: bool):
    from transformers import AutoConfig, Mistral3ForConditionalGeneration, MistralCommonBackend

    created_pg = _ensure_single_process_group()
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
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()
    return tok, llm, device


def encode_prompt(tok, prompt: str, device: str):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    enc = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    return input_ids, attn_mask


def normalize_token(tok: str) -> str:
    return tok.replace("▁", " ").strip().lower()


def find_subsequence(seq: Sequence[int], sub: Sequence[int]) -> List[int]:
    if not sub or len(sub) > len(seq):
        return []
    matches = []
    for i in range(len(seq) - len(sub) + 1):
        if list(seq[i : i + len(sub)]) == list(sub):
            matches.append(i)
    if not matches:
        return []
    # Prefer the last match to avoid earlier boilerplate tokens.
    start = matches[-1]
    return list(range(start, start + len(sub)))


def find_token_indices(
    tok,
    input_ids: Sequence[int],
    target: str,
) -> tuple[List[int], str | None]:
    tokens = tok.convert_ids_to_tokens(list(input_ids))
    target_norm = normalize_token(target)
    matches = [i for i, t in enumerate(tokens) if target_norm in normalize_token(t)]
    if matches:
        idx = matches[-1]
        return [idx], tokens[idx]

    # Fallback: try exact token ID subsequence for common punctuation variants.
    variants = [target, f"{target}?", f"{target}.", f"{target}!", f"{target},"]
    for variant in variants:
        sub_ids = tok(variant, add_special_tokens=False)["input_ids"]
        seq_idx = find_subsequence(input_ids, sub_ids)
        if seq_idx:
            return seq_idx, tokens[seq_idx[0]]
    return [], None


def token_like_from_prompt(tok, prompt: str, target: str) -> str:
    enc = tok(prompt, add_special_tokens=False)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"])
    target_norm = normalize_token(target)
    for t in tokens:
        if target_norm in normalize_token(t):
            return t
    # Fallback: pick a token piece from target tokenization (use longest piece).
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    target_tokens = tok.convert_ids_to_tokens(target_ids)
    if target_tokens:
        target_tokens = sorted(target_tokens, key=lambda s: len(normalize_token(s)), reverse=True)
        return target_tokens[0]
    return target


def hidden_states_at_layer(llm, input_ids, attn_mask, layer_idx: int):
    with torch.no_grad():
        out = llm(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = out.hidden_states
    if hidden_states is None or layer_idx >= len(hidden_states):
        raise RuntimeError("Model did not return hidden_states for requested layer.")
    return hidden_states[layer_idx]


def generate(tok, llm, prompt: str, max_new_tokens: int, device: str) -> str:
    input_ids, attn_mask = encode_prompt(tok, prompt, device)
    with torch.no_grad():
        out = llm.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
    return tok.decode(out[0].tolist(), skip_special_tokens=True)


@dataclass
class PatchConfig:
    layer: str
    token_indices: List[int]
    delta: torch.Tensor


def patch_at_tokens(llm, cfg: PatchConfig):
    layer_module = _resolve_layer_module(llm, cfg.layer)
    delta = cfg.delta

    def _hook(_mod, _inputs, output):
        if not isinstance(output, torch.Tensor):
            return output
        hs = output
        if not cfg.token_indices:
            return hs
        if hs.size(1) <= max(cfg.token_indices):
            return hs
        mask = hs.new_zeros(hs.shape[:2] + (1,))
        mask[:, cfg.token_indices, :] = 1.0
        d = delta.to(hs.device, dtype=hs.dtype)
        if d.ndim == 1:
            d = d.view(1, 1, -1)
        return hs + mask * d

    return layer_module.register_forward_hook(_hook)


def load_sae_encoder(path: Path) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "encoder" not in ckpt:
        raise ValueError("SAE checkpoint missing 'encoder' state dict")
    enc_state = ckpt["encoder"]
    weight = enc_state.get("weight")
    if weight is None:
        raise ValueError("Encoder state missing weight")
    latent_dim, hidden_dim = weight.shape
    bias = "bias" in enc_state
    encoder = torch.nn.Linear(hidden_dim, latent_dim, bias=bias)
    encoder.load_state_dict(enc_state)
    encoder.eval()
    return encoder


def backfill_sae_latents(db: LatentDB, base_layer: str, encoder: torch.nn.Module):
    latent_layer = f"latent_sae:{base_layer}"
    events = db.fetch_activations(layer=base_layer)
    if not events:
        raise RuntimeError(f"No activations found for layer '{base_layer}'")

    grouped = {}
    ctx_by_key = {}
    meta_by_key = {}

    def key_for(ev):
        if ev.prompt_index is not None and ev.token_index is not None:
            return (ev.run_id, ev.prompt_index, ev.token_index)
        return (ev.run_id, ev.step)

    for ev in events:
        key = key_for(ev)
        grouped.setdefault(key, {})[ev.channel] = ev.value_sum or sum(ev.tensor)
        ctx_by_key.setdefault(key, ev.context or {})
        if key not in meta_by_key:
            meta_by_key[key] = {
                "prompt": ev.prompt,
                "prompt_index": ev.prompt_index,
                "token_index": ev.token_index,
                "token": ev.token,
            }

    encoder.eval()
    with torch.no_grad():
        for key, vec_dict in grouped.items():
            run_id = key[0]
            step = key[1] if len(key) == 2 else key[1] * 10_000 + key[2]
            x = torch.tensor([vec_dict[i] for i in sorted(vec_dict)], dtype=torch.float32)
            z = encoder(x.unsqueeze(0)).squeeze(0)
            for idx, val in enumerate(z):
                db.write_event(
                    ActivationEvent(
                        run_id=run_id,
                        step=step,
                        layer=latent_layer,
                        channel=idx,
                        tensor=[float(val)],
                        prompt=meta_by_key[key]["prompt"],
                        prompt_index=meta_by_key[key]["prompt_index"],
                        token_index=meta_by_key[key]["token_index"],
                        token=meta_by_key[key]["token"],
                        context=ctx_by_key[key],
                        value_sum=float(val),
                        value_sq_sum=float(val * val),
                    )
                )
    db.flush()


def collect_to_db(db_path: Path, tok, llm, layer: str, prompts: List[str], device: str):
    if db_path.exists():
        db_path.unlink()
    db = LatentDB(f"sqlite:///{db_path}")
    collector = LLMCollector(
        db,
        layer_indices=[int(layer.split(".")[-1])],
        max_channels=128,
        device=device,
        log_every_prompts=1,
    )
    collector.run(
        llm,
        tok,
        prompts=prompts,
        max_new_tokens=0,
        batch_size=1,
    )
    return db


def _norm_label(token: str | None) -> str:
    if not token:
        return ""
    return token.replace("▁", " ").strip()


def _find_span_by_concat(tokens: List[str], target: str, max_span: int = 4) -> List[int]:
    target_norm = normalize_token(target).replace(" ", "")
    if not target_norm:
        return []
    toks_norm = [normalize_token(t).replace(" ", "") for t in tokens]
    for i in range(len(toks_norm)):
        acc = ""
        for j in range(i, min(i + max_span, len(toks_norm))):
            acc += toks_norm[j]
            if acc == target_norm:
                return list(range(i, j + 1))
    return []


def _align_sequences(
    tokens_a: List[str],
    vals_a: List[Dict[int, float]],
    tokens_b: List[str],
    vals_b: List[Dict[int, float]],
    target_a: str,
    target_b: str,
):
    span_a = _find_span_by_concat(tokens_a, target_a)
    span_b = _find_span_by_concat(tokens_b, target_b)
    if not span_a or not span_b:
        return tokens_a, vals_a, tokens_b, vals_b

    len_a = len(span_a)
    len_b = len(span_b)
    if len_a == len_b:
        return tokens_a, vals_a, tokens_b, vals_b

    if len_a > len_b:
        pad = len_a - len_b
        insert_at = span_b[-1] + 1
        for _ in range(pad):
            tokens_b.insert(insert_at, "<pad>")
            vals_b.insert(insert_at, {})
    else:
        pad = len_b - len_a
        insert_at = span_a[-1] + 1
        for _ in range(pad):
            tokens_a.insert(insert_at, "<pad>")
            vals_a.insert(insert_at, {})

    return tokens_a, vals_a, tokens_b, vals_b


def build_latent_diff_heatmap(
    db_path: Path,
    latent_layer: str,
    prompt_a: str,
    prompt_b: str,
    target_a: str,
    target_b: str,
    *,
    topk: int,
    out_path: Path,
):
    import matplotlib.pyplot as plt

    db = LatentDB(f"sqlite:///{db_path}")
    events = db.fetch_activations(layer=latent_layer)
    db.close()

    def collect_prompt(prompt_text: str):
        by_idx = {}
        tok_by_idx = {}
        for ev in events:
            if ev.prompt != prompt_text:
                continue
            if ev.token_index is None:
                continue
            by_idx.setdefault(ev.token_index, {})[ev.channel] = float(
                ev.value_sum if ev.value_sum is not None else (ev.tensor[0] if ev.tensor else 0.0)
            )
            if ev.token_index not in tok_by_idx:
                tok_by_idx[ev.token_index] = ev.token
        return by_idx, tok_by_idx

    a_by_idx, a_tok = collect_prompt(prompt_a)
    b_by_idx, b_tok = collect_prompt(prompt_b)
    token_indices = sorted(set(a_by_idx.keys()) | set(b_by_idx.keys()))
    if not token_indices:
        raise RuntimeError("No token activations found for prompts in latent DB.")

    tokens_a = [_norm_label(a_tok.get(idx)) for idx in sorted(a_by_idx.keys())]
    tokens_b = [_norm_label(b_tok.get(idx)) for idx in sorted(b_by_idx.keys())]
    vals_a = [a_by_idx.get(idx, {}) for idx in sorted(a_by_idx.keys())]
    vals_b = [b_by_idx.get(idx, {}) for idx in sorted(b_by_idx.keys())]

    tokens_a, vals_a, tokens_b, vals_b = _align_sequences(tokens_a, vals_a, tokens_b, vals_b, target_a, target_b)

    max_len = max(len(tokens_a), len(tokens_b))
    if len(tokens_a) < max_len:
        tokens_a.extend(["<pad>"] * (max_len - len(tokens_a)))
        vals_a.extend([{}] * (max_len - len(vals_a)))
    if len(tokens_b) < max_len:
        tokens_b.extend(["<pad>"] * (max_len - len(tokens_b)))
        vals_b.extend([{}] * (max_len - len(vals_b)))

    channels = set()
    for idx in range(max_len):
        channels.update(vals_a[idx].keys())
        channels.update(vals_b[idx].keys())
    channels = sorted(channels)

    # Match diff.py ranking: sort by |mean_B - mean_A| over the full prompt slice.
    channel_scores = []
    for ch in channels:
        sum_a = 0.0
        sum_b = 0.0
        cnt_a = 0
        cnt_b = 0
        for idx in range(max_len):
            if ch in vals_a[idx]:
                sum_a += vals_a[idx][ch]
                cnt_a += 1
            if ch in vals_b[idx]:
                sum_b += vals_b[idx][ch]
                cnt_b += 1
        mean_a = sum_a / cnt_a if cnt_a else 0.0
        mean_b = sum_b / cnt_b if cnt_b else 0.0
        score = abs(mean_b - mean_a)
        channel_scores.append((ch, score))
    channel_scores.sort(key=lambda x: x[1], reverse=True)
    top_channels = [ch for ch, _ in channel_scores[:topk]]

    # Build heatmap: rows=channels, cols=tokens, values=B-A.
    mat = np.zeros((len(top_channels), max_len), dtype=float)
    for r, ch in enumerate(top_channels):
        for c in range(max_len):
            a_val = vals_a[c].get(ch, 0.0)
            b_val = vals_b[c].get(ch, 0.0)
            mat[r, c] = b_val - a_val

    labels = []
    for i in range(max_len):
        ta = tokens_a[i]
        tb = tokens_b[i]
        if ta == tb:
            labels.append(ta)
        else:
            labels.append(f"{ta}|{tb}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(labels) * 0.35), 4 + len(top_channels) * 0.25))
    plt.imshow(mat, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Activation diff (B - A)")
    plt.yticks(ticks=range(len(top_channels)), labels=[str(ch) for ch in top_channels])
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=60, ha="right")
    plt.xlabel("Tokens (prompt A | prompt B if different)")
    plt.ylabel("Top diverging SAE channels")
    plt.title("Prompt-level SAE latent differences (B - A)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--layer", type=str, default="llm.layer.30")
    ap.add_argument("--latent-db", type=str, default="latents_snitch_report.db")
    ap.add_argument("--latent-layer", type=str, default="latent_sae:llm.layer.30")
    ap.add_argument("--sae", type=Path, default=Path("artifacts/sae_llm_layer_30_20251217_070930.pth"))
    ap.add_argument("--no-collect", action="store_true", help="Skip collecting a clean DB for prompts A/B.")
    ap.add_argument("--topk-diff", type=int, default=20)
    ap.add_argument("--heatmap-topk", type=int, default=10)
    ap.add_argument("--heatmap-out", type=Path, default=Path("vis/snitch_report_latent_diff_heatmap.png"))
    ap.add_argument("--prompt-a", type=str, default=DEFAULT_PROMPT_A)
    ap.add_argument("--prompt-b", type=str, default=DEFAULT_PROMPT_B)
    ap.add_argument("--target-a", type=str, default="snitch")
    ap.add_argument("--target-b", type=str, default="report")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    return ap.parse_args()


def main():
    args = parse_args()
    if os.environ.get("RUN_MINISTRAL3") != "1":
        print("Set RUN_MINISTRAL3=1 to run (downloads weights); skipping.")
        return

    trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1") == "1"
    tok, llm, device = load_model_and_tokenizer(args.model, trust_remote_code)

    if not args.no_collect and args.latent_db:
        db_path = Path(args.latent_db)
        print(f"[collect] Building clean DB at {db_path}...")
        db = collect_to_db(db_path, tok, llm, args.layer, [args.prompt_a, args.prompt_b], device)
        if args.sae.exists():
            print(f"[collect] Backfilling SAE latents from {args.sae}...")
            encoder = load_sae_encoder(args.sae)
            backfill_sae_latents(db, args.layer, encoder)
        else:
            print(f"[warn] SAE checkpoint not found at {args.sae}; skipping SAE backfill.")
        db.close()

    if args.latent_db and args.latent_layer:
        try:
            token_like_a = token_like_from_prompt(tok, args.prompt_a, args.target_a)
            token_like_b = token_like_from_prompt(tok, args.prompt_b, args.target_b)
            conn = _open_db(args.latent_db)
            table_full = latent_diff(
                conn,
                conn,
                layer=args.latent_layer,
                prompt_like_a=args.prompt_a,
                prompt_like_b=args.prompt_b,
                top=args.topk_diff,
            )
            table_tokens = latent_diff(
                conn,
                conn,
                layer=args.latent_layer,
                prompt_like_a=args.prompt_a,
                prompt_like_b=args.prompt_b,
                token_like_a=token_like_a,
                token_like_b=token_like_b,
                top=args.topk_diff,
            )
            print("[sae diff] Top SAE channel diffs (full prompt):")
            print(table_full, "\n")
            print("[sae diff] Top SAE channel diffs (target tokens only):")
            print(table_tokens, "\n")
        except Exception as exc:
            print(f"[warn] SAE diff failed: {exc}")

    if args.latent_db and args.latent_layer:
        try:
            build_latent_diff_heatmap(
                Path(args.latent_db),
                args.latent_layer,
                args.prompt_a,
                args.prompt_b,
                args.target_a,
                args.target_b,
                topk=args.heatmap_topk,
                out_path=args.heatmap_out,
            )
            print(f"[heatmap] Saved to {args.heatmap_out}\n")
        except Exception as exc:
            print(f"[warn] Heatmap failed: {exc}")

    layer_idx = int(args.layer.split(".")[-1])

    # Base generations
    print("[base] Prompt A:")
    base_a = generate(tok, llm, args.prompt_a, args.max_new_tokens, device)
    print(base_a, "\n")
    print("[base] Prompt B:")
    base_b = generate(tok, llm, args.prompt_b, args.max_new_tokens, device)
    print(base_b, "\n")

    # Compute hidden-state delta between target tokens.
    input_ids_a, attn_mask_a = encode_prompt(tok, args.prompt_a, device)
    input_ids_b, attn_mask_b = encode_prompt(tok, args.prompt_b, device)
    indices_a, token_a = find_token_indices(tok, input_ids_a[0].tolist(), args.target_a)
    indices_b, token_b = find_token_indices(tok, input_ids_b[0].tolist(), args.target_b)
    if not indices_a or not indices_b:
        raise RuntimeError("Could not locate target token span in one of the prompts.")

    hs_a = hidden_states_at_layer(llm, input_ids_a, attn_mask_a, layer_idx)[0]
    hs_b = hidden_states_at_layer(llm, input_ids_b, attn_mask_b, layer_idx)[0]
    vec_a = hs_a[indices_a].mean(dim=0)
    vec_b = hs_b[indices_b].mean(dim=0)
    delta = vec_a - vec_b

    print(f"[patch] target-a token: {token_a} indices: {indices_a}")
    print(f"[patch] target-b token: {token_b} indices: {indices_b}")

    cfg = PatchConfig(layer=args.layer, token_indices=indices_b, delta=delta)
    handle = patch_at_tokens(llm, cfg)
    try:
        print("[patched] Prompt B with A-B delta:")
        patched_b = generate(tok, llm, args.prompt_b, args.max_new_tokens, device)
        print(patched_b, "\n")
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
