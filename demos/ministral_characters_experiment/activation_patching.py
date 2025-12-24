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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--layer", type=str, default="llm.layer.30")
    ap.add_argument("--latent-db", type=str, default="latents_snitch_report.db")
    ap.add_argument("--latent-layer", type=str, default="latent_sae:llm.layer.30")
    ap.add_argument("--sae", type=Path, default=Path("artifacts/sae_llm_layer_30_20251217_070930.pth"))
    ap.add_argument("--no-collect", action="store_true", help="Skip collecting a clean DB for prompts A/B.")
    ap.add_argument("--topk-diff", type=int, default=20)
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
            conn = _open_db(args.latent_db)
            table = latent_diff(
                conn,
                conn,
                layer=args.latent_layer,
                token_like_a=args.target_a,
                token_like_b=args.target_b,
                top=args.topk_diff,
            )
            print("[sae diff] Top SAE channel diffs for target tokens:")
            print(table, "\n")
        except Exception as exc:
            print(f"[warn] SAE diff failed: {exc}")

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
    target_ids_a = tok(args.target_a, add_special_tokens=False)["input_ids"]
    target_ids_b = tok(args.target_b, add_special_tokens=False)["input_ids"]
    indices_a = find_subsequence(input_ids_a[0].tolist(), target_ids_a)
    indices_b = find_subsequence(input_ids_b[0].tolist(), target_ids_b)
    if not indices_a or not indices_b:
        raise RuntimeError("Could not locate target token span in one of the prompts.")

    hs_a = hidden_states_at_layer(llm, input_ids_a, attn_mask_a, layer_idx)[0]
    hs_b = hidden_states_at_layer(llm, input_ids_b, attn_mask_b, layer_idx)[0]
    vec_a = hs_a[indices_a].mean(dim=0)
    vec_b = hs_b[indices_b].mean(dim=0)
    delta = vec_a - vec_b

    print(f"[patch] target-a indices: {indices_a}")
    print(f"[patch] target-b indices: {indices_b}")

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
