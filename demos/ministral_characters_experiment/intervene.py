"""
Demo: apply latent interventions from an SAE to Ministral and compare outputs.

Usage:
RUN_MINISTRAL3=1 HF_TRUST_REMOTE_CODE=1 PYTHONPATH=. python demos/ministral_characters_experiment/intervene.py --model mistralai/Ministral-3-14B-Instruct-2512 --layer llm.layer.30 --sae artifacts/sae_llm_layer_30_YYYYMMDD_HHMMSS.pth --channels 17 31 --scale 3.0 --prompt "what's the capital of France?"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist

from interlatent.analysis.intervention import LatentIntervention


def _ensure_single_process_group():
    """
    Some Transformer loading paths (e.g., caching_allocator_warmup) expect a default
    process group. If none exists, initialize a local single-process group to avoid
    get_world_size() errors.
    """
    if not dist.is_available() or dist.is_initialized():
        return False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    try:
        dist.init_process_group("gloo", rank=0, world_size=1)
        return True
    except Exception as exc:  # pragma: no cover - best-effort safeguard
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


def generate(tok, llm, prompt: str, max_new_tokens: int, device: str) -> str:
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
    with torch.no_grad():
        out = llm.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    return tok.decode(out[0].tolist(), skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--layer", type=str, default="llm.layer.30", help="Base layer used to train the SAE.")
    ap.add_argument("--sae", type=Path, required=True, help="Path to SAE checkpoint (.pth) saved by SAEPipeline.")
    ap.add_argument("--channels", type=int, nargs="+", required=True, help="Latent channels to boost.")
    ap.add_argument("--scale", type=float, default=5.0, help="Scale to apply to each channel.")
    ap.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to test (formatted as a user message in the chat template).",
    )
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--prompt_only", action="store_true", help="Apply intervention only to prompt tokens.")
    args = ap.parse_args()

    if os.environ.get("RUN_MINISTRAL3") != "1":
        print("Set RUN_MINISTRAL3=1 to run (downloads weights); skipping.")
        return

    trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1") == "1"
    tok, llm, device = load_model_and_tokenizer(args.model, trust_remote_code)

    print("[base] Generating without intervention...")
    base_out = generate(tok, llm, args.prompt, args.max_new_tokens, device)
    print("\n[base completion]\n", base_out, "\n")

    inv = LatentIntervention.load_sae(args.sae, base_layer=args.layer)
    print(f"[intervene] Applying channels {args.channels} at scale {args.scale} on layer {args.layer}")
    with inv.patch_model(llm, channels=args.channels, scales=args.scale, prompt_only=args.prompt_only):
        patched_out = generate(tok, llm, args.prompt, args.max_new_tokens, device)

    print("\n[patched completion]\n", patched_out, "\n")


if __name__ == "__main__":
    main()
