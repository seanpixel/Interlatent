"""
Build a character-conditioned moral dilemma prompt set from kellycyy/daily_dilemmas.

Usage:
  PYTHONPATH=. python scripts/demos/ministral/character_ablations/build_dataset.py \
    --output data/character_dilemmas.csv --num 1

Requires: datasets (for HF loading), MISTRAL_API_KEY if using the default
rewriting via the Mistral API. If the key is missing, we fall back to a simple
templated rewrite so the dataset still materializes.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
from pathlib import Path
from typing import Tuple

from datasets import load_dataset

from interlatent.llm.prompt_dataset import PromptDataset

from characters import ch_1, ch_2, ch_3, ch_4
from utils import generate


CHARACTERS = {
    "ch1": ch_1,
    "ch2": ch_2,
    "ch3": ch_3,
    "ch4": ch_4,
}


def pick_text(example: dict) -> Tuple[str, str]:
    """
    Use dilemma_situation as the base prompt; fall back to a structured summary
    if it's missing.
    """
    if example.get("dilemma_situation"):
        return str(example["dilemma_situation"]), "dilemma_situation"

    parts = []
    for key in [
        "basic_situation",
        "action_type",
        "action",
        "negative_consequence",
        "values_aggregated",
        "topic_group",
    ]:
        val = example.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            val = ", ".join(map(str, val))
        parts.append(f"{key}: {val}")

    if parts:
        return "\n".join(parts), "fallback_structured"

    # ultimate fallback: any string field
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            return v, k
    raise KeyError(f"No usable text fields in example keys {example.keys()}")


def rewrite_prompt(base_prompt: str, character_text: str, use_api: bool) -> str:
    """
    Use the Mistral API to rewrite the prompt in the character style.
    Falls back to a simple templated prefix if the API key is absent or errors.
    """
    template = (
        "Rewrite the following prompt in the style of the character below without changing the meaning. "
        "Do not add or remove content beyond stylistic changes. Return the rewritten prompt only.\n"
        f"character:\n{character_text}\n"
        f"prompt:\n{base_prompt}"
    )
    if use_api and os.environ.get("MISTRAL_API_KEY"):
        try:
            return generate(prompt=template)
        except Exception as exc:
            print(f"[warn] generation failed ({exc}); falling back to templated output.")
    # fallback: prepend a short style note to preserve the label even without API
    character_title = character_text.splitlines()[0].strip("* ").strip()
    return f"[{character_title}] {base_prompt}"


async def rewrite_prompt_async(base_prompt: str, character_text: str, use_api: bool) -> str:
    return await asyncio.to_thread(rewrite_prompt, base_prompt, character_text, use_api)


async def build_dataset(split: str, n: int, seed: int, output: Path, use_api: bool) -> Path:
    ds = load_dataset("kellycyy/daily_dilemmas", split=split)
    rng = random.Random(seed)
    n = min(n, len(ds))
    indices = rng.sample(range(len(ds)), n)

    rows = []
    for idx in indices:
        ex = ds[int(idx)]
        base_text, field_used = pick_text(ex)
        tasks = []
        for label, character in CHARACTERS.items():
            tasks.append(
                rewrite_prompt_async(base_text, character, use_api=use_api)
            )
        rewrites = await asyncio.gather(*tasks)

        for (label, _), rewritten in zip(CHARACTERS.items(), rewrites):
            rows.append(
                {
                    "text": rewritten,
                    "label": label,
                    "character": label,
                    "dilemma_idx": ex.get("dilemma_idx", idx),
                    "idx": ex.get("idx", idx),
                    "action_type": ex.get("action_type"),
                    "action": ex.get("action"),
                    "negative_consequence": ex.get("negative_consequence"),
                    "values_aggregated": ex.get("values_aggregated"),
                    "topic": ex.get("topic"),
                    "topic_group": ex.get("topic_group"),
                    "source_field": field_used,
                    "source_prompt": base_text,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "label",
                "character",
                "dilemma_idx",
                "idx",
                "action_type",
                "action",
                "negative_consequence",
                "values_aggregated",
                "topic",
                "topic_group",
                "source_field",
                "source_prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Sanity check: can we load with PromptDataset
    _ = PromptDataset.from_csv(output, text_col="text", label_col="label")
    print(f"Wrote {len(rows)} rows to {output} (PromptDataset load OK).")
    return output


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/character_dilemmas.csv"))
    ap.add_argument("--num", type=int, default=1, help="Number of dilemmas to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no_api",
        action="store_true",
        help="Skip Mistral API rewrite and use a simple templated style prefix instead.",
    )
    ap.add_argument("--split", type=str, default="test")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(build_dataset(args.split, args.num, args.seed, args.output, use_api=not args.no_api))
