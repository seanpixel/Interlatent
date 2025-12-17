"""
Build a character-conditioned moral dilemma prompt set from kellycyy/daily_dilemmas.
Filters for dilemmas tagged with "honesty" in values_aggregated before sampling.

Usage:
  PYTHONPATH=. python scripts/demos/ministral/character_ablations/build_dataset.py \
    --output data/character_dilemmas.csv --num 20

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
    0: ch_1,
    1: ch_2,
    2: ch_3,
    3: ch_4,
}
TARGET_VALUE = "honesty"


def pick_text(example: dict) -> Tuple[str, str]:
    """
    Use dilemma_situation as the base prompt; fall back to a structured summary
    if it's missing.
    """
    if example.get("dilemma_situation"):
        return str(example["dilemma_situation"]), "dilemma_situation"

    raise KeyError("dilemma_situation missing; expected field not found.")


def has_target_value(example: dict, target: str) -> bool:
    values = example.get("values_aggregated")
    if not values:
        return False

    # values_aggregated is a list of strings; fall back to substring search to be robust.
    if isinstance(values, str):
        return target in values.lower()

    try:
        return any(target in str(val).lower() for val in values)
    except TypeError:
        return False


def rewrite_prompt(base_prompt: str, character_text: str, use_api: bool) -> str:
    """
    Use the Mistral API to rewrite the prompt in the character style.
    Falls back to a simple templated prefix if the API key is absent or errors.
    """
    template = (
        "Rewrite the following prompt in the style of the character below without changing the meaning. Output only the rewritten prompt.\n"
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
    candidate_indices = [idx for idx, ex in enumerate(ds) if has_target_value(ex, TARGET_VALUE)]
    if not candidate_indices:
        raise ValueError(f"No examples in split '{split}' contained {TARGET_VALUE!r} in values_aggregated.")
    print(f"Filtered {len(candidate_indices)} examples containing '{TARGET_VALUE}' in values_aggregated.")
    n = min(n, len(candidate_indices))
    indices = rng.sample(candidate_indices, n)

    rows = []
    for idx in indices:
        ex = ds[int(idx)]
        base_text, field_used = pick_text(ex)
        rewrites = []
        for label, character in CHARACTERS.items():
            # Respect ~1 rps API rate limit
            if use_api and os.environ.get("MISTRAL_API_KEY"):
                await asyncio.sleep(1.05)
            rewrites.append(await rewrite_prompt_async(base_text, character, use_api=use_api))

        for (label, _), rewritten in zip(CHARACTERS.items(), rewrites):
            rows.append(
                {
                    "text": rewritten,
                    "label": label,
                    "character": label,
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
    ap.add_argument("--num", type=int, default=20, help="Number of dilemmas to sample")
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
