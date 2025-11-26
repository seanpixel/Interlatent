from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence


@dataclass
class PromptExample:
    text: str
    label: Any | None = None
    meta: dict[str, Any] | None = None


class PromptDataset:
    """
    Lightweight helper to manage prompt sets with labels/metadata for LLM
    interpretability runs. It produces both a prompt list and a callable that
    attaches prompt-level labels to each token via LLMCollector's
    `prompt_context_fn` and `token_metrics_fn`.
    """

    def __init__(self, examples: Sequence[PromptExample]):
        self.examples = list(examples)

    # ----------------------- construction helpers -----------------------
    @classmethod
    def from_pairs(cls, texts: Sequence[str], labels: Sequence[Any] | None = None):
        labels = labels or [None] * len(texts)
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        return cls([PromptExample(t, l) for t, l in zip(texts, labels)])

    @classmethod
    def from_jsonl(cls, path: str | Path, text_key: str = "text", label_key: str = "label"):
        examples: list[PromptExample] = []
        with Path(path).open() as f:
            for line in f:
                obj = json.loads(line)
                examples.append(
                    PromptExample(
                        text=obj[text_key],
                        label=obj.get(label_key),
                        meta={k: v for k, v in obj.items() if k not in (text_key, label_key)},
                    )
                )
        return cls(examples)

    @classmethod
    def from_csv(cls, path: str | Path, text_col: str = "text", label_col: str = "label"):
        examples: list[PromptExample] = []
        with Path(path).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                examples.append(
                    PromptExample(
                        text=row[text_col],
                        label=row.get(label_col),
                        meta={k: v for k, v in row.items() if k not in (text_col, label_col)},
                    )
                )
        return cls(examples)

    # ---------------------------- outputs --------------------------------
    @property
    def texts(self) -> list[str]:
        return [ex.text for ex in self.examples]

    def prompt_context_fn(self) -> Callable[[str, int], dict[str, Any]]:
        """
        Returns a function suitable for LLMCollector(prompt_context_fn=...)
        that adds prompt-level labels/metadata into each token context.
        """
        def fn(prompt_text: str, prompt_idx: int):
            ex = self.examples[prompt_idx]
            ctx = {"prompt_label": ex.label}
            if ex.meta:
                ctx["prompt_meta"] = ex.meta
            return ctx

        return fn

    def token_metrics_fn(self, metric_name: str = "prompt_label") -> Callable[..., dict[str, float]]:
        """
        Returns a token_metrics_fn that injects the prompt label as a metric
        for every token, so probe datasets can target it directly.
        """
        def fn(prompt_idx: int, **_):
            ex = self.examples[prompt_idx]
            if ex.label is None:
                return {}
            try:
                val = float(ex.label)
            except Exception:
                # try integer categories if not numeric
                val = float(hash(ex.label) % 1_000_000)
            return {metric_name: val}

        return fn
