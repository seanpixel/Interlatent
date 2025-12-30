from __future__ import annotations

import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from interlatent.api import LatentDB


class ActivationVectorDataset(Dataset):
    """
    Build per-step activation vectors for a single layer.

    Each sample is a tuple of (x, context) where x is a float tensor shaped
    (num_channels,) aggregated over spatial dims and context is the first
    stored context for that (run_id, step) pair. This is a light-weight
    foundation for autoencoders / SAEs on LLM activations where only a single
    stream is available (no pre/post pair).
    """

    def __init__(
        self,
        db: LatentDB,
        layer: str,
        *,
        limit: int | None = None,
    ):
        x, meta = db.fetch_vectors(layer=layer, limit=limit)
        if x.size:
            channel_order = list(range(x.shape[1]))
            samples: list[tuple[torch.Tensor, Dict]] = []
            ctx_by_key: Dict[Tuple[str, int, int] | Tuple[str, int], Dict] = {}
            vec_by_key: Dict[Tuple[str, int, int] | Tuple[str, int], torch.Tensor] = {}

            steps = meta.get("step")
            prompt_index = meta.get("prompt_index")
            token_index = meta.get("token_index")
            run_ids = meta.get("run_id") or [None] * x.shape[0]
            contexts = meta.get("context")
            context_ids = meta.get("context_id")
            context_table = meta.get("contexts")

            for i in range(x.shape[0]):
                run_id = run_ids[i] if isinstance(run_ids, list) else run_ids[i]
                pi = int(prompt_index[i]) if prompt_index is not None and prompt_index[i] >= 0 else None
                ti = int(token_index[i]) if token_index is not None and token_index[i] >= 0 else None
                if pi is not None and ti is not None:
                    key = (run_id, pi, ti)
                else:
                    step = int(steps[i]) if steps is not None else i
                    key = (run_id, step)

                ctx = {}
                if contexts is not None:
                    raw = contexts[i]
                    if isinstance(raw, str) and raw:
                        ctx = json.loads(raw)
                    elif isinstance(raw, dict):
                        ctx = raw
                elif context_ids is not None and context_table is not None:
                    cid = int(context_ids[i])
                    if cid >= 0:
                        ctx = json.loads(context_table[cid])

                vec_by_key[key] = torch.tensor(x[i], dtype=torch.float32)
                if key not in ctx_by_key:
                    ctx_by_key[key] = ctx

            for key, vec in vec_by_key.items():
                samples.append((vec, ctx_by_key.get(key, {})))

            if not samples:
                raise ValueError(f"No activation vectors could be built for layer '{layer}'")

            self.samples = samples
            self.in_dim = len(channel_order)
            self.channel_order = channel_order
            return

        events = db.fetch_activations(layer=layer, limit=limit)
        if not events:
            raise ValueError(f"No activations found for layer '{layer}'")

        def key_for(ev):
            # Prefer grouping by prompt/token if present; otherwise step.
            if ev.prompt_index is not None and ev.token_index is not None:
                return (ev.run_id, ev.prompt_index, ev.token_index)
            return (ev.run_id, ev.step)

        grouped: Dict[Tuple[str, int, int] | Tuple[str, int], Dict[int, float]] = {}
        ctx_by_key: Dict[Tuple[str, int, int] | Tuple[str, int], Dict] = {}
        channels_seen: set[int] = set()

        for ev in events:
            key = key_for(ev)
            grouped.setdefault(key, {})[ev.channel] = ev.value_sum if ev.value_sum is not None else sum(ev.tensor)
            channels_seen.add(ev.channel)
            if key not in ctx_by_key:
                ctx_by_key[key] = ev.context or {}

        channel_order = sorted(channels_seen)
        samples: list[tuple[torch.Tensor, Dict]] = []
        for key, vec_dict in grouped.items():
            vec = torch.tensor([vec_dict.get(ch, 0.0) for ch in channel_order], dtype=torch.float32)
            samples.append((vec, ctx_by_key[key]))

        if not samples:
            raise ValueError(f"No activation vectors could be built for layer '{layer}'")

        self.samples = samples
        self.in_dim = len(channel_order)
        self.channel_order = channel_order

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
