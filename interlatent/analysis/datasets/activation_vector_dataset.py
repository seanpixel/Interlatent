from __future__ import annotations

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
        events = db.fetch_activations(layer=layer, limit=limit)
        if not events:
            raise ValueError(f"No activations found for layer '{layer}'")

        grouped: Dict[Tuple[str, int], Dict[int, float]] = {}
        ctx_by_key: Dict[Tuple[str, int], Dict] = {}
        channels_seen: set[int] = set()

        for ev in events:
            key = (ev.run_id, ev.step)
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
