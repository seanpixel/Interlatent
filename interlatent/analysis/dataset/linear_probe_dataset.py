from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from interlatent.api import LatentDB


class LinearProbeDataset(Dataset):
    """
    Builds (activation_vector, target) pairs for a given layer using a metric key
    stored in ActivationEvent.context["metrics"] or directly in context.
    """

    def __init__(
        self,
        db: LatentDB,
        layer: str,
        target_key: str,
        *,
        limit: int | None = None,
    ):
        events = db.fetch_activations(layer=layer, limit=limit)
        if not events:
            raise ValueError(f"No activations found for layer '{layer}'")

        grouped: Dict[Tuple[str, int], Dict[int, float]] = {}
        targets: Dict[Tuple[str, int], float] = {}
        channels_seen: set[int] = set()

        for ev in events:
            key = (ev.run_id, ev.step)
            val = ev.value_sum if ev.value_sum is not None else sum(ev.tensor)
            grouped.setdefault(key, {})[ev.channel] = val
            channels_seen.add(ev.channel)

            metrics = (ev.context or {}).get("metrics", {})
            tgt = metrics.get(target_key)
            if tgt is None:
                tgt = (ev.context or {}).get(target_key)
            if tgt is not None:
                targets[key] = float(tgt)

        channel_order = sorted(channels_seen)
        samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key, vec_dict in grouped.items():
            if key not in targets:
                continue  # skip steps without target
            x = torch.tensor([vec_dict.get(ch, 0.0) for ch in channel_order], dtype=torch.float32)
            y = torch.tensor(targets[key], dtype=torch.float32)
            samples.append((x, y))

        if not samples:
            raise ValueError(
                f"No targets found for '{target_key}' in context metrics; cannot build probe dataset."
            )

        self.samples = samples
        self.in_dim = len(channel_order)
        self.channel_order = channel_order

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
