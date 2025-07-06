from __future__ import annotations
from typing import Sequence, List

import torch
from torch.utils.data import Dataset

from interlatent.api import LatentDB
from interlatent.schema import ActivationEvent

class ActivationPairDataset(Dataset):
    """
    Returns (x_pre, x_post) pairs for training a linear transcoder.
    Assumes both layers were hooked and each step recorded once per channel.
    """

    def __init__(
        self,
        db: LatentDB,
        layer_pre: str,
        layer_post: str,
        *,
        sample_limit: int | None = None,
    ):
        rows_pre = db.fetch_activations(layer=layer_pre, limit=sample_limit)
        if not rows_pre:
            raise ValueError(f"No activations for {layer_pre}")

        rows_post = db.fetch_activations(layer=layer_post, limit=sample_limit)
        if not rows_post:
            raise ValueError(f"No activations for {layer_post}")

        primary_run = rows_pre[0].run_id        
        rows_pre  = [r for r in rows_pre  if r.run_id == primary_run]
        rows_post = [r for r in rows_post if r.run_id == primary_run]

        # group by (step) to build full-channel vectors
        def to_vectors(rows):
            buf, last_step, out = [], None, []
            for ev in rows:
                if last_step is None:
                    last_step = ev.step
                if ev.step != last_step:
                    out.append(torch.tensor(buf, dtype=torch.float32))
                    buf, last_step = [], ev.step
                buf.append(sum(ev.tensor))     # channel scalar
            if buf:
                out.append(torch.tensor(buf, dtype=torch.float32))
            return out

        vec_pre, vec_post = to_vectors(rows_pre), to_vectors(rows_post)
        assert len(vec_pre) == len(vec_post), "step alignment mismatch"

        self.samples = list(zip(vec_pre, vec_post))
        self.in_dim  = len(vec_pre[0])
        self.out_dim = len(vec_post[0])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
