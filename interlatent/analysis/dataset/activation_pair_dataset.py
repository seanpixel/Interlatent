from __future__ import annotations
from typing import Sequence, List

import torch
from torch.utils.data import Dataset

from interlatent.api import LatentDB


class ActivationPairDataset(Dataset):
    """
    Returns (x_pre, x_post) for a single layer.
    Assumes you logged with names  {layer}:pre  and  {layer}:post
    """

    def __init__(self, db: LatentDB, layer: str, *, limit: int | None = None):
        pre_tag, post_tag = f"{layer}:pre", f"{layer}:post"

        x_pre, meta_pre = db.fetch_vectors(layer=pre_tag, limit=limit)
        x_post, meta_post = db.fetch_vectors(layer=post_tag, limit=limit)
        if x_pre.size and x_post.size:
            run_ids_pre = meta_pre.get("run_id") or []
            run_ids_post = meta_post.get("run_id") or []
            run_id = run_ids_pre[0] if run_ids_pre else None
            if run_id is not None:
                mask_pre = [i for i, rid in enumerate(run_ids_pre) if rid == run_id]
                mask_post = [i for i, rid in enumerate(run_ids_post) if rid == run_id]
                x_pre = x_pre[mask_pre] if mask_pre else x_pre
                x_post = x_post[mask_post] if mask_post else x_post
            n = min(x_pre.shape[0], x_post.shape[0])
            self.samples = list(zip(
                [torch.tensor(x_pre[i], dtype=torch.float32) for i in range(n)],
                [torch.tensor(x_post[i], dtype=torch.float32) for i in range(n)],
            ))
            self.in_dim = x_pre.shape[1]
            self.out_dim = x_post.shape[1]
            return

        rows_pre = db.fetch_activations(layer=pre_tag, limit=limit)
        rows_post = db.fetch_activations(layer=post_tag, limit=limit)
        if not rows_pre or not rows_post:
            raise ValueError(f"Missing activations for '{layer}'")

        run_id = rows_pre[0].run_id  # keep a single run
        rows_pre = [r for r in rows_pre if r.run_id == run_id]
        rows_post = [r for r in rows_post if r.run_id == run_id]

        self.samples = list(zip(*map(self._pack, (rows_pre, rows_post))))
        self.in_dim, self.out_dim = map(len, self.samples[0])

    # ----------------------------------------------------------------------
    @staticmethod
    def _pack(rows):
        vecs, buf, step_prev = [], [], None
        for ev in rows:
            if step_prev is None:
                step_prev = ev.step
            if ev.step != step_prev:
                vecs.append(torch.tensor(buf, dtype=torch.float32))
                buf, step_prev = [], ev.step
            buf.append(sum(ev.tensor))  # scalar per channel
        if buf:
            vecs.append(torch.tensor(buf, dtype=torch.float32))
        return vecs

    # standard Dataset API
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
