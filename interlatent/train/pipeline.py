# interlatent/train/pipeline.py
from __future__ import annotations

import datetime as _dt
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from interlatent.schema import ActivationEvent, Artifact
from interlatent.train.dataset import ActivationPairDataset
from interlatent.train.trainer import TranscoderTrainer


class TranscoderPipeline:
    """
    Learn a sparse bottleneck for ONE layer, then

      1.  Saves the encoder/decoder weights to disk
          and registers an `Artifact` row in the DB.
      2.  Re-computes latent activations for *every* step
          of the original run and stores them under the
          synthetic layer name  `latent:{layer}` so they
          participate in stats / correlations like any
          other channel.
    """

    def __init__(
        self,
        db,
        layer: str,
        *,
        k: int = 32,
        epochs: int = 5,
        artifacts_dir: str | os.PathLike = "artifacts",
    ):
        self.db = db
        self.layer = layer
        self.k = k
        self.epochs = epochs
        self.artifacts_dir = Path(artifacts_dir)

    # ------------------------------------------------------------------ public

    def run(self):
        # ---- 0  fetch dataset ------------------------------------------------
        ds = ActivationPairDataset(self.db, self.layer)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        # ---- 1  train sparse AE ---------------------------------------------
        trainer = TranscoderTrainer(ds.in_dim, ds.out_dim, self.k)
        trainer.train(loader, epochs=self.epochs)

        # ---- 2  persist weights ---------------------------------------------
        self._save_artifact(trainer)

        # ---- 3  back-fill latent activations --------------------------------
        self._backfill_latents(trainer.T)

        return trainer

    # ------------------------------------------------------------------ helpers

    def _save_artifact(self, trainer):
        """
        Dump encoder & decoder weights to a .pth file and register in DB.
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        tag = self.layer.replace(".", "_")
        path = self.artifacts_dir / f"transcoder_{tag}_{ts}.pth"

        torch.save(
            {
                "encoder": trainer.T.state_dict(),
                "decoder": trainer.R.state_dict(),
                "meta": {"layer": self.layer, "k": self.k, "epochs": self.epochs},
            },
            path,
        )

        # LatentDB facade doesn’t expose write_artifact directly,
        # so we go through the underlying backend.
        self.db._store.write_artifact(
            Artifact(
                kind="transcoder",
                path=str(path),
                meta={"layer": self.layer, "k": self.k, "epochs": self.epochs},
            )
        )

    def _backfill_latents(self, encoder: torch.nn.Module):
        """
        For every (run_id, step) pair in the *post* activations of the
        target layer, run the encoder, then write one ActivationEvent
        per latent channel with identical context.
        """
        latent_layer = f"latent:{self.layer}"
        encoder.eval()

        # 1. pull original pre activations
        events = self.db.fetch_activations(layer=f"{self.layer}:pre")
        if not events:  # fallback to bare layer name
            events = self.db.fetch_activations(layer=self.layer)
        if not events:
            raise RuntimeError(
                f"No activations found for layer '{self.layer}'. "
                "Did you log with PrePostHookCtx?"
            )
        
        # 2. Grab metrics from the post activations
        ctx_events = self.db.fetch_activations(layer=f"{self.layer}:post")
        ctx_by_key = {
            (ev.run_id, ev.step): ev.context
            for ev in ctx_events
            if ev.context.get("metrics")          # ensure metrics present
        }

        # 3. group by (run_id, step)   →   {channel: scalar_sum}
        grouped: Dict[Tuple[str, int], Dict[int, float]] = defaultdict(dict)
        ctx_by_key: Dict[Tuple[str, int], Dict] = {}
        for ev in events:
            key = (ev.run_id, ev.step)
            grouped[key][ev.channel] = ev.value_sum or sum(ev.tensor)
            ctx_by_key.setdefault(key, ev.context or {})

        # 4. push latent events
        with torch.no_grad():
            for (run_id, step), vec_dict in grouped.items():
                # ordered vector by channel idx
                x = torch.tensor(
                    [vec_dict[i] for i in sorted(vec_dict)], dtype=torch.float32
                )
                z = encoder(x.unsqueeze(0)).squeeze(0)  # (k,)

                for idx, val in enumerate(z):
                    self.db.write_event(
                        ActivationEvent(
                            run_id=run_id,
                            step=step,
                            layer=latent_layer,
                            channel=idx,
                            tensor=[float(val)],
                            context=ctx_by_key[(run_id, step)],
                            value_sum=float(val),
                            value_sq_sum=float(val * val),
                        )
                    )
        self.db.flush()
