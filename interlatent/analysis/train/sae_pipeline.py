from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from interlatent.analysis.dataset import ActivationVectorDataset
from interlatent.analysis.train.sae_trainer import SAETrainer
from interlatent.schema import ActivationEvent, Artifact


class SAEPipeline:
    """
    Train a sparse autoencoder on a single layer of activations and
    back-fill latent activations into the DB under `latent_sae:{layer}`.
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

    def run(self):
        ds = ActivationVectorDataset(self.db, self.layer)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        trainer = SAETrainer(ds.in_dim, self.k)
        model = trainer.train(loader, epochs=self.epochs)

        self._save_artifact(model)
        self._backfill_latents(model.encoder)
        return model

    # ------------------------------------------------------------------ helpers
    def _save_artifact(self, model: torch.nn.Module):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        tag = self.layer.replace(".", "_")
        path = self.artifacts_dir / f"sae_{tag}_{ts}.pth"

        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "meta": {"layer": self.layer, "k": self.k, "epochs": self.epochs},
            },
            path,
        )

        self.db._store.write_artifact(
            Artifact(
                kind="sae",
                path=str(path),
                meta={"layer": self.layer, "k": self.k, "epochs": self.epochs},
            )
        )

    def _backfill_latents(self, encoder: torch.nn.Module):
        latent_layer = f"latent_sae:{self.layer}"
        encoder.eval()

        events = self.db.fetch_activations(layer=self.layer)
        if not events:
            raise RuntimeError(f"No activations found for layer '{self.layer}'")

        grouped: Dict[Tuple[str, int, int] | Tuple[str, int], Dict[int, float]] = {}
        ctx_by_key: Dict[Tuple[str, int, int] | Tuple[str, int], Dict] = {}
        meta_by_key: Dict[Tuple[str, int, int] | Tuple[str, int], Dict[str, object]] = {}
        def key_for(ev):
            if ev.prompt_index is not None and ev.token_index is not None:
                return (ev.run_id, ev.prompt_index, ev.token_index)
            return (ev.run_id, ev.step)
        for ev in events:
            key = key_for(ev)
            grouped.setdefault(key, {})[ev.channel] = ev.value_sum or sum(ev.tensor)
            ctx_by_key.setdefault(key, ev.context or {})
            if key not in meta_by_key:
                meta_by_key[key] = {
                    "prompt": ev.prompt,
                    "prompt_index": ev.prompt_index,
                    "token_index": ev.token_index,
                    "token": ev.token,
                }

        with torch.no_grad():
            for key, vec_dict in grouped.items():
                run_id = key[0]
                step = key[1] if len(key) == 2 else key[1] * 10_000 + key[2]
                x = torch.tensor([vec_dict[i] for i in sorted(vec_dict)], dtype=torch.float32)
                z = encoder(x.unsqueeze(0)).squeeze(0)
                for idx, val in enumerate(z):
                    self.db.write_event(
                        ActivationEvent(
                            run_id=run_id,
                            step=step,
                            layer=latent_layer,
                            channel=idx,
                            tensor=[float(val)],
                            prompt=meta_by_key[key]["prompt"],
                            prompt_index=meta_by_key[key]["prompt_index"],
                            token_index=meta_by_key[key]["token_index"],
                            token=meta_by_key[key]["token"],
                            context=ctx_by_key[key],
                            value_sum=float(val),
                            value_sq_sum=float(val * val),
                        )
                    )
        self.db.flush()
