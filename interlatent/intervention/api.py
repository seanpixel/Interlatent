"""
High-level API for latent interventions.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch.nn as nn

from .latents import LatentBasis, load_sae_basis, load_transcoder_basis
from .patch import InterventionConfig, patch_layer, prompt_only_mask


def _resolve_layer_module(model: nn.Module, base_layer: str):
    """
    Resolve a HF layer module given a layer name like "llm.layer.30".
    """
    if not base_layer.startswith("llm.layer."):
        raise ValueError(f"Expected base_layer like 'llm.layer.N', got {base_layer}")
    idx = int(base_layer.split(".")[-1])
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "layer"):  # some HF models expose .layer
        layers = model.layer
    else:
        raise ValueError("Could not locate layers on model (expected .model.layers, .layers, or .layer)")
    try:
        return layers[idx]
    except Exception as exc:
        raise ValueError(f"Layer index {idx} out of range") from exc


@dataclass
class LatentIntervention:
    basis: LatentBasis

    @classmethod
    def load_sae(cls, artifact: str | Path, base_layer: str) -> "LatentIntervention":
        basis = load_sae_basis(Path(artifact), base_layer)
        return cls(basis=basis)

    @classmethod
    def load_transcoder(cls, artifact: str | Path, base_layer: str) -> "LatentIntervention":
        basis = load_transcoder_basis(Path(artifact), base_layer)
        return cls(basis=basis)

    def get_vector(self, channels: Sequence[int], scales: Sequence[float] | float = 1.0):
        """Return a composed intervention vector in base hidden space."""
        return self.basis.composed_vector(channels, scales)

    @contextlib.contextmanager
    def patch_model(
        self,
        model: nn.Module,
        *,
        channels: Sequence[int],
        scales: Sequence[float] | float = 1.0,
        prompt_only: bool = False,
    ):
        """
        Context manager that patches the underlying HF model, adding the
        intervention vector to the specified base layer hidden states.

        Args:
          channels: latent indices to activate
          scales: scalar or list of scales per channel
          prompt_only: if True, apply only to prompt tokens (attention_mask>0)
        """
        layer_module = _resolve_layer_module(model, self.basis.base_layer)
        mask_fn = prompt_only_mask if prompt_only else None
        cfg = InterventionConfig(
            basis=self.basis,
            channels=channels,
            scales=scales if isinstance(scales, (list, tuple)) else [scales] * len(channels),
            mask_fn=mask_fn or (lambda hs, attn: hs.new_ones(hs.shape[:2] + (1,))),
        )
        with patch_layer(model, layer_module, cfg) as patched:
            yield patched
