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


def _get_attr_path(obj, path: Sequence[str]):
    cur = obj
    for p in path:
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _resolve_layer_module(model: nn.Module, base_layer: str):
    """
    Resolve a HF layer module given a layer name like "llm.layer.30".
    Includes a few common layouts and a fallback search over ModuleList/Sequential.
    """
    if not base_layer.startswith("llm.layer."):
        raise ValueError(f"Expected base_layer like 'llm.layer.N', got {base_layer}")
    idx = int(base_layer.split(".")[-1])

    candidate_paths = [
        ("model.layers", ["model", "layers"]),
        ("model.language_model.layers", ["model", "language_model", "layers"]),
        ("language_model.layers", ["language_model", "layers"]),
        ("layers", ["layers"]),
        ("layer", ["layer"]),
        ("model.decoder.layers", ["model", "decoder", "layers"]),
        ("decoder.layers", ["decoder", "layers"]),
        ("encoder.layers", ["encoder", "layers"]),
        ("transformer.h", ["transformer", "h"]),
    ]
    for name, path in candidate_paths:
        layers = _get_attr_path(model, path)
        if layers is not None:
            try:
                return layers[idx]
            except Exception:
                continue

    # Fallback: scan attributes for a ModuleList/Sequential with enough blocks.
    for attr_name in dir(model):
        if attr_name.startswith("_"):
            continue
        val = getattr(model, attr_name, None)
        if isinstance(val, (nn.ModuleList, nn.Sequential)) and len(val) > idx:
            return val[idx]

    raise ValueError(
        "Could not locate layers on model (tried common HF layouts and ModuleList/Sequential fallback)"
    )


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
