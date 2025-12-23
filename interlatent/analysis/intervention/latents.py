"""
Utilities to load latent models (SAE/Transcoder) and expose their decoded
intervention vectors in the base hidden space.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


@dataclass
class LatentBasis:
    """
    Represents a latent basis tied to a base layer.

    - `decoder`: maps latent -> base hidden space (Linear)
    - `base_layer`: e.g., "llm.layer.30"
    - `latent_dim`: number of latent channels
    - `hidden_dim`: base hidden size
    """

    decoder: nn.Linear
    base_layer: str

    @property
    def latent_dim(self) -> int:
        return self.decoder.weight.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.decoder.weight.shape[0]

    def channel_vector(self, ch: int) -> torch.Tensor:
        """Return the decoded basis vector for latent channel *ch* (1D tensor, length=hidden_dim)."""
        if ch < 0 or ch >= self.latent_dim:
            raise IndexError(f"latent channel {ch} out of range [0, {self.latent_dim})")
        # Decoder weight shape: (hidden_dim, latent_dim). Column = basis vector for latent.
        return self.decoder.weight[:, ch].detach()

    def composed_vector(self, channels: Sequence[int], scale: float | Sequence[float] = 1.0) -> torch.Tensor:
        """
        Compose a single intervention vector by summing decoder rows for the
        requested channels, scaled.
        """
        if isinstance(scale, (int, float)):
            scales = [float(scale)] * len(channels)
        else:
            scales = [float(s) for s in scale]
            if len(scales) != len(channels):
                raise ValueError("scale length must match channels length")

        vec = torch.zeros(self.hidden_dim, dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)
        for ch, s in zip(channels, scales):
            vec = vec + s * self.channel_vector(ch)
        return vec


def load_sae_basis(path: Path, base_layer: str) -> LatentBasis:
    """
    Load an SAE artifact saved by SAEPipeline (expects encoder/decoder state dicts).
    """
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected SAE checkpoint format: {type(ckpt)}")
    if "decoder" not in ckpt:
        raise ValueError("SAE checkpoint missing 'decoder' state dict")
    dec_state = ckpt["decoder"]
    weight = dec_state.get("weight")
    if weight is None:
        raise ValueError("Decoder state missing weight")
    # Torch Linear: (in_features, out_features). Decoder is latent->hidden, so
    # in_features = latent_dim, out_features = hidden_dim. Stored weight shape is
    # (out_features, in_features) = (hidden_dim, latent_dim).
    hidden_dim, latent_dim = weight.shape
    bias = "bias" in dec_state
    decoder = nn.Linear(latent_dim, hidden_dim, bias=bias)
    decoder.load_state_dict(dec_state)
    decoder.eval()
    return LatentBasis(decoder=decoder, base_layer=base_layer)


def load_transcoder_basis(path: Path, base_layer: str) -> LatentBasis:
    """
    Load a Transcoder artifact saved by TranscoderPipeline (T weight is latent->hidden).
    """
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected Transcoder checkpoint format: {type(ckpt)}")
    # Prefer decoder (R) if present; fall back to T^T.
    state = ckpt.get("decoder") or ckpt.get("R")
    if state is not None:
        weight = state.get("weight")
        if weight is None:
            raise ValueError("Transcoder decoder state missing weight")
        hidden_dim, latent_dim = weight.shape
        decoder = nn.Linear(latent_dim, hidden_dim, bias="bias" in state)
        decoder.load_state_dict(state)
    else:
        # Fallback: use encoder T weight and treat rows as latent basis.
        weight = ckpt.get("T")
        if weight is None:
            raise ValueError("Transcoder checkpoint missing T/decoder weights")
        weight_t = torch.tensor(weight) if not isinstance(weight, torch.Tensor) else weight
        latent_dim, hidden_dim = weight_t.shape
        decoder = nn.Linear(latent_dim, hidden_dim, bias=False)
        decoder.weight.data.copy_(weight_t.T)

    decoder.eval()
    return LatentBasis(decoder=decoder, base_layer=base_layer)
